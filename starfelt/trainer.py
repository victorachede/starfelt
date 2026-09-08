"""Starfelt Trainer SDK — wrap a standard PyTorch training loop.

Usage::

    from starfelt import Trainer

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        loss_fn=criterion,          # optional; defaults to model output if callable
        device="cuda",              # optional
        epochs=10,                  # or max_steps=
    )
    trainer.fit()

What it handles for you:
- Automatic checkpointing
- Early stopping via StarfeltCallback
- Per-epoch cost + GPU util + loss / LR telemetry
- Works standalone or under ``starfelt run`` (picks up STARFELT_RUN_ID etc.)
- Zero required config beyond the objects you already have
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Optional, Sequence

from starfelt.callbacks import StarfeltCallback
from starfelt.core.config import StarfeltConfig, load_config
from starfelt.core.cost import _runs_dir, load_history
from starfelt.core.gpu import GpuMonitor
from starfelt.core.telemetry import TelemetryHints, build_run_telemetry, size_bucket, workload_id
from starfelt.hooks import fire_checkpoint, fire_epoch_end


def _try_torch():
    try:
        import torch

        return torch
    except ImportError:
        return None


@dataclass
class EpochRecord:
    epoch: int
    loss: float
    lr: float | None
    duration_s: float
    cost_usd: float
    gpu_util: float | None
    timestamp: float


@dataclass
class TrainerResult:
    run_id: str
    epochs_completed: int
    final_loss: float | None
    duration_s: float
    cost_usd: float
    stopped_early: bool
    checkpoint_path: str | None
    epoch_history: list[dict[str, Any]]
    workload_id: str | None = None


class Trainer:
    """High-level PyTorch training wrapper used as the viral adoption path.

    Minimal call::

        Trainer(model, optimizer, train_loader, epochs=5).fit()
    """

    def __init__(
        self,
        model: Any,
        optimizer: Any,
        train_loader: Iterable,
        *,
        loss_fn: Callable[..., Any] | None = None,
        device: str | None = None,
        epochs: int | None = None,
        max_steps: int | None = None,
        val_loader: Iterable | None = None,
        scheduler: Any | None = None,
        callback: StarfeltCallback | None = None,
        checkpoint_dir: str | Path | None = None,
        checkpoint_every_epochs: int = 1,
        project_config: StarfeltConfig | None = None,
        run_id: str | None = None,
        log_every: int = 1,
        amp: bool = False,
        on_epoch_end: Callable[[int, float, float], None] | None = None,
    ) -> None:
        self.torch = _try_torch()
        if self.torch is None:
            raise ImportError(
                "Trainer requires PyTorch. Install with: pip install torch"
            )

        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.loss_fn = loss_fn
        self.val_loader = val_loader
        self.scheduler = scheduler
        self.epochs = epochs
        self.max_steps = max_steps
        self.checkpoint_every_epochs = max(1, checkpoint_every_epochs)
        self.log_every = max(1, log_every)
        self.amp = amp
        self.on_epoch_end = on_epoch_end

        # Device
        if device is None:
            device = "cuda" if self.torch.cuda.is_available() else "cpu"
        self.device = self.torch.device(device)
        self.model.to(self.device)

        # Config / run identity
        try:
            self.cfg = project_config or load_config()
        except Exception:
            self.cfg = StarfeltConfig()

        self.run_id = (
            run_id
            or os.environ.get("STARFELT_RUN_ID")
            or uuid.uuid4().hex
        )
        os.environ.setdefault("STARFELT_RUN_ID", self.run_id)

        # Early-stop callback
        self.callback = callback or StarfeltCallback(
            patience_steps=self.cfg.patience_steps
            if hasattr(self.cfg, "patience_steps")
            else None
        )

        # Checkpoint dir
        if checkpoint_dir is None:
            checkpoint_dir = Path.cwd() / ".starfelt" / "checkpoints" / self.run_id
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Telemetry / cost
        self.gpu = GpuMonitor()
        self.epoch_history: list[EpochRecord] = []
        self._t0 = time.time()
        self._step = 0
        self._stopped_early = False
        self._last_checkpoint: Path | None = None
        self._scaler = None
        if self.amp and self.device.type == "cuda":
            self._scaler = self.torch.cuda.amp.GradScaler()

        # Resume support
        resume_path = os.environ.get("STARFELT_RESUME_FROM")
        if resume_path and Path(resume_path).exists():
            self._load_checkpoint(Path(resume_path))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self) -> TrainerResult:
        """Run the training loop and return a rich result object."""
        if self.epochs is None and self.max_steps is None:
            self.epochs = 1

        self.model.train()
        epoch = 0
        total_epochs = self.epochs or 10_000  # safety if only max_steps

        while epoch < total_epochs:
            if self.max_steps is not None and self._step >= self.max_steps:
                break

            ep_loss, ep_lr, ep_dur, ep_gpu = self._run_epoch(epoch)
            cost_so_far = ((time.time() - self._t0) / 3600.0) * self.cfg.gpu_hour_usd

            rec = EpochRecord(
                epoch=epoch,
                loss=ep_loss,
                lr=ep_lr,
                duration_s=ep_dur,
                cost_usd=cost_so_far,
                gpu_util=ep_gpu,
                timestamp=time.time(),
            )
            self.epoch_history.append(rec)

            if self.on_epoch_end:
                try:
                    self.on_epoch_end(epoch, ep_loss, cost_so_far)
                except Exception:
                    pass
            fire_epoch_end(epoch, ep_loss, cost_so_far)

            # Checkpoint
            if (epoch + 1) % self.checkpoint_every_epochs == 0:
                path = self._save_checkpoint(epoch, ep_loss)
                fire_checkpoint(str(path))

            # Early stop (callback already may have SIGTERM'd; we also check flag)
            if self.callback.stopped or self._stopped_early:
                self._stopped_early = True
                break

            epoch += 1

        result = self._finalize(epochs_completed=len(self.epoch_history))
        return result

    def save_checkpoint(self, tag: str = "manual") -> Path:
        """Force a checkpoint write. Returns path."""
        loss = self.epoch_history[-1].loss if self.epoch_history else 0.0
        path = self.checkpoint_dir / f"ckpt_{tag}.pt"
        self._write_ckpt(path, epoch=len(self.epoch_history), loss=loss)
        self._last_checkpoint = path
        return path

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _run_epoch(self, epoch: int) -> tuple[float, float | None, float, float | None]:
        t0 = time.time()
        running_loss = 0.0
        n_batches = 0
        last_lr: float | None = None

        for batch in self.train_loader:
            if self.max_steps is not None and self._step >= self.max_steps:
                break

            loss_val = self._train_step(batch)
            running_loss += loss_val
            n_batches += 1
            self._step += 1

            # LR
            if self.optimizer.param_groups:
                last_lr = float(self.optimizer.param_groups[0].get("lr", 0.0))

            # Callback step (early stop)
            if self.callback.step(loss_val):
                self._stopped_early = True
                break

            # Optional GPU sample every few steps
            if self._step % 20 == 0:
                self.gpu.poll()

        if self.scheduler is not None:
            try:
                self.scheduler.step()
            except Exception:
                pass

        avg_loss = running_loss / max(n_batches, 1)
        duration = time.time() - t0
        gpu_util = self.gpu.last.util_pct if self.gpu.last else None

        if (epoch + 1) % self.log_every == 0:
            util_s = f"  gpu={gpu_util:.0f}%" if gpu_util is not None else ""
            lr_s = f"  lr={last_lr:.2e}" if last_lr is not None else ""
            print(
                f"[starfelt] epoch {epoch + 1}  loss={avg_loss:.4f}{lr_s}"
                f"  {duration:.1f}s{util_s}",
                flush=True,
            )

        return avg_loss, last_lr, duration, gpu_util

    def _train_step(self, batch: Any) -> float:
        self.optimizer.zero_grad(set_to_none=True)

        # Unpack common batch shapes
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            x, y = batch[0], batch[1]
        else:
            x, y = batch, None

        x = self._to_device(x)
        if y is not None:
            y = self._to_device(y)

        if self.amp and self._scaler is not None:
            with self.torch.cuda.amp.autocast():
                loss = self._compute_loss(x, y)
            self._scaler.scale(loss).backward()
            self._scaler.step(self.optimizer)
            self._scaler.update()
        else:
            loss = self._compute_loss(x, y)
            loss.backward()
            self.optimizer.step()

        return float(loss.detach().item())

    def _compute_loss(self, x: Any, y: Any) -> Any:
        if self.loss_fn is not None:
            out = self.model(x)
            return self.loss_fn(out, y)
        # Assume model returns loss when called with (x, y) or just x
        if y is not None:
            try:
                return self.model(x, y)
            except TypeError:
                out = self.model(x)
                # last-ditch: if model already returns scalar loss
                if out.ndim == 0:
                    return out
                raise
        out = self.model(x)
        if out.ndim == 0:
            return out
        raise RuntimeError(
            "Trainer: provide loss_fn= or make model(x) / model(x, y) return a scalar loss"
        )

    def _to_device(self, obj: Any) -> Any:
        if hasattr(obj, "to"):
            return obj.to(self.device)
        if isinstance(obj, (list, tuple)):
            return type(obj)(self._to_device(o) for o in obj)
        return obj

    def _save_checkpoint(self, epoch: int, loss: float) -> Path:
        path = self.checkpoint_dir / f"epoch_{epoch:04d}.pt"
        self._write_ckpt(path, epoch=epoch, loss=loss)
        # Also keep a "latest"
        latest = self.checkpoint_dir / "latest.pt"
        self._write_ckpt(latest, epoch=epoch, loss=loss)
        self._last_checkpoint = path
        return path

    def _write_ckpt(self, path: Path, *, epoch: int, loss: float) -> None:
        state = {
            "epoch": epoch,
            "step": self._step,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "loss": loss,
            "run_id": self.run_id,
            "epoch_history": [r.__dict__ for r in self.epoch_history],
        }
        if self.scheduler is not None and hasattr(self.scheduler, "state_dict"):
            state["scheduler"] = self.scheduler.state_dict()
        if self._scaler is not None:
            state["scaler"] = self._scaler.state_dict()
        self.torch.save(state, path)

    def _load_checkpoint(self, path: Path) -> None:
        ckpt = self.torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self._step = int(ckpt.get("step", 0))
        if self.scheduler is not None and "scheduler" in ckpt:
            self.scheduler.load_state_dict(ckpt["scheduler"])
        if self._scaler is not None and "scaler" in ckpt:
            self._scaler.load_state_dict(ckpt["scaler"])
        # Restore history if present
        for raw in ckpt.get("epoch_history") or []:
            self.epoch_history.append(EpochRecord(**raw))
        print(f"[starfelt] resumed from {path}  step={self._step}", flush=True)

    def _finalize(self, epochs_completed: int) -> TrainerResult:
        duration_s = time.time() - self._t0
        hours = duration_s / 3600.0
        cost = hours * self.cfg.gpu_hour_usd
        baseline = cost * self.cfg.baseline_multiplier
        final_loss = self.epoch_history[-1].loss if self.epoch_history else None

        # Build per-epoch telemetry payload
        epoch_rows = [
            {
                "epoch": r.epoch,
                "loss": r.loss,
                "lr": r.lr,
                "duration_s": r.duration_s,
                "cost_usd": r.cost_usd,
                "gpu_util": r.gpu_util,
                "ts": r.timestamp,
            }
            for r in self.epoch_history
        ]

        # Infer simple hints
        batch_size = None
        try:
            # common DataLoader attribute
            batch_size = getattr(self.train_loader, "batch_size", None)
        except Exception:
            pass

        hints = TelemetryHints(
            framework="pytorch",
            frameworks_detected=["pytorch"],
            batch_size=batch_size,
            epochs=epochs_completed,
            learning_rate=(
                float(self.optimizer.param_groups[0]["lr"])
                if self.optimizer.param_groups
                else None
            ),
            optimization_flags=["amp"] if self.amp else [],
        )

        # Model param count
        try:
            param_count = sum(p.numel() for p in self.model.parameters())
        except Exception:
            param_count = None

        model_bucket = size_bucket(param_count, kind="model")
        wl = workload_id(
            framework="pytorch",
            model_size_bucket=model_bucket,
            dataset_size_bucket=None,
            batch_size=batch_size,
            epochs=epochs_completed,
        )

        row = build_run_telemetry(
            run_id=self.run_id,
            script="starfelt.Trainer",
            duration_s=duration_s,
            cost_usd=cost,
            baseline_cost_usd=baseline,
            exit_code=0,
            hints=hints,
            model_param_count=param_count,
            gpu_name=(self.gpu.last.name if self.gpu.last else None),
            gpu_util_avg=self.gpu.average_util(),
            gpu_samples=len(self.gpu.samples),
            interrupted=None,
            extra={
                "ts": time.time(),
                "source": "trainer_sdk",
                "epochs_completed": epochs_completed,
                "stopped_early": self._stopped_early,
                "epoch_history": epoch_rows,
                "final_loss": final_loss,
                "checkpoint": str(self._last_checkpoint) if self._last_checkpoint else None,
            },
        )

        # Persist like CostTracker does
        hist_path = Path.cwd() / ".starfelt" / "history.json"
        hist = load_history()
        hist.append(row)
        hist_path.parent.mkdir(parents=True, exist_ok=True)
        hist_path.write_text(json.dumps(hist, indent=2), encoding="utf-8")
        (_runs_dir() / f"{self.run_id}.json").write_text(
            json.dumps(row, indent=2), encoding="utf-8"
        )

        return TrainerResult(
            run_id=self.run_id,
            epochs_completed=epochs_completed,
            final_loss=final_loss,
            duration_s=duration_s,
            cost_usd=cost,
            stopped_early=self._stopped_early,
            checkpoint_path=str(self._last_checkpoint) if self._last_checkpoint else None,
            epoch_history=epoch_rows,
            workload_id=wl,
        )
