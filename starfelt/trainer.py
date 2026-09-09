"""Starfelt Trainer SDK — wrap a standard PyTorch training loop.

Usage::

    from starfelt import Trainer

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        loss_fn=criterion,
        val_loader=val_loader,          # optional — enables val early-stop
        epochs=10,
        grad_accum_steps=4,             # effective larger batch
        amp=True,
    )
    result = trainer.fit()

What it handles:
- Gradient accumulation
- Eval loop + early-stop on validation loss
- Automatic checkpointing + resume (STARFELT_RESUME_FROM)
- AMP, optional DataParallel
- Per-epoch telemetry: loss, val_loss, lr, samples/sec, GPU util, memory, cost
- Works standalone or under ``starfelt run``
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from starfelt.callbacks import StarfeltCallback
from starfelt.core.config import StarfeltConfig, load_config
from starfelt.core.cost import _runs_dir, load_history
from starfelt.core.gpu import GpuMonitor
from starfelt.core.telemetry import TelemetryHints, build_run_telemetry, size_bucket, workload_id
from starfelt.hooks import fire_checkpoint, fire_epoch_end
from starfelt.notebook import StarfeltDisplay, drive_mounted, in_colab


def _try_torch():
    try:
        import torch

        return torch
    except ImportError:
        return None


def _colab_checkpoint_dir(run_id: str) -> Path | None:
    """Return a Drive-backed checkpoint dir if in Colab and Drive is mounted."""
    drive_root = Path("/content/drive/MyDrive/.starfelt/checkpoints")
    if in_colab() and drive_mounted("/content/drive/MyDrive"):
        drive_root.mkdir(parents=True, exist_ok=True)
        return drive_root / run_id
    return None


def _warn_colab_no_drive() -> None:
    """Warn once if in Colab but Drive isn't mounted."""
    if in_colab() and not drive_mounted("/content/drive/MyDrive"):
        print(
            "\n[starfelt] ⚠️  Google Drive not mounted.\n"
            "   Checkpoints will be lost if Colab disconnects.\n"
            "   Mount Drive with:\n"
            "       from starfelt.notebook import mount_drive\n"
            "       mount_drive()\n",
            flush=True,
        )


@dataclass
class EpochRecord:
    epoch: int
    loss: float
    val_loss: float | None = None
    lr: float | None = None
    duration_s: float = 0.0
    cost_usd: float = 0.0
    gpu_util: float | None = None
    samples_per_sec: float | None = None
    peak_mem_mb: float | None = None
    timestamp: float = 0.0


@dataclass
class TrainerResult:
    run_id: str
    epochs_completed: int
    final_loss: float | None
    final_val_loss: float | None
    duration_s: float
    cost_usd: float
    stopped_early: bool
    checkpoint_path: str | None
    epoch_history: list[dict[str, Any]]
    workload_id: str | None = None
    best_val_loss: float | None = None


class Trainer:
    """High-level PyTorch training wrapper — the viral adoption path.

    Minimal::

        Trainer(model, optimizer, train_loader, epochs=5).fit()

    Real fine-tune style::

        Trainer(
            model, optimizer, train_loader,
            loss_fn=criterion,
            val_loader=val_loader,
            epochs=3,
            grad_accum_steps=4,
            amp=True,
            early_stop_patience=2,
        ).fit()
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
        grad_accum_steps: int = 1,
        early_stop_patience: int | None = None,
        eval_every_epochs: int = 1,
        data_parallel: bool = False,
        on_epoch_end: Callable[[int, float, float], None] | None = None,
        notebook_display: StarfeltDisplay | None = None,
        wandb: bool = False,
        wandb_project: str | None = None,
        wandb_run_name: str | None = None,
        auto_mount_drive: bool = False,
    ) -> None:
        self.torch = _try_torch()
        if self.torch is None:
            raise ImportError("Trainer requires PyTorch. Install with: pip install torch")

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
        self.grad_accum_steps = max(1, grad_accum_steps)
        self.early_stop_patience = early_stop_patience
        self.eval_every_epochs = max(1, eval_every_epochs)
        self.on_epoch_end = on_epoch_end

        if device is None:
            device = "cuda" if self.torch.cuda.is_available() else "cpu"
        self.device = self.torch.device(device)
        self.model.to(self.device)

        if data_parallel and self.device.type == "cuda" and self.torch.cuda.device_count() > 1:
            self.model = self.torch.nn.DataParallel(self.model)
            self._is_dp = True
        else:
            self._is_dp = False

        try:
            self.cfg = project_config or load_config()
        except Exception:
            self.cfg = StarfeltConfig()

        self.run_id = run_id or os.environ.get("STARFELT_RUN_ID") or uuid.uuid4().hex
        os.environ.setdefault("STARFELT_RUN_ID", self.run_id)

        self.callback = callback or StarfeltCallback(
            patience_steps=getattr(self.cfg, "patience_steps", None)
        )

        if checkpoint_dir is None:
            checkpoint_dir = Path.cwd() / ".starfelt" / "checkpoints" / self.run_id
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.gpu = GpuMonitor()
        self.epoch_history: list[EpochRecord] = []
        self._t0 = time.time()
        self._step = 0
        self._stopped_early = False
        self._last_checkpoint: Path | None = None
        self._best_val: float | None = None
        self._val_stale = 0
        self._scaler = None
        if self.amp and self.device.type == "cuda":
            self._scaler = self.torch.cuda.amp.GradScaler()

        # Notebook / Colab setup
        self._display = notebook_display
        self._use_wandb = wandb
        self._wandb_project = wandb_project or self.cfg.project
        self._wandb_run = None
        _warn_colab_no_drive()

        if auto_mount_drive and in_colab() and not drive_mounted("/content/drive/MyDrive"):
            from starfelt.notebook import mount_drive
            mount_drive()

        # Override checkpoint_dir to Drive if in Colab and Drive is mounted
        if checkpoint_dir is None:
            colab_dir = _colab_checkpoint_dir(self.run_id)
            if colab_dir is not None:
                self.checkpoint_dir = colab_dir
                self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
                print(f"[starfelt] Checkpointing to Drive: {self.checkpoint_dir}", flush=True)

        if self._use_wandb:
            self._init_wandb(wandb_run_name)

        resume_path = os.environ.get("STARFELT_RESUME_FROM")
        if resume_path and Path(resume_path).exists():
            self._load_checkpoint(Path(resume_path))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self) -> TrainerResult:
        if self.epochs is None and self.max_steps is None:
            self.epochs = 1

        self.model.train()
        epoch = 0
        total_epochs = self.epochs or 10_000

        while epoch < total_epochs:
            if self.max_steps is not None and self._step >= self.max_steps:
                break

            train_stats = self._run_epoch(epoch)
            val_loss = None
            if self.val_loader is not None and (epoch + 1) % self.eval_every_epochs == 0:
                val_loss = self._evaluate()

            cost_so_far = ((time.time() - self._t0) / 3600.0) * self.cfg.gpu_hour_usd

            rec = EpochRecord(
                epoch=epoch,
                loss=train_stats["loss"],
                val_loss=val_loss,
                lr=train_stats["lr"],
                duration_s=train_stats["duration_s"],
                cost_usd=cost_so_far,
                gpu_util=train_stats["gpu_util"],
                samples_per_sec=train_stats["samples_per_sec"],
                peak_mem_mb=train_stats["peak_mem_mb"],
                timestamp=time.time(),
            )
            self.epoch_history.append(rec)

            if self.on_epoch_end:
                try:
                    self.on_epoch_end(epoch, train_stats["loss"], cost_so_far)
                except Exception:
                    pass
            fire_epoch_end(epoch, train_stats["loss"], cost_so_far)
            self._log_epoch(
                epoch, train_stats["loss"],
                val_loss=val_loss,
                cost_so_far=cost_so_far,
                lr=train_stats["lr"],
                gpu_util=train_stats["gpu_util"],
                samples_per_sec=train_stats["samples_per_sec"],
            )

            if (epoch + 1) % self.checkpoint_every_epochs == 0:
                path = self._save_checkpoint(epoch, train_stats["loss"])
                fire_checkpoint(str(path))

            # Val-based early stop
            if val_loss is not None and self.early_stop_patience is not None:
                if self._best_val is None or val_loss < self._best_val - 1e-4:
                    self._best_val = val_loss
                    self._val_stale = 0
                    # keep best checkpoint
                    self._save_checkpoint(epoch, train_stats["loss"], tag="best")
                else:
                    self._val_stale += 1
                    if self._val_stale >= self.early_stop_patience:
                        print(
                            f"[starfelt] early stop — val loss stalled for "
                            f"{self.early_stop_patience} evals (best={self._best_val:.4f})",
                            flush=True,
                        )
                        self._stopped_early = True
                        break

            if self.callback.stopped or self._stopped_early:
                self._stopped_early = True
                break

            epoch += 1

        return self._finalize(epochs_completed=len(self.epoch_history))

    def _init_wandb(self, run_name: str | None) -> None:
        try:
            import wandb as _wandb  # type: ignore[import]
            self._wandb_run = _wandb.init(
                project=self._wandb_project,
                name=run_name or self.run_id[:8],
                config={
                    "run_id": self.run_id,
                    "gpu_hour_usd": self.cfg.gpu_hour_usd,
                    "amp": self.amp,
                    "grad_accum_steps": self.grad_accum_steps,
                },
                reinit=True,
            )
            print(f"[starfelt] wandb run: {self._wandb_run.url}", flush=True)
        except ImportError:
            print("[starfelt] wandb not installed — skipping. pip install wandb", flush=True)
            self._use_wandb = False

    def _log_epoch(
        self,
        epoch: int,
        loss: float,
        *,
        val_loss: float | None = None,
        cost_so_far: float = 0.0,
        lr: float | None = None,
        gpu_util: float | None = None,
        samples_per_sec: float | None = None,
    ) -> None:
        if self._display is not None:
            self._display.update(
                epoch, loss,
                val_loss=val_loss,
                lr=lr,
                cost_usd=cost_so_far,
                gpu_util=gpu_util,
                samples_per_sec=samples_per_sec,
            )
        if self._use_wandb and self._wandb_run is not None:
            try:
                import wandb as _wandb  # type: ignore[import]
                log_data: dict = {
                    "epoch": epoch + 1,
                    "train/loss": loss,
                    "train/cost_usd": cost_so_far,
                }
                if val_loss is not None:
                    log_data["val/loss"] = val_loss
                if lr is not None:
                    log_data["train/lr"] = lr
                if gpu_util is not None:
                    log_data["gpu/util_pct"] = gpu_util
                if samples_per_sec is not None:
                    log_data["train/samples_per_sec"] = samples_per_sec
                _wandb.log(log_data, step=epoch + 1)
            except Exception:
                pass

    def save_checkpoint(self, tag: str = "manual") -> Path:
        loss = self.epoch_history[-1].loss if self.epoch_history else 0.0
        path = self.checkpoint_dir / f"ckpt_{tag}.pt"
        self._write_ckpt(path, epoch=len(self.epoch_history), loss=loss)
        self._last_checkpoint = path
        return path

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _run_epoch(self, epoch: int) -> dict[str, Any]:
        self.model.train()
        t0 = time.time()
        running_loss = 0.0
        n_batches = 0
        n_samples = 0
        last_lr: float | None = None
        self.optimizer.zero_grad(set_to_none=True)

        if self.device.type == "cuda":
            self.torch.cuda.reset_peak_memory_stats()

        try:
            loader_len = len(self.train_loader)  # type: ignore[arg-type]
        except TypeError:
            loader_len = None

        for batch_idx, batch in enumerate(self.train_loader):
            if self.max_steps is not None and self._step >= self.max_steps:
                break

            is_last_batch = loader_len is not None and batch_idx + 1 == loader_len
            reaches_max_steps = (
                self.max_steps is not None and self._step + 1 >= self.max_steps
            )
            loss_val, batch_size = self._train_step(
                batch,
                batch_idx,
                force_step=is_last_batch or reaches_max_steps,
            )
            running_loss += loss_val
            n_batches += 1
            n_samples += batch_size
            self._step += 1

            if self.optimizer.param_groups:
                last_lr = float(self.optimizer.param_groups[0].get("lr", 0.0))

            if self.callback.step(loss_val):
                self._stopped_early = True
                break

            if self._step % 20 == 0:
                self.gpu.poll()

        if self.scheduler is not None:
            try:
                self.scheduler.step()
            except Exception:
                pass

        duration = time.time() - t0
        avg_loss = running_loss / max(n_batches, 1)
        samples_per_sec = n_samples / duration if duration > 0 else None
        gpu_util = self.gpu.last.util_pct if self.gpu.last else None
        peak_mem = None
        if self.device.type == "cuda":
            try:
                peak_mem = self.torch.cuda.max_memory_allocated() / (1024 * 1024)
            except Exception:
                pass

        if (epoch + 1) % self.log_every == 0:
            parts = [f"[starfelt] epoch {epoch + 1}  loss={avg_loss:.4f}"]
            if last_lr is not None:
                parts.append(f"lr={last_lr:.2e}")
            parts.append(f"{duration:.1f}s")
            if samples_per_sec is not None:
                parts.append(f"{samples_per_sec:.0f} smp/s")
            if gpu_util is not None:
                parts.append(f"gpu={gpu_util:.0f}%")
            if peak_mem is not None:
                parts.append(f"mem={peak_mem:.0f}MB")
            print("  ".join(parts), flush=True)

        return {
            "loss": avg_loss,
            "lr": last_lr,
            "duration_s": duration,
            "gpu_util": gpu_util,
            "samples_per_sec": samples_per_sec,
            "peak_mem_mb": peak_mem,
        }

    def _train_step(
        self,
        batch: Any,
        batch_idx: int,
        *,
        force_step: bool = False,
    ) -> tuple[float, int]:
        x, y, batch_size = self._unpack_batch(batch)
        x = self._to_device(x)
        if y is not None:
            y = self._to_device(y)

        if self.amp and self._scaler is not None:
            with self.torch.cuda.amp.autocast():
                loss = self._compute_loss(x, y) / self.grad_accum_steps
            self._scaler.scale(loss).backward()
            if force_step or (batch_idx + 1) % self.grad_accum_steps == 0:
                self._scaler.step(self.optimizer)
                self._scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
        else:
            loss = self._compute_loss(x, y) / self.grad_accum_steps
            loss.backward()
            if force_step or (batch_idx + 1) % self.grad_accum_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

        return float(loss.detach().item() * self.grad_accum_steps), batch_size

    def _evaluate(self) -> float:
        self.model.eval()
        total = 0.0
        n = 0
        with self.torch.no_grad():
            for batch in self.val_loader:
                x, y, bs = self._unpack_batch(batch)
                x = self._to_device(x)
                if y is not None:
                    y = self._to_device(y)
                if self.amp and self.device.type == "cuda":
                    with self.torch.cuda.amp.autocast():
                        loss = self._compute_loss(x, y)
                else:
                    loss = self._compute_loss(x, y)
                total += float(loss.item()) * bs
                n += bs
        self.model.train()
        avg = total / max(n, 1)
        print(f"[starfelt]          val_loss={avg:.4f}", flush=True)
        return avg

    def _unpack_batch(self, batch: Any) -> tuple[Any, Any, int]:
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            x, y = batch[0], batch[1]
        elif isinstance(batch, dict):
            # HF-style batch
            y = batch.get("labels")
            if y is None:
                y = batch.get("label")
            x = {k: v for k, v in batch.items() if k not in ("labels", "label")}
            if len(x) == 1:
                x = next(iter(x.values()))
        else:
            x, y = batch, None

        batch_size = 1
        try:
            if hasattr(x, "size"):
                batch_size = int(x.size(0))
            elif isinstance(x, dict):
                for v in x.values():
                    if hasattr(v, "size"):
                        batch_size = int(v.size(0))
                        break
            elif isinstance(x, (list, tuple)) and x:
                batch_size = len(x)
        except Exception:
            pass
        return x, y, batch_size

    def _compute_loss(self, x: Any, y: Any) -> Any:
        if self.loss_fn is not None:
            if isinstance(x, dict):
                out = self.model(**x)
            else:
                out = self.model(x)
            # HF models often return an object with .loss
            if hasattr(out, "loss") and out.loss is not None and y is None:
                return out.loss
            if hasattr(out, "logits"):
                out = out.logits
            return self.loss_fn(out, y)
        if isinstance(x, dict):
            out = self.model(**x)
            if hasattr(out, "loss") and out.loss is not None:
                return out.loss
        if y is not None:
            try:
                return self.model(x, y)
            except TypeError:
                out = self.model(x)
                if hasattr(out, "loss"):
                    return out.loss
                if out.ndim == 0:
                    return out
                raise
        out = self.model(x)
        if hasattr(out, "loss") and out.loss is not None:
            return out.loss
        if out.ndim == 0:
            return out
        raise RuntimeError(
            "Trainer: provide loss_fn= or make model return a scalar loss / .loss"
        )

    def _to_device(self, obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: self._to_device(v) for k, v in obj.items()}
        if hasattr(obj, "to"):
            return obj.to(self.device)
        if isinstance(obj, (list, tuple)):
            return type(obj)(self._to_device(o) for o in obj)
        return obj

    def _model_state(self) -> dict:
        m = self.model.module if self._is_dp else self.model
        return m.state_dict()

    def _load_model_state(self, state: dict) -> None:
        m = self.model.module if self._is_dp else self.model
        m.load_state_dict(state)

    def _save_checkpoint(self, epoch: int, loss: float, tag: str | None = None) -> Path:
        if tag:
            path = self.checkpoint_dir / f"{tag}.pt"
        else:
            path = self.checkpoint_dir / f"epoch_{epoch:04d}.pt"
        self._write_ckpt(path, epoch=epoch, loss=loss)
        try:
            self._write_ckpt(self.checkpoint_dir / "latest.pt", epoch=epoch, loss=loss)
        except Exception:
            pass
        self._last_checkpoint = path
        return path

    def _write_ckpt(self, path: Path, *, epoch: int, loss: float) -> None:
        state = {
            "epoch": epoch,
            "step": self._step,
            "model": self._model_state(),
            "optimizer": self.optimizer.state_dict(),
            "loss": loss,
            "run_id": self.run_id,
            "best_val": self._best_val,
            "epoch_history": [asdict(r) for r in self.epoch_history],
        }
        if self.scheduler is not None and hasattr(self.scheduler, "state_dict"):
            state["scheduler"] = self.scheduler.state_dict()
        if self._scaler is not None:
            state["scaler"] = self._scaler.state_dict()
        tmp = path.with_suffix(path.suffix + ".tmp")
        self.torch.save(state, tmp)
        tmp.replace(path)

    def _load_checkpoint(self, path: Path) -> None:
        ckpt = self.torch.load(path, map_location=self.device, weights_only=False)
        self._load_model_state(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self._step = int(ckpt.get("step", 0))
        self._best_val = ckpt.get("best_val")
        if self.scheduler is not None and "scheduler" in ckpt:
            self.scheduler.load_state_dict(ckpt["scheduler"])
        if self._scaler is not None and "scaler" in ckpt:
            self._scaler.load_state_dict(ckpt["scaler"])
        for raw in ckpt.get("epoch_history") or []:
            self.epoch_history.append(EpochRecord(**raw))
        print(f"[starfelt] resumed from {path}  step={self._step}", flush=True)

    def _finalize(self, epochs_completed: int) -> TrainerResult:
        duration_s = time.time() - self._t0
        hours = duration_s / 3600.0
        cost = hours * self.cfg.gpu_hour_usd
        final_loss = self.epoch_history[-1].loss if self.epoch_history else None
        final_val = self.epoch_history[-1].val_loss if self.epoch_history else None

        epoch_rows = [asdict(r) for r in self.epoch_history]

        batch_size = getattr(self.train_loader, "batch_size", None)
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
        if self.grad_accum_steps > 1:
            hints.optimization_flags.append(f"grad_accum_{self.grad_accum_steps}")
        if self._is_dp:
            hints.optimization_flags.append("data_parallel")

        try:
            m = self.model.module if self._is_dp else self.model
            param_count = sum(p.numel() for p in m.parameters())
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
            exit_code=0,
            hints=hints,
            model_param_count=param_count,
            gpu_name=(self.gpu.last.name if self.gpu.last else None),
            gpu_util_avg=self.gpu.average_util(),
            gpu_samples=len(self.gpu.samples),
            gpu_power_avg_w=self.gpu.average_power_w(),
            gpu_energy_j=self.gpu.energy_j if self.gpu.power_samples else None,
            interrupted=None,
            extra={
                "ts": time.time(),
                "source": "trainer_sdk",
                "epochs_completed": epochs_completed,
                "stopped_early": self._stopped_early,
                "epoch_history": epoch_rows,
                "final_loss": final_loss,
                "final_val_loss": final_val,
                "best_val_loss": self._best_val,
                "checkpoint": str(self._last_checkpoint) if self._last_checkpoint else None,
                "grad_accum_steps": self.grad_accum_steps,
            },
        )

        hist_path = Path.cwd() / ".starfelt" / "history.json"
        hist = load_history()
        hist.append(row)
        hist_path.parent.mkdir(parents=True, exist_ok=True)
        hist_path.write_text(json.dumps(hist, indent=2), encoding="utf-8")
        (_runs_dir() / f"{self.run_id}.json").write_text(
            json.dumps(row, indent=2), encoding="utf-8"
        )

        if self._display is not None:
            self._display.summary()
        if self._use_wandb and self._wandb_run is not None:
            try:
                import wandb as _wandb  # type: ignore[import]
                _wandb.finish()
            except Exception:
                pass

        return TrainerResult(
            run_id=self.run_id,
            epochs_completed=epochs_completed,
            final_loss=final_loss,
            final_val_loss=final_val,
            duration_s=duration_s,
            cost_usd=cost,
            stopped_early=self._stopped_early,
            checkpoint_path=str(self._last_checkpoint) if self._last_checkpoint else None,
            epoch_history=epoch_rows,
            workload_id=wl,
            best_val_loss=self._best_val,
        )
