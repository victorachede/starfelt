"""Starfelt notebook support — live display for Colab / Jupyter.

Usage in a notebook cell::

    from starfelt.notebook import StarfeltDisplay
    from starfelt import Trainer

    display = StarfeltDisplay()
    trainer = Trainer(model, optimizer, loader, epochs=5, notebook_display=display)
    result = trainer.fit()

    # Or mount Drive first (Colab only):
    from starfelt.notebook import mount_drive
    mount_drive()
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Environment detection
# ---------------------------------------------------------------------------

def in_colab() -> bool:
    """True when running inside Google Colab."""
    return "COLAB_GPU" in os.environ or "COLAB_RELEASE_TAG" in os.environ


def in_notebook() -> bool:
    """True when running inside any Jupyter-style kernel."""
    try:
        from IPython import get_ipython
        return get_ipython() is not None
    except ImportError:
        return False


def drive_mounted(mount_point: str = "/content/drive/MyDrive") -> bool:
    """True when Google Drive is already mounted at the expected path."""
    return os.path.isdir(mount_point)


def mount_drive(mount_point: str = "/content/drive") -> bool:
    """Mount Google Drive in Colab. No-op outside Colab. Returns True on success."""
    if not in_colab():
        print("[starfelt] Not in Colab — skipping Drive mount.")
        return False
    try:
        from google.colab import drive  # type: ignore[import]
        drive.mount(mount_point)
        return True
    except Exception as e:
        print(f"[starfelt] Drive mount failed: {e}")
        return False


def resume_latest(
    model: Any,
    optimizer: Any,
    train_loader: Any,
    *,
    run_id: str | None = None,
    checkpoint_dir: str | Path | None = None,
    **trainer_kwargs: Any,
):
    """Build a Trainer from the newest valid Drive checkpoint.

    Run this after reconnecting a Colab runtime. The returned Trainer resumes
    from ``latest.pt`` and can continue from the middle of an epoch when step
    checkpoints are enabled.
    """
    if in_colab() and not drive_mounted(
        os.environ.get("STARFELT_DRIVE_ROOT", "/content/drive/MyDrive")
    ):
        mount_drive()

    if checkpoint_dir is None:
        drive_root = Path(
            os.environ.get("STARFELT_DRIVE_ROOT", "/content/drive/MyDrive")
        )
        checkpoints_root = drive_root / ".starfelt" / "checkpoints"
        if run_id:
            checkpoint_dir = checkpoints_root / run_id
        else:
            candidates = [
                path
                for path in checkpoints_root.glob("*/latest.pt")
                if path.is_file()
            ]
            if not candidates:
                raise FileNotFoundError(
                    f"No Starfelt checkpoints found under {checkpoints_root}"
                )
            checkpoint_dir = max(candidates, key=lambda path: path.stat().st_mtime).parent

    checkpoint_dir = Path(checkpoint_dir)
    latest = checkpoint_dir / "latest.pt"
    if not latest.exists():
        raise FileNotFoundError(f"No latest.pt found under {checkpoint_dir}")

    resolved_run_id = run_id or checkpoint_dir.name
    from starfelt.trainer import Trainer

    trainer_kwargs.setdefault("checkpoint_dir", checkpoint_dir)
    trainer_kwargs.setdefault("run_id", resolved_run_id)
    previous_resume = os.environ.get("STARFELT_RESUME_FROM")
    previous_run_id = os.environ.get("STARFELT_RUN_ID")
    os.environ["STARFELT_RESUME_FROM"] = str(latest)
    os.environ["STARFELT_RUN_ID"] = resolved_run_id
    try:
        return Trainer(model, optimizer, train_loader, **trainer_kwargs)
    finally:
        if previous_resume is None:
            os.environ.pop("STARFELT_RESUME_FROM", None)
        else:
            os.environ["STARFELT_RESUME_FROM"] = previous_resume
        if previous_run_id is None:
            os.environ.pop("STARFELT_RUN_ID", None)
        else:
            os.environ["STARFELT_RUN_ID"] = previous_run_id


# ---------------------------------------------------------------------------
# Live display
# ---------------------------------------------------------------------------

class StarfeltDisplay:
    """Live epoch-by-epoch display for Jupyter / Colab output cells.

    Pass an instance to ``Trainer(notebook_display=display)`` and
    each epoch updates the same output cell in place.
    """

    def __init__(self, show_cost: bool = True, show_gpu: bool = True) -> None:
        self.show_cost = show_cost
        self.show_gpu = show_gpu
        self._handle = None
        self._rows: list[dict[str, Any]] = []
        self._start = time.time()

        if not in_notebook():
            print("[starfelt] StarfeltDisplay: not in a notebook — falling back to print.")

    def update(
        self,
        epoch: int,
        loss: float,
        *,
        val_loss: float | None = None,
        lr: float | None = None,
        cost_usd: float = 0.0,
        gpu_util: float | None = None,
        samples_per_sec: float | None = None,
    ) -> None:
        """Call after each epoch. Updates the output cell in place."""
        self._rows.append({
            "epoch": epoch + 1,
            "loss": loss,
            "val_loss": val_loss,
            "lr": lr,
            "cost_usd": cost_usd,
            "gpu_util": gpu_util,
            "smp_s": samples_per_sec,
        })
        self._render()

    def _render(self) -> None:
        if not in_notebook():
            r = self._rows[-1]
            parts = [f"epoch {r['epoch']}  loss={r['loss']:.4f}"]
            if r["val_loss"] is not None:
                parts.append(f"val={r['val_loss']:.4f}")
            if r["lr"] is not None:
                parts.append(f"lr={r['lr']:.2e}")
            if self.show_cost:
                parts.append(f"${r['cost_usd']:.4f}")
            print("  ".join(parts), flush=True)
            return

        try:
            from IPython.display import HTML, clear_output, display  # type: ignore[import]
        except ImportError:
            return

        elapsed = time.time() - self._start
        h, rem = divmod(int(elapsed), 3600)
        m, s = divmod(rem, 60)
        elapsed_s = f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"

        rows_html = ""
        for r in self._rows:
            val_td = f"{r['val_loss']:.4f}" if r["val_loss"] is not None else "—"
            lr_td = f"{r['lr']:.2e}" if r["lr"] is not None else "—"
            smp_td = f"{r['smp_s']:.0f}" if r["smp_s"] is not None else "—"
            gpu_td = f"{r['gpu_util']:.0f}%" if r["gpu_util"] is not None else "—"
            cost_td = f"${r['cost_usd']:.4f}"
            rows_html += (
                f"<tr>"
                f"<td>{r['epoch']}</td>"
                f"<td>{r['loss']:.4f}</td>"
                f"<td>{val_td}</td>"
                f"<td>{lr_td}</td>"
                f"<td>{smp_td}</td>"
                f"<td>{gpu_td}</td>"
                f"<td>{cost_td}</td>"
                f"</tr>"
            )

        html = f"""
        <div style="font-family:monospace;font-size:13px">
          <b>⚡ Starfelt</b> &nbsp;·&nbsp; elapsed: {elapsed_s}
          <table border="1" cellpadding="4" cellspacing="0" style="margin-top:6px;border-collapse:collapse">
            <tr style="background:#f0f0f0">
              <th>Epoch</th><th>Loss</th><th>Val</th>
              <th>LR</th><th>smp/s</th><th>GPU</th><th>Cost</th>
            </tr>
            {rows_html}
          </table>
        </div>
        """
        clear_output(wait=True)
        display(HTML(html))

    def summary(self) -> None:
        """Print a final summary after training completes."""
        if not self._rows:
            return
        last = self._rows[-1]
        print(
            f"\n[starfelt] done — "
            f"epochs={last['epoch']}  "
            f"final_loss={last['loss']:.4f}"
            + (f"  val={last['val_loss']:.4f}" if last["val_loss"] is not None else "")
            + (f"  total_cost=${last['cost_usd']:.4f}" if self.show_cost else "")
        )
