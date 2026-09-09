"""PyTorch-friendly early-stop helper.

Usage (optional one-liner in your loop)::

    from starfelt.callbacks import StarfeltCallback
    cb = StarfeltCallback()
    ...
    cb.step(loss)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class StarfeltCallback:
    """Monitor loss plateau and request a graceful stop when patience is exceeded.

    Respects ``STARFELT_EARLY_STOP`` (default on if unset or \"1\").
    Patience from ``STARFELT_PATIENCE_STEPS`` or constructor.
    """

    patience_steps: int | None = None
    min_delta: float = 1e-4
    _best: float | None = field(default=None, init=False, repr=False)
    _stale: int = field(default=0, init=False, repr=False)
    _steps: int = field(default=0, init=False, repr=False)
    stopped: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        if self.patience_steps is None:
            raw = os.environ.get("STARFELT_PATIENCE_STEPS") or os.environ.get(
                "STARFELT_CHECKPOINT_EVERY"
            )
            try:
                self.patience_steps = int(raw) if raw else 500
            except ValueError:
                self.patience_steps = 500

    @property
    def enabled(self) -> bool:
        flag = os.environ.get("STARFELT_EARLY_STOP", "1")
        return flag not in {"0", "false", "False", "no"}

    def step(self, loss: float) -> bool:
        """Record loss. Returns True if training should stop."""
        self._steps += 1
        if not self.enabled:
            return False
        if self._best is None or loss < self._best - self.min_delta:
            self._best = float(loss)
            self._stale = 0
            return False
        self._stale += 1
        patience = self.patience_steps or 500
        if self._stale >= patience:
            self.stopped = True
            return True
        return False
