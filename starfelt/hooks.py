"""Plugin hooks for advanced users and third-party extensions.

Register callables that Starfelt (and the Trainer SDK) will invoke at key
points in a run.

Example::

    from starfelt.hooks import on_epoch_end, on_checkpoint, on_budget_warning

    @on_epoch_end
    def my_logger(epoch: int, loss: float, cost_so_far: float) -> None:
        print(f"custom: epoch={epoch} loss={loss:.4f} cost=${cost_so_far:.3f}")

    @on_checkpoint
    def upload(path: str) -> None:
        ...  # e.g. push to object store

    @on_budget_warning
    def alert(pct_used: float) -> None:
        if pct_used > 0.9:
            send_slack(...)
"""

from __future__ import annotations

from typing import Callable

_EPOCH_END: list[Callable[[int, float, float], None]] = []
_CHECKPOINT: list[Callable[[str], None]] = []
_BUDGET_WARNING: list[Callable[[float], None]] = []


def on_epoch_end(fn: Callable[[int, float, float], None]) -> Callable[[int, float, float], None]:
    """Decorator / registrar: called after each epoch with (epoch, loss, cost_so_far)."""
    _EPOCH_END.append(fn)
    return fn


def on_checkpoint(fn: Callable[[str], None]) -> Callable[[str], None]:
    """Decorator / registrar: called with the checkpoint path after a save."""
    _CHECKPOINT.append(fn)
    return fn


def on_budget_warning(fn: Callable[[float], None]) -> Callable[[float], None]:
    """Decorator / registrar: called with fraction of budget used (0.0–1.0+)."""
    _BUDGET_WARNING.append(fn)
    return fn


def clear_hooks() -> None:
    """Remove all registered hooks (useful in tests)."""
    _EPOCH_END.clear()
    _CHECKPOINT.clear()
    _BUDGET_WARNING.clear()


def fire_epoch_end(epoch: int, loss: float, cost_so_far: float) -> None:
    for fn in list(_EPOCH_END):
        try:
            fn(epoch, loss, cost_so_far)
        except Exception:
            pass


def fire_checkpoint(path: str) -> None:
    for fn in list(_CHECKPOINT):
        try:
            fn(path)
        except Exception:
            pass


def fire_budget_warning(pct_used: float) -> None:
    for fn in list(_BUDGET_WARNING):
        try:
            fn(pct_used)
        except Exception:
            pass
