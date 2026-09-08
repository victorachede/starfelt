"""Starfelt — training efficiency layer."""

from __future__ import annotations

__version__ = "0.1.0"

# Public SDK surface
from starfelt.callbacks import StarfeltCallback
from starfelt.trainer import Trainer, TrainerResult

__all__ = [
    "__version__",
    "StarfeltCallback",
    "Trainer",
    "TrainerResult",
]
