"""Optional model parameter counting (Batch 3).

Runs a short probe in a subprocess when the training script exposes a
``build_model()`` or ``model = ...`` that can be imported safely.

Stage 1 approach: look for a companion convention —
  STARFELT_MODEL_PARAMS env set by user, or
  script defines ``starfelt_model_param_count() -> int``.

We also attempt a safe AST-free import of the script as a module only when
``STARFELT_PROBE_MODEL=1``.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path


def estimate_model_params(script: Path) -> int | None:
    """Return parameter count if discoverable without running full training."""
    env_val = os.environ.get("STARFELT_MODEL_PARAMS")
    if env_val:
        try:
            return int(env_val)
        except ValueError:
            pass

    if os.environ.get("STARFELT_PROBE_MODEL") != "1":
        return None

    # Optional probe — disabled by default (training scripts often have side effects)
    try:
        spec = importlib.util.spec_from_file_location("_starfelt_probe", script)
        if spec is None or spec.loader is None:
            return None
        mod = importlib.util.module_from_spec(spec)
        # Prevent accidental __main__ execution patterns somewhat
        sys.modules["_starfelt_probe"] = mod
        spec.loader.exec_module(mod)
        if hasattr(mod, "starfelt_model_param_count"):
            return int(mod.starfelt_model_param_count())
        model = getattr(mod, "model", None)
        if model is not None and hasattr(model, "parameters"):
            return int(sum(p.numel() for p in model.parameters()))
    except Exception:  # noqa: BLE001
        return None
    return None
