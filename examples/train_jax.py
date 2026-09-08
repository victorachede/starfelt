"""Minimal JAX training loop so Starfelt framework detection is exercised.

Starfelt's Trainer SDK is PyTorch-first today; this script is for
``starfelt analyze`` / ``starfelt run`` to prove JAX is detected correctly.
"""

from __future__ import annotations

import os
import time

# Keep imports at module level so AST / import analysis sees jax
try:
    import jax
    import jax.numpy as jnp
    from jax import grad, jit, random
except ImportError:
    print("[jax] JAX not installed — install with: pip install jax")
    raise SystemExit(0)


def loss_fn(params, x, y):
    pred = jnp.dot(x, params)
    return jnp.mean((pred - y) ** 2)


def main() -> None:
    run_id = os.environ.get("STARFELT_RUN_ID", "local")
    print(f"[jax] starfelt run_id={run_id}", flush=True)
    key = random.PRNGKey(0)
    params = random.normal(key, (8,))
    x = random.normal(key, (32, 8))
    y = random.normal(key, (32,))

    grad_fn = jit(grad(loss_fn))
    epochs = 5
    lr = 1e-2
    for ep in range(epochs):
        g = grad_fn(params, x, y)
        params = params - lr * g
        l = float(loss_fn(params, x, y))
        time.sleep(0.05)
        print(f"[jax] epoch {ep + 1}/{epochs} loss={l:.4f}", flush=True)
    print("[jax] done", flush=True)


if __name__ == "__main__":
    main()
