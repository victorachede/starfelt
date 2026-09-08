"""Toy training script for Starfelt demos — no GPU required."""

import os
import time

batch_size = 32
lr = 3e-4
epochs = 3
num_workers = 2


def main() -> None:
    run_id = os.environ.get("STARFELT_RUN_ID", "local")
    print(f"[toy] starfelt run_id={run_id}")
    print(f"[toy] batch_size={batch_size} lr={lr} epochs={epochs}")
    for ep in range(epochs):
        time.sleep(0.2)
        print(f"[toy] epoch {ep + 1}/{epochs} loss={1.0 / (ep + 1):.4f}")
    print("[toy] done")


if __name__ == "__main__":
    main()
