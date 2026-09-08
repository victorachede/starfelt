"""Real-ish fine-tune example with the Starfelt Trainer SDK.

- Train + validation split
- Gradient accumulation
- Early-stop on val loss
- AMP when CUDA is available
- Rich per-epoch metrics (samples/sec, memory, val_loss)

No Hugging Face required — pure PyTorch so it runs anywhere.
Swap the model/data for your own (or load an HF model and pass it in).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

from starfelt import Trainer


class Classifier(nn.Module):
    """Small MLP stand-in for a fine-tune head / tiny model."""

    def __init__(self, dim: int = 128, n_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def main() -> None:
    torch.manual_seed(0)
    n, dim, n_classes = 800, 128, 10
    x = torch.randn(n, dim)
    y = torch.randint(0, n_classes, (n,))
    ds = TensorDataset(x, y)
    train_ds, val_ds = random_split(ds, [640, 160])

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64)

    model = Classifier(dim, n_classes)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss()

    trainer = Trainer(
        model=model,
        optimizer=opt,
        train_loader=train_loader,
        loss_fn=loss_fn,
        val_loader=val_loader,
        epochs=5,
        grad_accum_steps=2,          # effective batch 64
        amp=torch.cuda.is_available(),
        early_stop_patience=2,       # stop if val doesn't improve
        eval_every_epochs=1,
        checkpoint_every_epochs=1,
    )
    result = trainer.fit()

    print()
    print(
        f"[finetune] done\n"
        f"  epochs={result.epochs_completed}  stopped_early={result.stopped_early}\n"
        f"  final_loss={result.final_loss:.4f}  final_val={result.final_val_loss}\n"
        f"  best_val={result.best_val_loss}  cost=${result.cost_usd:.4f}\n"
        f"  run_id={result.run_id}\n"
        f"  ckpt={result.checkpoint_path}"
    )


if __name__ == "__main__":
    main()
