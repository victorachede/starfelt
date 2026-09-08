"""Minimal ResNet-style image classifier using the Starfelt Trainer SDK.

No external dataset required — uses random tensors so it runs anywhere.
Copy-paste starting point for real PyTorch image training.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from starfelt import Trainer


class TinyResNet(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.block = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = x + self.block(x)
        return self.head(x)


def main() -> None:
    # Synthetic "dataset" — replace with real ImageFolder / CIFAR later
    n = 256
    images = torch.randn(n, 3, 32, 32)
    labels = torch.randint(0, 10, (n,))
    loader = DataLoader(TensorDataset(images, labels), batch_size=32, shuffle=True)

    model = TinyResNet()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    loss_fn = nn.CrossEntropyLoss()

    trainer = Trainer(
        model=model,
        optimizer=opt,
        train_loader=loader,
        loss_fn=loss_fn,
        epochs=3,
        checkpoint_every_epochs=1,
        amp=torch.cuda.is_available(),
    )
    result = trainer.fit()
    print(
        f"[resnet] done  epochs={result.epochs_completed}  "
        f"final_loss={result.final_loss:.4f}  cost=${result.cost_usd:.4f}  "
        f"run_id={result.run_id[:8]}"
    )


if __name__ == "__main__":
    main()
