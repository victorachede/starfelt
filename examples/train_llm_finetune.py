"""Tiny causal LM fine-tune sketch using the Starfelt Trainer SDK.

Uses a toy embedding + transformer block so it runs on CPU in seconds.
Replace the model / data with your real HF model + dataset.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from starfelt import Trainer


class TinyCausalLM(nn.Module):
    def __init__(self, vocab: int = 128, dim: int = 64, n_layers: int = 2):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=4, dim_feedforward=128, batch_first=True, dropout=0.0
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.lm_head = nn.Linear(dim, vocab, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T)
        h = self.embed(x)
        h = self.blocks(h)
        return self.lm_head(h)


def main() -> None:
    vocab, seq_len, n = 128, 32, 128
    tokens = torch.randint(0, vocab, (n, seq_len + 1))
    # next-token prediction: input = tokens[:, :-1], target = tokens[:, 1:]
    x = tokens[:, :-1]
    y = tokens[:, 1:]
    loader = DataLoader(TensorDataset(x, y), batch_size=16, shuffle=True)

    model = TinyCausalLM(vocab=vocab)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    def wrapped_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits (B, T, V) → (B*T, V)
        return loss_fn(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

    trainer = Trainer(
        model=model,
        optimizer=opt,
        train_loader=loader,
        loss_fn=wrapped_loss,
        epochs=2,
        checkpoint_every_epochs=1,
    )
    result = trainer.fit()
    print(
        f"[llm] done  epochs={result.epochs_completed}  "
        f"final_loss={result.final_loss:.4f}  cost=${result.cost_usd:.4f}  "
        f"run_id={result.run_id[:8]}"
    )


if __name__ == "__main__":
    main()
