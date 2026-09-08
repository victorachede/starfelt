"""Batch 5 — Trainer SDK, inspect helpers, hooks."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from starfelt.hooks import clear_hooks, fire_epoch_end, on_epoch_end
from starfelt.cli.main import _find_run


def test_hooks_fire():
    clear_hooks()
    seen = []

    @on_epoch_end
    def capture(epoch, loss, cost):
        seen.append((epoch, loss, cost))

    fire_epoch_end(0, 1.23, 0.01)
    assert seen == [(0, 1.23, 0.01)]
    clear_hooks()


def test_find_run_empty(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".starfelt" / "runs").mkdir(parents=True)
    assert _find_run("nonexistent") is None


def test_find_run_from_history(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    d = tmp_path / ".starfelt"
    d.mkdir()
    row = {"run_id": "abc123def", "cost_usd": 1.5, "script": "x.py"}
    (d / "history.json").write_text(json.dumps([row]), encoding="utf-8")
    found = _find_run("abc123")
    assert found is not None
    assert found["run_id"] == "abc123def"


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("torch") is None,
    reason="torch not installed",
)
def test_trainer_smoke(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".starfelt").mkdir()

    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    from starfelt import Trainer

    x = torch.randn(32, 4)
    y = torch.randn(32, 1)
    loader = DataLoader(TensorDataset(x, y), batch_size=8)
    model = nn.Linear(4, 1)
    opt = torch.optim.SGD(model.parameters(), lr=0.05)
    trainer = Trainer(
        model, opt, loader, loss_fn=nn.MSELoss(), epochs=2, checkpoint_every_epochs=1
    )
    result = trainer.fit()
    assert result.epochs_completed == 2
    assert len(result.epoch_history) == 2
    assert result.cost_usd >= 0
    run_file = tmp_path / ".starfelt" / "runs" / f"{result.run_id}.json"
    assert run_file.exists()
    data = json.loads(run_file.read_text())
    assert data.get("source") == "trainer_sdk"
    assert "epoch_history" in data
