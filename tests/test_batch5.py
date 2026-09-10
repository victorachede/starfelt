"""Batch 5 — Trainer SDK, inspect helpers, hooks."""

from __future__ import annotations

import json

import pytest

from starfelt.cli.main import _find_run
from starfelt.hooks import clear_hooks, fire_epoch_end, on_epoch_end


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
        model,
        opt,
        loader,
        loss_fn=nn.MSELoss(),
        epochs=2,
        checkpoint_every_epochs=1,
        grad_accum_steps=2,
    )
    assert trainer.callback is None
    result = trainer.fit()
    assert result.epochs_completed == 2
    assert len(result.epoch_history) == 2
    assert result.cost_usd >= 0
    assert "samples_per_sec" in result.epoch_history[0]
    run_file = tmp_path / ".starfelt" / "runs" / f"{result.run_id}.json"
    assert run_file.exists()
    data = json.loads(run_file.read_text())
    assert data.get("source") == "trainer_sdk"
    assert "epoch_history" in data
    assert data.get("grad_accum_steps") == 2


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("torch") is None,
    reason="torch not installed",
)
def test_trainer_flushes_partial_accumulation(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".starfelt").mkdir()

    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    from starfelt import Trainer

    class CountingSGD(torch.optim.SGD):
        def __init__(self, params):
            super().__init__(params, lr=0.05)
            self.step_calls = 0

        def step(self, closure=None):
            self.step_calls += 1
            return super().step(closure)

    x = torch.ones(3, 1)
    y = torch.zeros(3, 1)
    loader = DataLoader(TensorDataset(x, y), batch_size=1)
    model = nn.Linear(1, 1)
    optimizer = CountingSGD(model.parameters())

    Trainer(
        model,
        optimizer,
        loader,
        loss_fn=nn.MSELoss(),
        epochs=1,
        grad_accum_steps=2,
    ).fit()

    assert optimizer.step_calls == 2


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("torch") is None,
    reason="torch not installed",
)
def test_trainer_resume_continues_after_checkpointed_epoch(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".starfelt").mkdir()

    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    from starfelt import Trainer

    x = torch.randn(8, 2)
    y = torch.randn(8, 1)
    loader = DataLoader(TensorDataset(x, y), batch_size=4)
    checkpoint_dir = tmp_path / "checkpoints"

    first_model = nn.Linear(2, 1)
    first_optimizer = torch.optim.SGD(first_model.parameters(), lr=0.05)
    first = Trainer(
        first_model,
        first_optimizer,
        loader,
        loss_fn=nn.MSELoss(),
        epochs=2,
        checkpoint_dir=checkpoint_dir,
        run_id="resume-test",
    )
    first_result = first.fit()
    latest = checkpoint_dir / "latest.pt"
    assert first_result.epochs_completed == 2
    assert latest.exists()

    monkeypatch.setenv("STARFELT_RESUME_FROM", str(latest))
    resumed_model = nn.Linear(2, 1)
    resumed_optimizer = torch.optim.SGD(resumed_model.parameters(), lr=0.05)
    resumed = Trainer(
        resumed_model,
        resumed_optimizer,
        loader,
        loss_fn=nn.MSELoss(),
        epochs=4,
        checkpoint_dir=checkpoint_dir,
        run_id="resume-test",
    )
    result = resumed.fit()

    assert result.epochs_completed == 4
    assert [row["epoch"] for row in result.epoch_history] == [0, 1, 2, 3]


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("torch") is None,
    reason="torch not installed",
)
def test_trainer_quality_target_budget_and_colab_checkpoint(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".starfelt").mkdir()

    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    from starfelt import Trainer
    from starfelt.core.config import StarfeltConfig

    drive_mount = tmp_path / "MyDrive"
    drive_mount.mkdir()
    monkeypatch.setenv("STARFELT_DRIVE_ROOT", str(drive_mount))
    monkeypatch.setattr("starfelt.trainer.in_colab", lambda: True)
    monkeypatch.setattr(
        "starfelt.trainer.drive_mounted",
        lambda path: path == str(drive_mount),
    )
    drive_dir = drive_mount / ".starfelt" / "checkpoints" / "colab-test"
    x = torch.ones(4, 1)
    y = torch.zeros(4, 1)
    loader = DataLoader(TensorDataset(x, y), batch_size=2)
    model = nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    trainer = Trainer(
        model,
        optimizer,
        loader,
        loss_fn=nn.MSELoss(),
        val_loader=loader,
        epochs=5,
        checkpoint_every_epochs=1,
        project_config=StarfeltConfig(
            budget_usd_per_run=0,
            gpu_hour_usd=1.2,
        ),
        run_id="colab-test",
        target_val_loss=1000.0,
        stop_at_target=True,
    )

    assert trainer.checkpoint_dir == drive_dir
    result = trainer.fit()

    assert result.target_val_loss == 1000.0
    assert result.cost_to_target_usd is not None
    assert result.epochs_completed == 1
    assert (drive_dir / "latest.pt").exists()
    assert result.budget_exceeded is False


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("torch") is None,
    reason="torch not installed",
)
def test_trainer_stops_at_budget(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".starfelt").mkdir()

    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    from starfelt import Trainer
    from starfelt.core.config import StarfeltConfig

    x = torch.ones(8, 1)
    y = torch.zeros(8, 1)
    loader = DataLoader(TensorDataset(x, y), batch_size=2)
    model = nn.Linear(1, 1)
    trainer = Trainer(
        model,
        torch.optim.SGD(model.parameters(), lr=0.01),
        loader,
        loss_fn=nn.MSELoss(),
        epochs=10,
        project_config=StarfeltConfig(
            budget_usd_per_run=0.000000000001,
            gpu_hour_usd=1.2,
        ),
    )

    result = trainer.fit()

    assert result.budget_exceeded
    assert result.stopped_early
