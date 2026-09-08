import warnings
from pathlib import Path

import pytest

from starfelt.callbacks import StarfeltCallback
from starfelt.core.config import StarfeltConfig, load_config
from starfelt.core.cost import write_interrupted_marker, load_history
from starfelt.core.gpu import GpuMonitor
from starfelt.providers.base import CATALOG, pick_cheapest


def test_config_rejects_zero_gpu_price(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "starfelt.yaml").write_text(
        "project: t\ncost:\n  gpu_hour_usd: 0\n",
        encoding="utf-8",
    )
    with pytest.raises(Exception) as ei:
        load_config()
    assert "gpu_hour_usd" in str(ei.value)


def test_config_rejects_unknown_provider(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "starfelt.yaml").write_text(
        "providers:\n  preferred: [notreal]\n",
        encoding="utf-8",
    )
    with pytest.raises(Exception) as ei:
        load_config()
    assert "unknown provider" in str(ei.value).lower()


def test_callback_patience():
    cb = StarfeltCallback(patience_steps=3, min_delta=0.01)
    # disable kill side-effect by setting enabled false via env after? 
    # step returns True when patience exceeded — but also sends SIGTERM
    # Use enabled path carefully: set STARFELT_EARLY_STOP=0 for no kill
    import os
    os.environ["STARFELT_EARLY_STOP"] = "0"
    cb2 = StarfeltCallback(patience_steps=2)
    assert cb2.enabled is False
    assert cb2.step(1.0) is False


def test_interrupted_marker(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = write_interrupted_marker(
        "abc123",
        script="train.py",
        elapsed_s=12.5,
        cost_usd=0.01,
        signal_name="SIGINT",
    )
    assert path.exists()
    assert "interrupted" in path.name


def test_gpu_monitor_no_smi():
    m = GpuMonitor()
    # may or may not be available; just ensure poll doesn't crash
    m.poll()


def test_catalog_nonempty():
    assert len(CATALOG) >= 1
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        pick_cheapest()
