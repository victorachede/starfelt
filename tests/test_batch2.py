import warnings
from pathlib import Path

import pytest

from starfelt.callbacks import StarfeltCallback
from starfelt.core.config import load_config
from starfelt.core.cost import write_interrupted_marker
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


def test_callback_patience(monkeypatch):
    monkeypatch.setenv("STARFELT_EARLY_STOP", "0")
    cb = StarfeltCallback(patience_steps=2)
    assert cb.enabled is False
    assert cb.step(1.0) is False


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
