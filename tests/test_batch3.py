from pathlib import Path

from starfelt.core.analyze import analyze_script
from starfelt.core.config import StarfeltConfig
from starfelt.core.telemetry import TelemetryHints, build_run_telemetry, workload_id


def test_framework_pytorch(tmp_path: Path):
    script = tmp_path / "t.py"
    script.write_text(
        "import torch\n"
        "from torch.utils.data import DataLoader\n"
        "batch_size = 32\n"
        "lr = 1e-3\n"
        "epochs = 2\n"
        "dataset_size = 1000\n"
        "loader = DataLoader(range(10), batch_size=32, num_workers=2)\n"
        "opt = torch.optim.AdamW([], lr=lr)\n",
        encoding="utf-8",
    )
    report = analyze_script(script, StarfeltConfig())
    assert report.telemetry.framework == "pytorch"
    assert report.telemetry.batch_size == 32
    assert report.telemetry.dataset_size == 1000
    assert any(c.name == "framework" and c.level == "ok" for c in report.checks)


def test_workload_id_stable():
    a = workload_id(
        framework="pytorch",
        model_size_bucket="1m_100m",
        dataset_size_bucket="1k_100k",
        batch_size=32,
        epochs=10,
    )
    b = workload_id(
        framework="pytorch",
        model_size_bucket="1m_100m",
        dataset_size_bucket="1k_100k",
        batch_size=32,
        epochs=10,
    )
    assert a == b
    assert a.startswith("wl_")


def test_build_telemetry_schema_fields():
    row = build_run_telemetry(
        run_id="x",
        script="t.py",
        duration_s=1.0,
        cost_usd=0.1,
        exit_code=0,
        hints=TelemetryHints(framework="pytorch", batch_size=32, epochs=5),
        model_param_count=2_000_000,
    )
    assert row["schema_version"] == "1.2.0"
    assert row["workload_id"].startswith("wl_")
    assert row["model_size_bucket"] == "1m_100m"
    assert "framework" in row
