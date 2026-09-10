from pathlib import Path

from starfelt.core.analyze import analyze_script
from starfelt.core.config import StarfeltConfig, validate_environment
from starfelt.core.runner import run_wrapped


def test_validate_environment_has_python():
    rows = validate_environment()
    names = {r[0] for r in rows}
    assert "python" in names


def test_fail_level_high_lr(tmp_path: Path):
    script = tmp_path / "bad_lr.py"
    script.write_text("lr = 0.5\nbatch_size = 32\nprint('hi')\n", encoding="utf-8")
    report = analyze_script(script, StarfeltConfig())
    assert any(c.level == "fail" and c.name == "learning_rate" for c in report.checks)


def test_run_force_skips_prompt(tmp_path: Path):
    script = tmp_path / "bad_lr.py"
    script.write_text(
        "lr = 0.5\nbatch_size = 32\nprint('ok')\n",
        encoding="utf-8",
    )
    result = run_wrapped(
        script,
        [],
        StarfeltConfig(),
        force=True,
        confirm_fails=False,
    )
    assert result.exit_code == 0
    assert not result.aborted


def test_run_stops_when_budget_is_reached(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    script = tmp_path / "slow.py"
    script.write_text("import time\ntime.sleep(2)\n", encoding="utf-8")

    result = run_wrapped(
        script,
        [],
        StarfeltConfig(budget_usd_per_run=0.000001),
        force=True,
        confirm_fails=False,
    )

    assert result.budget_exceeded
    assert result.exit_code != 0
    assert (tmp_path / ".starfelt" / "history.json").exists()
