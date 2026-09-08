from pathlib import Path

from starfelt.core.analyze import analyze_script
from starfelt.core.config import StarfeltConfig


def test_analyze_toy_example():
    script = Path(__file__).resolve().parents[1] / "examples" / "train_toy.py"
    report = analyze_script(script, StarfeltConfig())
    names = {c.name for c in report.checks}
    assert "batch_size" in names
    assert "learning_rate" in names
    assert report.est_cost_usd > 0
