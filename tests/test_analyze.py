from pathlib import Path

from starfelt.core.analyze import analyze_script
from starfelt.core.config import StarfeltConfig


def test_analyze_toy_example():
    script = Path(__file__).resolve().parents[1] / "examples" / "train_toy.py"
    report = analyze_script(script, StarfeltConfig())
    names = {c.name for c in report.checks}
    assert "batch_size" in names
    assert "learning_rate" in names
    assert "scheduler" in names
    assert report.est_cost_usd > 0


def test_analyze_torch_style_ast(tmp_path: Path):
    script = tmp_path / "train.py"
    script.write_text(
        """
import torch
from torch.utils.data import DataLoader

batch_size = 64
lr = 1e-3
epochs = 5

loader = DataLoader(range(10), batch_size=batch_size, num_workers=4)
opt = torch.optim.AdamW(params=[], lr=lr)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=5)
""",
        encoding="utf-8",
    )
    report = analyze_script(script, StarfeltConfig())
    by = {c.name: c for c in report.checks}
    assert by["batch_size"].level == "ok"
    assert by["learning_rate"].level == "ok"
    assert by["data_pipeline"].level == "ok"
    assert by["scheduler"].level == "ok"
