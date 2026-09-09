from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path

from starfelt.core.config import StarfeltConfig
from starfelt.core.telemetry import TelemetryHints


@dataclass
class Check:
    name: str
    level: str  # ok | warn | fail
    detail: str


@dataclass
class AnalysisReport:
    checks: list[Check]
    est_hours: float
    est_cost_usd: float
    est_cost_optimized_usd: float
    telemetry: TelemetryHints = field(default_factory=TelemetryHints)


class _TrainVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.assigns: dict[str, float | int | str] = {}
        self.batch_size: int | None = None
        self.lr: float | None = None
        self.epochs: int | None = None
        self.has_dataloader = False
        self.dataloader_num_workers: int | None = None
        self.has_scheduler = False
        self.has_torch_optim = False
        self.argparse_defaults: dict[str, float | int] = {}
        self.imports: set[str] = set()
        self.dataset_size: int | None = None
        self.optimization_flags: list[str] = []

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.imports.add(alias.name.split(".")[0])
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module:
            root = node.module.split(".")[0]
            self.imports.add(root)
            if node.module.startswith("torch.cuda.amp") or "amp" in (node.module or ""):
                self.optimization_flags.append("amp")
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            val = self._literal(node.value)
            if val is not None:
                self.assigns[name] = val
                self._bind_known(name, val)
            # len(...) assignment
            size = self._len_call(node.value)
            if size is not None and name.lower() in {
                "dataset_size",
                "n_samples",
                "num_samples",
                "data_len",
            }:
                self.dataset_size = size
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if isinstance(node.target, ast.Name) and node.value is not None:
            val = self._literal(node.value)
            if val is not None:
                self.assigns[node.target.id] = val
                self._bind_known(node.target.id, val)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        fname = self._call_name(node)
        if fname:
            low = fname.lower()
            if low.endswith("dataloader") or low == "dataloader":
                self.has_dataloader = True
                for kw in node.keywords:
                    if kw.arg == "num_workers":
                        v = self._literal(kw.value)
                        if isinstance(v, int):
                            self.dataloader_num_workers = v
                        elif isinstance(kw.value, ast.Name) and kw.value.id in self.assigns:
                            av = self.assigns[kw.value.id]
                            if isinstance(av, int):
                                self.dataloader_num_workers = av
            if "scheduler" in low:
                self.has_scheduler = True
            if low.startswith("torch.optim.") or low in {
                "adam",
                "adamw",
                "sgd",
                "rmsprop",
            }:
                self.has_torch_optim = True
                for kw in node.keywords:
                    if kw.arg in {"lr", "learning_rate"}:
                        v = self._resolve_value(kw.value)
                        if isinstance(v, (int, float)):
                            self.lr = float(v)
                if self.lr is None and len(node.args) >= 2:
                    v = self._resolve_value(node.args[1])
                    if isinstance(v, (int, float)):
                        self.lr = float(v)

            if low in {"add_argument"} or fname.endswith("add_argument"):
                self._parse_add_argument(node)

            # torch.compile(...)
            if low.endswith("compile") and "torch" in low:
                self.optimization_flags.append("torch.compile")
            if "autocast" in low:
                self.optimization_flags.append("amp")
            if "gradscaler" in low or low.endswith("gradscaler"):
                self.optimization_flags.append("grad_scaler")

            # len(dataset) or len(dataloader.dataset) used as call in expression
            if low == "len" and node.args:
                size = self._len_from_arg(node.args[0])
                if size is not None:
                    self.dataset_size = size

        self.generic_visit(node)

    def _len_call(self, node: ast.AST) -> int | None:
        if isinstance(node, ast.Call) and self._call_name(node) == "len" and node.args:
            return self._len_from_arg(node.args[0])
        return None

    def _len_from_arg(self, arg: ast.AST) -> int | None:
        # len(something) where something is a list/tuple literal
        if isinstance(arg, (ast.List, ast.Tuple)):
            return len(arg.elts)
        # len(range(N))
        if isinstance(arg, ast.Call) and self._call_name(arg) == "range":
            if arg.args and isinstance(arg.args[0], ast.Constant) and isinstance(
                arg.args[0].value, int
            ):
                return int(arg.args[0].value)
        return None

    def _parse_add_argument(self, node: ast.Call) -> None:
        flag = None
        if node.args and isinstance(node.args[0], ast.Constant):
            flag = str(node.args[0].value)
        default = None
        for kw in node.keywords:
            if kw.arg == "default":
                default = self._literal(kw.value)
        if flag and default is not None and isinstance(default, (int, float)):
            key = flag.lstrip("-").replace("-", "_")
            self.argparse_defaults[key] = default
            self._bind_known(key, default)

    def _bind_known(self, name: str, val: float | int | str) -> None:
        if not isinstance(val, (int, float)):
            return
        n = name.lower()
        if n in {"batch_size", "batch", "bs", "train_batch_size"}:
            self.batch_size = int(val)
        if n in {"lr", "learning_rate", "init_lr"}:
            self.lr = float(val)
        if n in {"epochs", "num_epochs", "n_epochs", "max_epochs"}:
            self.epochs = int(val)
        if n == "num_workers":
            self.dataloader_num_workers = int(val)
        if n in {"dataset_size", "n_samples", "num_samples"}:
            self.dataset_size = int(val)

    def _resolve_value(self, node: ast.AST) -> float | int | str | None:
        lit = self._literal(node)
        if lit is not None:
            return lit
        if isinstance(node, ast.Name) and node.id in self.assigns:
            return self.assigns[node.id]
        return None

    @staticmethod
    def _literal(node: ast.AST) -> float | int | str | None:
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float, str)):
                return node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            v = _TrainVisitor._literal(node.operand)
            if isinstance(v, (int, float)):
                return -v
        return None

    @staticmethod
    def _call_name(node: ast.Call) -> str | None:
        f = node.func
        parts: list[str] = []
        while isinstance(f, ast.Attribute):
            parts.append(f.attr)
            f = f.value
        if isinstance(f, ast.Name):
            parts.append(f.id)
            return ".".join(reversed(parts))
        return None


def detect_frameworks(imports: set[str]) -> tuple[str, list[str]]:
    found: list[str] = []
    if "torch" in imports:
        found.append("pytorch")
    if "jax" in imports or "flax" in imports:
        found.append("jax")
    if "tensorflow" in imports or "tf" in imports:
        found.append("tensorflow")
    if "keras" in imports:
        found.append("keras")
    primary = found[0] if found else "unknown"
    return primary, found


def analyze_script(script: Path, cfg: StarfeltConfig) -> AnalysisReport:
    text = script.read_text(encoding="utf-8", errors="replace")
    checks: list[Check] = []

    try:
        tree = ast.parse(text)
        checks.append(Check("syntax", "ok", "Python parses cleanly"))
    except SyntaxError as e:
        checks.append(Check("syntax", "fail", f"SyntaxError: {e.msg}"))
        return AnalysisReport(
            checks=checks,
            est_hours=0.1,
            est_cost_usd=0.1 * cfg.gpu_hour_usd,
            est_cost_optimized_usd=0.1 * cfg.gpu_hour_usd * 0.74,
            telemetry=TelemetryHints(),
        )

    visitor = _TrainVisitor()
    visitor.visit(tree)

    primary, frameworks = detect_frameworks(visitor.imports)
    if primary == "unknown":
        checks.append(
            Check("framework", "warn", "No torch/jax/tensorflow/keras import detected")
        )
    else:
        checks.append(
            Check(
                "framework",
                "ok",
                f"{primary}"
                + (f" (+{', '.join(frameworks[1:])})" if len(frameworks) > 1 else ""),
            )
        )

    batch = visitor.batch_size
    lr = visitor.lr
    epochs = visitor.epochs

    if batch is None:
        checks.append(Check("batch_size", "warn", "No batch_size assignment found in AST"))
    elif batch < 8:
        checks.append(
            Check(
                "batch_size",
                "warn",
                f"batch_size={batch} may under-utilize GPU; consider grad accumulation",
            )
        )
    elif batch > 512:
        checks.append(
            Check(
                "batch_size",
                "warn",
                f"batch_size={batch} is large — OOM / noisy steps risk",
            )
        )
    else:
        checks.append(Check("batch_size", "ok", f"batch_size={batch}"))

    if lr is None:
        checks.append(
            Check(
                "learning_rate",
                "warn",
                "No lr found (assignments, torch.optim kwargs, or argparse defaults)",
            )
        )
    elif lr > 0.1:
        checks.append(
            Check("learning_rate", "fail", f"lr={lr} is very high — likely wasted runs")
        )
    elif lr < 1e-6:
        checks.append(
            Check(
                "learning_rate",
                "warn",
                f"lr={lr} is extremely small — slow convergence risk",
            )
        )
    else:
        checks.append(Check("learning_rate", "ok", f"lr={lr}"))

    if visitor.has_dataloader:
        if visitor.dataloader_num_workers == 0:
            checks.append(
                Check(
                    "data_pipeline",
                    "warn",
                    "DataLoader num_workers=0 — GPU may idle; raise workers + prefetch",
                )
            )
        else:
            detail = "DataLoader found"
            if visitor.dataloader_num_workers is not None:
                detail += f" (num_workers={visitor.dataloader_num_workers})"
            checks.append(Check("data_pipeline", "ok", detail))
    else:
        checks.append(Check("data_pipeline", "warn", "No DataLoader() call found in AST"))

    if visitor.has_scheduler:
        checks.append(Check("scheduler", "ok", "LR scheduler usage detected"))
    else:
        checks.append(
            Check(
                "scheduler",
                "warn",
                "No LR scheduler detected — fixed LR often wastes compute late in training",
            )
        )

    if visitor.has_torch_optim:
        checks.append(Check("optimizer", "ok", "torch.optim-style optimizer detected"))
    elif primary == "pytorch":
        checks.append(
            Check("optimizer", "warn", "PyTorch import but no torch.optim call found")
        )
    else:
        checks.append(
            Check(
                "optimizer",
                "warn",
                "No torch.optim optimizer call found (ok if custom / non-PyTorch)",
            )
        )

    if visitor.dataset_size is not None:
        checks.append(
            Check("dataset_size", "ok", f"estimated dataset_size={visitor.dataset_size}")
        )
    else:
        checks.append(
            Check(
                "dataset_size",
                "warn",
                "Could not infer dataset size (len(...) / dataset_size assign)",
            )
        )

    if visitor.optimization_flags:
        checks.append(
            Check(
                "optimizations",
                "ok",
                "flags: " + ", ".join(sorted(set(visitor.optimization_flags))),
            )
        )

    ep = epochs or 10
    est_hours = max(0.1, ep * 0.15 * (1.0 if (batch or 32) >= 16 else 1.4))
    est_cost = est_hours * cfg.gpu_hour_usd
    # Heuristic sketch only — real savings come from starfelt compare run_a run_b
    est_opt = est_cost * 0.74

    checks.append(
        Check(
            "budget",
            "ok" if est_cost <= cfg.budget_usd_per_run else "warn",
            f"est ${est_cost:.2f} vs budget ${cfg.budget_usd_per_run:.2f}",
        )
    )

    telem = TelemetryHints(
        framework=primary,
        frameworks_detected=frameworks,
        batch_size=batch,
        epochs=epochs,
        learning_rate=lr,
        dataset_size=visitor.dataset_size,
        optimization_flags=sorted(set(visitor.optimization_flags)),
    )

    return AnalysisReport(
        checks=checks,
        est_hours=est_hours,
        est_cost_usd=est_cost,
        est_cost_optimized_usd=est_opt,
        telemetry=telem,
    )
