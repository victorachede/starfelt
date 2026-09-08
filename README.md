# Starfelt

**Training efficiency layer.** Point it at your training script — Starfelt analyzes, monitors, and tracks cost. No infra rewrite.

```bash
pip install -e ".[dev]"   # or: pip install starfelt (after PyPI)
starfelt init
starfelt doctor
starfelt run examples/train_toy.py
```

> Private · Stage 1 — pre-run analysis + live run monitoring  
> Goal: one user who says *“this saved me money.”*

---

## Quickstart (copy-paste)

```bash
git clone https://github.com/victorachede/starfelt.git
cd starfelt
pip install -e ".[dev]"

starfelt init
starfelt analyze examples/train_toy.py
starfelt run examples/train_toy.py
starfelt status
starfelt cost
```

Optional while a run is active in another terminal:

```bash
starfelt status --watch
```

---

## Commands

| Command | What it does |
|---------|----------------|
| `starfelt init` | Write `starfelt.yaml` + `.starfelt/`, validate Python/deps |
| `starfelt analyze SCRIPT` | Pre-run AST checks (batch, LR, DataLoader, scheduler, budget) |
| `starfelt run SCRIPT` | Preflight table → stream logs → live cost ticker → cost record |
| `starfelt run SCRIPT --force` | Skip “critical issue — run anyway?” prompt |
| `starfelt run SCRIPT --dry-run` | Analyze only |
| `starfelt status` | Active run + last 5 + total spend |
| `starfelt status --watch` | Live refresh every second |
| `starfelt cost` | Full local cost history |
| `starfelt cost --compare` | Actual vs baseline + savings |
| `starfelt providers list` | Static GPU price catalog |
| `starfelt doctor` | Full environment diagnostics |
| `starfelt login` / `sync` | Optional Supabase history |


---

## What Stage 1 is

- **PyTorch-first** analysis (AST, not string soup)
- **Live** `run`: streamed stdout, cost ticker, fail confirmation
- **Terminal** cost/status (no web dashboard yet)
- Local history under `.starfelt/`
- Optional GPU util via `nvidia-smi`
- `StarfeltCallback` for early-stop in PyTorch loops
- Interrupt markers on SIGINT/SIGTERM
- Run telemetry + `workload_id` fingerprints (see `docs/TELEMETRY.md`)

## What Stage 1 is not

- Multi-cloud provisioning  
- Enterprise billing  
- Web app  

---

## Layout

```
starfelt/
  cli/        # init | analyze | run | status | cost
  core/       # analysis, runner, cost, config
  providers/  # static price stubs only
examples/train_toy.py
```

---

## License

Proprietary · private repository
