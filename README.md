# Starfelt

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/victorachede/starfelt/blob/main/examples/colab_quickstart.ipynb)

**Stop burning GPU hours you don’t need.**

Starfelt is a training efficiency layer. Point it at any script — or drop in the one-line Trainer SDK — and it analyzes, monitors, checkpoints, tracks cost, and surfaces the data that actually cuts waste.

```bash
pip install -e ".[dev]"          # or: pip install starfelt
starfelt init
starfelt run examples/train_toy.py
```

> Private · Batch 5 — Trainer SDK + per-epoch telemetry + inspect / compare / resume  
> Goal: one researcher who says *“this saved me money.”*

---

## 30-second quickstart

```bash
git clone https://github.com/victorachede/starfelt.git
cd starfelt
pip install -e ".[dev]"

starfelt init
starfelt analyze examples/train_toy.py
starfelt run examples/train_toy.py
starfelt status
starfelt compare <run_a> <run_b>
```

**Or go nuclear with the SDK (zero CLI required):**

```python
from starfelt import Trainer

trainer = Trainer(model, optimizer, train_loader, loss_fn=criterion, epochs=10)
result = trainer.fit()
# → auto checkpoints, early-stop, per-epoch cost + loss curve, run JSON
```

---

## Why Starfelt

| Without Starfelt | With Starfelt |
|------------------|---------------|
| Guessing GPU-hour cost after the fact | Live cost ticker + history |
| Manual early-stop hacks | Built-in `StarfeltCallback` + Trainer |
| “Did run A or B waste more?” | `starfelt compare runA runB` |
| Lost progress on interrupt | Resumable checkpoints + `starfelt resume` |
| Static $/hr guesses | `starfelt benchmark` calibrates to *your* box |

The pre-run cost sketch is intentionally approximate. Real signal comes from
comparing the same workload with different flags after both runs finish.

---

## Commands

| Command | What it does |
|---------|----------------|
| `starfelt init` | Write `starfelt.yaml` + `.starfelt/` |
| `starfelt analyze SCRIPT` | Pre-run AST checks (batch, LR, DataLoader, budget) |
| `starfelt run SCRIPT` | Preflight → stream logs → live cost → telemetry |
| `starfelt run SCRIPT --force` | Skip critical-issue prompt |
| `starfelt run SCRIPT --no-budget` | Intentionally disable the configured runtime budget stop |
| `starfelt run SCRIPT --dry-run` | Analyze only |
| `starfelt status` / `--watch` | Active run + last 5 + total spend |
| `starfelt cost` | Local estimated-cost history |
| `starfelt inspect {run_id}` | Full telemetry, loss curve, recommendations |
| `starfelt compare {id1} {id2}` | Side-by-side cost / duration / efficiency |
| `starfelt resume {run_id}` | Reload last checkpoint and continue |
| `starfelt benchmark` | Repeatable mini job → throughput, energy, and estimated cost |
| `starfelt config doctor` | Validate `starfelt.yaml` vs real environment |
| `starfelt providers list` | Static GPU price catalog |
| `starfelt doctor` | Full environment diagnostics |
| `starfelt login` / `sync` | Optional Supabase history |

---

## Trainer SDK

```python
from starfelt import Trainer

trainer = Trainer(
    model=model,
    optimizer=optimizer,
    train_loader=train_loader,
    loss_fn=criterion,
    val_loader=val_loader,       # enables val metrics + early-stop
    epochs=10,
    grad_accum_steps=4,          # effective larger batch
    amp=True,                    # mixed precision on CUDA
    early_stop_patience=3,       # stop when val stalls
    data_parallel=True,          # multi-GPU when available
)
result = trainer.fit()

print(result.final_loss, result.best_val_loss, result.cost_usd)
print(result.epoch_history)  # loss, val_loss, lr, smp/s, gpu, mem, cost
```

Under the hood:

- Gradient accumulation + AMP + optional DataParallel
- Eval loop and opt-in early-stop on validation loss
- Checkpoints to `.starfelt/checkpoints/{run_id}/` (incl. `best.pt`)
- Per-epoch telemetry: loss, val_loss, samples/sec, GPU util, peak memory, cost
- Resume via `STARFELT_RESUME_FROM` (continues after the checkpointed epoch)
- Runtime budget stop with a 10-second graceful-shutdown window
- Optional `target_val_loss` telemetry for cost-to-quality comparisons
- Plugin hooks (`on_epoch_end`, `on_checkpoint`, `on_budget_warning`)

---

## Examples

```
examples/
  train_toy.py              # zero-dep smoke test
  train_resnet.py           # tiny ResNet + Trainer SDK
  train_llm_finetune.py     # causal LM sketch + Trainer SDK
  finetune_classifier.py    # train+val, grad accum, early-stop on val
  train_jax.py              # JAX loop (framework detection)
```

```bash
starfelt run examples/train_resnet.py
starfelt inspect <run_id>
```

---

## Telemetry (per-epoch in Batch 5)

Every finished run writes `.starfelt/runs/{run_id}.json` and appends to `history.json`.

New in Batch 5:

- `epoch_history[]` — loss, lr, duration, cost_so_far, gpu_util per epoch
- `source: "trainer_sdk"` when using the SDK
- Same `workload_id` fingerprint for later chip-design aggregation

See `docs/TELEMETRY.md`.

---

## Layout

```
starfelt/
  cli/          # init | analyze | run | status | cost | inspect | compare | …
  core/         # analysis, runner, cost, config, telemetry, gpu
  trainer.py    # Trainer SDK
  callbacks.py   # StarfeltCallback
  hooks.py      # plugin surface
  providers/    # static price catalog
examples/
tests/
```

---

## License

Proprietary · private repository
