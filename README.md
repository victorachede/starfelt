# Starfelt

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
starfelt cost --compare
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

Real signal from the first runs: same model, different flags → measurable $ difference. That comparison is the internal sales tool.

---

## Commands

| Command | What it does |
|---------|----------------|
| `starfelt init` | Write `starfelt.yaml` + `.starfelt/` |
| `starfelt analyze SCRIPT` | Pre-run AST checks (batch, LR, DataLoader, budget) |
| `starfelt run SCRIPT` | Preflight → stream logs → live cost → telemetry |
| `starfelt run SCRIPT --force` | Skip critical-issue prompt |
| `starfelt run SCRIPT --dry-run` | Analyze only |
| `starfelt status` / `--watch` | Active run + last 5 + total spend |
| `starfelt cost` / `--compare` | History + baseline savings |
| `starfelt inspect {run_id}` | Full telemetry, loss curve, recommendations |
| `starfelt compare {id1} {id2}` | Side-by-side cost / duration / efficiency |
| `starfelt resume {run_id}` | Reload last checkpoint and continue |
| `starfelt benchmark` | Mini job → effective throughput + suggested `$/hr` |
| `starfelt config doctor` | Validate `starfelt.yaml` vs real environment |
| `starfelt providers list` | Static GPU price catalog |
| `starfelt doctor` | Full environment diagnostics |
| `starfelt login` / `sync` | Optional Supabase history |

---

## Trainer SDK

```python
from starfelt import Trainer, StarfeltCallback

trainer = Trainer(
    model=model,
    optimizer=optimizer,
    train_loader=train_loader,
    loss_fn=criterion,          # optional if model returns scalar loss
    epochs=10,
    amp=True,                   # mixed precision when CUDA available
    checkpoint_every_epochs=1,
)
result = trainer.fit()

print(result.run_id, result.cost_usd, result.final_loss)
print(result.epoch_history)     # per-epoch loss / lr / gpu / cost
```

Under the hood it:

- Checkpoints to `.starfelt/checkpoints/{run_id}/`
- Runs `StarfeltCallback` early-stop (respects `STARFELT_EARLY_STOP`)
- Records per-epoch telemetry into the same run JSON the CLI uses
- Honors `STARFELT_RESUME_FROM` for seamless resume

Plugin hooks for advanced users:

```python
from starfelt.hooks import on_epoch_end, on_checkpoint, on_budget_warning

@on_epoch_end
def log(epoch, loss, cost_so_far):
    ...
```

---

## Examples

```
examples/
  train_toy.py           # zero-dep smoke test
  train_resnet.py        # tiny ResNet + Trainer SDK
  train_llm_finetune.py  # causal LM sketch + Trainer SDK
  train_jax.py           # JAX loop (framework detection)
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
  callbacks.py  # StarfeltCallback
  hooks.py      # plugin surface
  providers/    # static price catalog
examples/
tests/
```

---

## License

Proprietary · private repository
