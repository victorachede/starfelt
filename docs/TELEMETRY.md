# Run telemetry schema (v1.2.0)

Every finished run writes a JSON object under `.starfelt/runs/{run_id}.json` and appends to `.starfelt/history.json`.

## Fields

| Field | Meaning |
|-------|---------|
| `schema_version` | Telemetry schema version |
| `run_id` | Unique run id |
| `script` | Path to training script |
| `duration_s` / `cost_usd` | Duration and estimated cost (real savings via `starfelt compare`) |
| `exit_code` | Process exit |
| `framework` | Primary: pytorch / jax / tensorflow / keras / unknown |
| `frameworks_detected` | All detected |
| `batch_size` / `epochs` / `learning_rate` | From AST |
| `dataset_size` | Inferred when possible |
| `model_param_count` | If `STARFELT_MODEL_PARAMS` or probe |
| `model_size_bucket` / `dataset_size_bucket` | Coarse buckets for grouping |
| `gpu_name` / `gpu_util_avg` / `gpu_samples` | From nvidia-smi when present |
| `gpu_power_avg_w` / `gpu_energy_j` | Average GPU power and sampled energy when nvidia-smi exposes power draw |
| `optimization_flags` | e.g. amp, torch.compile |
| `workload_id` | Fingerprint: framework + buckets + batch + epochs |
| `interrupted` | Signal name if SIGINT/SIGTERM |
| `budget_exceeded` / `budget_usd` | Whether the configured budget stopped the run and the limit used |
| `target_val_loss` / `target_reached` | Optional validation-loss target and whether it was reached |
| `cost_to_target_usd` | Estimated cost at the first epoch that reached the target |

## Model params

```bash
STARFELT_MODEL_PARAMS=12345678 starfelt run train.py
# or opt-in probe (scripts must be import-safe):
STARFELT_PROBE_MODEL=1 starfelt run train.py
```

## Why

Workload fingerprints are the passive dataset for later chip decisions — what people actually train, not synthetic benches alone.

## Batch 5 additions

| Field | Meaning |
|-------|---------|
| `epoch_history` | Array of per-epoch records: `epoch`, `loss`, `lr`, `duration_s`, `cost_usd`, `gpu_util`, `ts` |
| `source` | `"trainer_sdk"` when produced by the Trainer; otherwise omitted / CLI |
| `epochs_completed` | Actual epochs finished (may be < planned on early stop) |
| `stopped_early` | Boolean |
| `final_loss` | Last epoch average loss |
| `checkpoint` | Path to last checkpoint if any |

Per-epoch data is the primary feed for future chip-design decisions: which phases of training burn compute.

## Quality-aware comparisons

The Trainer can record cost-to-quality without pretending that a cheaper run is
better:

```python
trainer = Trainer(
    model,
    optimizer,
    train_loader,
    val_loader=val_loader,
    target_val_loss=0.25,
)
```

Use `stop_at_target=True` when the first acceptable validation result should
end the run. `starfelt compare` then shows final validation loss and cost to
target when both runs provide those fields.

The CLI wrapper enforces `budget_usd_per_run` by asking the child process to
stop at the limit, then force-terminating it after a 10-second grace period.
Use `--no-budget` only when an operator intentionally accepts that risk.
