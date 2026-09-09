# Starfelt Quickstart

## Install

```bash
pip install git+https://github.com/victorachede/starfelt.git
```

---

## Colab (recommended starting point)

```python
from starfelt import Trainer
from starfelt.notebook import StarfeltDisplay

display = StarfeltDisplay()

trainer = Trainer(
    model=model,
    optimizer=optimizer,
    train_loader=train_loader,
    loss_fn=loss_fn,         # optional if model returns scalar loss
    epochs=10,
    auto_mount_drive=True,   # saves checkpoints to Drive automatically
    notebook_display=display,
)
result = trainer.fit()
```

That's it. Starfelt will:
- Warn you about bad hyperparameters before training starts
- Show a live table updating every epoch (loss, val, LR, cost, GPU)
- Save checkpoints to Google Drive automatically
- Stop early if loss plateaus

---

## If Colab disconnects mid-run

Come back, remount Drive, then:

```python
import os
os.environ["STARFELT_RESUME_FROM"] = f"/content/drive/MyDrive/.starfelt/checkpoints/{run_id}/latest.pt"
os.environ["STARFELT_RUN_ID"] = run_id  # printed at start of previous run

trainer = Trainer(model, optimizer, train_loader, epochs=10)
result = trainer.fit()
# picks up from where it stopped
```

Your `run_id` is printed at the start of every run and in `result.run_id`.

---

## Local / SSH (no Colab)

```bash
starfelt init
starfelt analyze train.py      # catch issues before wasting compute
starfelt run train.py          # live cost ticker in terminal
starfelt status                # see active run + history
starfelt compare run_a run_b   # which run was cheaper and why
```

---

## Trainer parameters

| Parameter | Default | What it does |
|-----------|---------|--------------|
| `model` | required | PyTorch model |
| `optimizer` | required | any PyTorch optimizer |
| `train_loader` | required | PyTorch DataLoader |
| `loss_fn` | None | loss function — skip if model returns scalar loss |
| `val_loader` | None | enables validation loss + val-based early stop |
| `epochs` | 1 | number of training epochs |
| `amp` | False | mixed precision — faster on CUDA |
| `grad_accum_steps` | 1 | gradient accumulation steps |
| `early_stop_patience` | None | stop if val loss doesn't improve for N evals |
| `checkpoint_every_epochs` | 1 | how often to save checkpoints |
| `auto_mount_drive` | False | mount Google Drive automatically in Colab |
| `notebook_display` | None | pass `StarfeltDisplay()` for live Jupyter output |
| `wandb` | False | log to Weights & Biases if installed |

---

## Common errors

**`Trainer requires PyTorch`**
Install PyTorch first: `pip install torch`

**`No checkpoint found`**
Drive wasn't mounted before the run. Use `auto_mount_drive=True` next time.

**`loss_fn not provided`**
Either pass `loss_fn=nn.CrossEntropyLoss()` or make your model return a scalar loss directly.

**Colab warning: Drive not mounted**
Run `from starfelt.notebook import mount_drive; mount_drive()` before training, or pass `auto_mount_drive=True` to Trainer.

---

## What gets saved

Every run writes to `.starfelt/runs/{run_id}.json` — framework, epochs, loss curve, cost, GPU utilization, model size. In Colab with Drive mounted, checkpoints go to `/content/drive/MyDrive/.starfelt/checkpoints/{run_id}/`.
