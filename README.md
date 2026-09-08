# Starfelt

**Training efficiency layer.** Point it at your PyTorch training script — Starfelt analyzes, monitors, and tracks cost. No infra rewrite.

```bash
pip install -e .
starfelt init
starfelt run train.py
```

> Private · Stage 1 — analysis + runtime monitoring · **one user who says “this saved me money”**

---

## Stage 1 non-negotiables

- CLI is zero-friction: `starfelt run train.py` just works  
- **PyTorch first** — everything else later  
- Dashboard can be a **terminal UI** to start  
- Ship **pre-run analysis + runtime monitoring** before cloud orchestration  

### Explicitly later

- Multi-provider orchestration (RunPod / AWS / …)  
- Enterprise tier / SLA packaging  
- Full web cost console  

---

## Commands

| Command | |
|---------|--|
| `starfelt init` | Write `starfelt.yaml` + `.starfelt/` |
| `starfelt analyze train.py` | Pre-run checks (batch, LR, data, cost sketch) |
| `starfelt run train.py` | Wrap process, track cost, emit run id |
| `starfelt cost` | Terminal cost history |

```bash
starfelt analyze examples/train_toy.py
starfelt run examples/train_toy.py
starfelt cost
```

---

## Layout

```
starfelt/
  cli/           # init | analyze | run | cost
  core/          # analysis, runner, cost, data helpers
  providers/     # stubs only — not Stage 1 product surface
examples/
docs/STAGE1.md
```

---

## Stack (Stage 1)

| Piece | Choice |
|-------|--------|
| CLI / SDK | Python |
| Dashboard | **Terminal first** (`starfelt cost` + rich tables) |
| Run history | **Local** `.starfelt/` → optional **Supabase** when you want sync across machines |

Same Supabase org as ASKTC is fine later; don’t block Stage 1 on it.

---

## Goal

One serious training user: *“this saved me money.”* That’s the exit criterion for Stage 1.
