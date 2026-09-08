# Starfelt

**Training efficiency layer.** Point it at your training script — Starfelt makes the run cheaper, faster, and smarter. No infra rewrite. No new framework.

```bash
pip install starfelt
starfelt init
starfelt run train.py
```

> Private · Stage 1 — Training Efficiency Layer

---

## What it does

| Layer | Behavior |
|-------|----------|
| **Pre-run analysis** | Flags bad batch size, risky LR, data bottlenecks; estimates cost/time |
| **Runtime optimization** | Dynamic LR, grad accumulation, early-stop signals, checkpoint cadence |
| **Data pipeline** | Prefetch/cache, sharding, skip redundant samples |
| **Cost dashboard** | Live spend, projected total, vs baseline without Starfelt |
| **Multi-provider** | AWS / GCP / Lambda Labs / RunPod — cheapest fit + spot resume |

## What it is not

- Not a cloud provider  
- Not a training framework  
- Works with **PyTorch, JAX, anything** you already run  

## Quick start (local)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
starfelt init
starfelt analyze examples/train_toy.py
starfelt run examples/train_toy.py
```

## Package layout

```
starfelt/
  cli/           # starfelt init | analyze | run | cost
  core/          # analysis, runtime hooks, data pipeline, cost model
  providers/     # compute backend adapters (stubs → real)
  dashboard/     # local cost history
examples/
docs/
```

## Business model (indicative)

| Tier | |
|------|--|
| Free | 5 runs / month |
| Pro | $49 / month unlimited |
| Enterprise | Custom + SLA |

## Passive learning → chip

Every run contributes workload patterns and which optimizations actually move cost — signal for later silicon.

---

Stage 1 scope is the **wrapper CLI + SDK**. Cloud orchestration and full dashboard ship as the adapters harden.
