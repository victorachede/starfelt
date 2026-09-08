# Starfelt Stage 1 — Training Efficiency Layer

## Scope

CLI + SDK wrapper around existing training scripts:

1. Pre-run analysis  
2. Runtime monitoring hooks (env signals; deeper auto-tune next)  
3. Data pipeline helpers  
4. Local cost history  
5. Provider catalog stubs (pick cheapest)

## Non-goals (Stage 1)

- Owning GPUs / being a cloud  
- Replacing PyTorch / JAX  
- Full multi-cloud live APIs  

## CLI

- `starfelt init`
- `starfelt analyze train.py`
- `starfelt run train.py`
- `starfelt cost`
