# Stage 1 — focus

## Ship

1. Pre-run analysis (batch, LR, dataloader, cost sketch)  
2. Runtime monitoring (wrap process, env signals, duration/cost)  
3. Terminal cost history (`starfelt cost`)  
4. PyTorch-oriented checks and examples  

## Defer

- Multi-provider orchestration  
- Enterprise pricing UI  
- Web dashboard (until terminal loop is proven)  
- Non-PyTorch frameworks  

## Dashboard decision

**Terminal first.** Web only after someone runs real jobs and asks for shareable history.

## Backend decision

**Local JSON under `.starfelt/` for Stage 1.**  
Optional Supabase (same muscle as ASKTC) when you need multi-device history or a hosted free/pro gate — not required to get the first “saved me money” quote.
