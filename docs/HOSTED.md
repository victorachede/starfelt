# Hosted run history (optional)

Local JSON under `.starfelt/` is the default. Hosted sync is **opt-in**.

## Setup

1. Create a Supabase project (same org as ASKTC is fine).
2. SQL:

```sql
create table if not exists starfelt_runs (
  run_id text primary key,
  script text,
  cost_usd double precision,
  framework text,
  workload_id text,
  payload jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now()
);

alter table starfelt_runs enable row level security;
-- For a private service key from CLI, you may use service role and skip user RLS.
-- Or add policies for authenticated users later.
```

3. CLI:

```bash
starfelt login
starfelt sync
```

Credentials live in `~/.starfelt/auth.json` (not committed).

## PyPI

```bash
# after CI is green on main
git tag v0.1.0
git push origin v0.1.0
```

Configure trusted publisher on PyPI for `victorachede/starfelt` + workflow `publish.yml`.
