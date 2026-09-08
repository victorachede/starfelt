"""Optional hosted run history (Supabase REST). Opt-in only."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def auth_path() -> Path:
    return Path.home() / ".starfelt" / "auth.json"


def load_auth() -> dict[str, Any] | None:
    path = auth_path()
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def save_auth(data: dict[str, Any]) -> Path:
    path = auth_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def clear_auth() -> None:
    path = auth_path()
    if path.exists():
        path.unlink()


def sync_runs(runs: list[dict[str, Any]]) -> tuple[int, str]:
    """Push runs to Supabase table `starfelt_runs` via REST.

    Auth file keys: supabase_url, supabase_key (service or anon with RLS).
    Returns (count_synced, message).
    """
    auth = load_auth()
    if not auth:
        return 0, "Not logged in — run: starfelt login"
    url = (auth.get("supabase_url") or "").rstrip("/")
    key = auth.get("supabase_key") or ""
    if not url or not key:
        return 0, "auth.json missing supabase_url or supabase_key"

    endpoint = f"{url}/rest/v1/starfelt_runs"
    headers = {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Prefer": "resolution=merge-duplicates,return=minimal",
    }
    # Upsert on run_id if unique constraint exists
    payload = []
    for r in runs:
        payload.append(
            {
                "run_id": r.get("run_id"),
                "payload": r,
                "script": r.get("script"),
                "cost_usd": r.get("cost_usd"),
                "framework": r.get("framework"),
                "workload_id": r.get("workload_id"),
            }
        )
    if not payload:
        return 0, "No local runs to sync"

    req = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            _ = resp.read()
        return len(payload), f"Synced {len(payload)} run(s) → {endpoint}"
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        return 0, f"HTTP {e.code}: {body[:300]}"
    except urllib.error.URLError as e:
        return 0, f"Network error: {e.reason}"
