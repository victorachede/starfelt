from __future__ import annotations

import warnings
from dataclasses import dataclass

# Static sketch prices until live APIs are wired — verify before spending.
PRICES_LAST_UPDATED = "2026-09-08"

CATALOG_WARNING = (
    f"Prices are estimates (last updated {PRICES_LAST_UPDATED}) — "
    "verify before committing spend."
)


@dataclass
class ProviderOffer:
    name: str
    gpu: str
    usd_per_hour: float
    spot: bool = False


CATALOG = [
    ProviderOffer("runpod", "A100-40G", 1.09, spot=True),
    ProviderOffer("lambda", "A100-40G", 1.25, spot=False),
    ProviderOffer("aws", "p4d.24xlarge", 3.50, spot=True),
    ProviderOffer("gcp", "a2-highgpu-1g", 3.20, spot=True),
]


def pick_cheapest(
    preferred: list[str] | None = None,
    allow_spot: bool = True,
) -> ProviderOffer:
    warnings.warn(CATALOG_WARNING, UserWarning, stacklevel=2)
    offers = [o for o in CATALOG if allow_spot or not o.spot]
    if preferred:
        ranked = [o for p in preferred for o in offers if o.name == p]
        if ranked:
            return min(ranked, key=lambda o: o.usd_per_hour)
    return min(offers, key=lambda o: o.usd_per_hour)
