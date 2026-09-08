"""Compute providers — RunPod, Lambda, AWS, GCP adapters (stubs in Stage 1)."""

from starfelt.providers.base import pick_cheapest, ProviderOffer

__all__ = ["pick_cheapest", "ProviderOffer"]
