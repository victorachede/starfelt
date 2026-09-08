"""Compute providers — Stage 1: stubs only. Orchestration is explicitly deferred."""

from starfelt.providers.base import ProviderOffer, pick_cheapest

__all__ = ["pick_cheapest", "ProviderOffer"]
