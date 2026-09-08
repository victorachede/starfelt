"""Data pipeline helpers — prefetch, shard, skip redundant samples.

Stage 1 ships interfaces + light utilities. Deep framework hooks follow.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, TypeVar

T = TypeVar("T")


@dataclass
class PipelineHints:
    num_workers: int = 4
    prefetch_factor: int = 2
    pin_memory: bool = True
    skip_redundant: bool = True


def shard_indices(n: int, rank: int, world_size: int) -> range:
    """Simple index shard for dataset partitioning."""
    if world_size <= 0:
        raise ValueError("world_size must be >= 1")
    return range(rank, n, world_size)


def dedupe_keep_order(items: Iterable[T]) -> list[T]:
    seen: set[T] = set()
    out: list[T] = []
    for x in items:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def prefetch_stub(it: Iterator[T], buffer_size: int = 8) -> Iterator[T]:
    """Placeholder for async prefetch; yields from iterator as-is in Stage 1."""
    yield from it
