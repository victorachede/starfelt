"""Data pipeline helpers.

TODO (Stage 2):
- Real async prefetch with background thread/process
- Automatic dataset sharding integrated with DistributedSampler
- Content-hash redundant sample skipping for common dataset types
- PyTorch DataLoader monkey-patch / wrapper that applies hints by default
"""

from __future__ import annotations

import warnings
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


def prefetch(it: Iterator[T], buffer_size: int = 8) -> Iterator[T]:
    """Passthrough until Stage 2 implements real prefetch."""
    warnings.warn(
        "prefetch not yet implemented — using passthrough",
        UserWarning,
        stacklevel=2,
    )
    _ = buffer_size
    yield from it
