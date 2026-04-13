"""Build pair relationships between dataset items after grouping.

This module does not scan or parse the dataset itself. It takes already-known
items, such as parsed source files or prepared listening clips, and provides
the common rules for turning them into pair candidates for comparison or export.
"""

from __future__ import annotations

from collections import Counter
from itertools import combinations
from typing import Any, Iterator, TypeVar


T = TypeVar("T")


def _organ_id_of(item: Any) -> str | None:
    direct = getattr(item, "organ_id", None)
    if direct not in (None, ""):
        return str(direct)
    meta = getattr(item, "meta", None)
    if isinstance(meta, dict):
        value = meta.get("organ_id")
        return None if value in (None, "") else str(value)
    return None


def is_cross_organ_pair(item_a: Any, item_b: Any) -> bool:
    """Return whether two items come from different known organs."""
    organ_a = _organ_id_of(item_a)
    organ_b = _organ_id_of(item_b)
    return organ_a not in (None, "") and organ_b not in (None, "") and organ_a != organ_b


def iter_all_pairs(items: list[T]) -> Iterator[tuple[T, T]]:
    """Yield all unique unordered pairs."""
    yield from combinations(items, 2)


def iter_cross_organ_pairs(items: list[T]) -> Iterator[tuple[T, T]]:
    """Yield unique unordered pairs restricted to different organs."""
    yield from (
        (item_a, item_b)
        for item_a, item_b in combinations(items, 2)
        if is_cross_organ_pair(item_a, item_b)
    )


def all_pair_count(items: list[T]) -> int:
    """Count unique unordered pairs."""
    total = len(items)
    return (total * (total - 1)) // 2


def cross_organ_pair_count(items: list[T]) -> int:
    """Count unique unordered pairs restricted to different organs."""
    counts = Counter(
        str(_organ_id_of(item))
        for item in items
        if _organ_id_of(item) not in (None, "")
    )
    total = sum(counts.values())
    same_organ_pairs = sum(count * count for count in counts.values())
    return (total * total - same_organ_pairs) // 2
