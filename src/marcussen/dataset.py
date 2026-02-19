"""Dataset utilities for scanning, grouping, and sampling recordings."""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from pathlib import Path
import random
from typing import Any, Iterator

from .parsing import ParsedItem, parse_filename

logger = logging.getLogger(__name__)

DEFAULT_GROUP_KEYS: tuple[str, ...] = (
    "family",
    "registration_raw",
    "division",
    "pitch",
    "mic_location",
)

SKIP_FILENAME_SUBSTRINGS: tuple[str, ...] = ("aanspraaktest",)


def make_group_id(meta: dict[str, Any], keys: list[str] | tuple[str, ...] | None = None) -> str:
    """Build a deterministic class/group ID from selected metadata keys."""
    group_keys = tuple(keys) if keys is not None else DEFAULT_GROUP_KEYS
    segments: list[str] = []
    for key in group_keys:
        value = meta.get(key)
        if isinstance(value, list):
            value_str = "+".join(str(v) for v in value)
        elif value is None or value == "":
            value_str = "_"
        else:
            value_str = str(value)
        segments.append(f"{key}={value_str}")
    return "|".join(segments)


@dataclass
class MarcussenDataset:
    """Filesystem-backed wrapper for class-based comparison workflows."""

    root: str | Path
    group_keys: list[str] = field(default_factory=lambda: list(DEFAULT_GROUP_KEYS))

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self._items: list[ParsedItem] | None = None

    def _scan(self) -> list[ParsedItem]:
        files = sorted(
            (path for path in self.root.rglob("*") if path.is_file() and path.suffix.lower() == ".flac"),
            key=lambda p: str(p),
        )
        kept_files = [
            path
            for path in files
            if not any(marker in path.stem.lower() for marker in SKIP_FILENAME_SUBSTRINGS)
        ]
        logger.info(
            "Scanning %s, found %d FLAC files, keeping %d after skip filters",
            self.root,
            len(files),
            len(kept_files),
        )
        return [parse_filename(str(path)) for path in kept_files]

    def _ensure_scanned(self) -> None:
        if self._items is None:
            self._items = self._scan()

    def iter_flat_items(self) -> Iterator[ParsedItem]:
        """Yield flat files in scan order; mainly for debugging/indexing."""
        self._ensure_scanned()
        assert self._items is not None
        return iter(self._items)

    def flat_items_list(self) -> list[ParsedItem]:
        """Return a concrete flat file list (not grouped for comparison)."""
        self._ensure_scanned()
        assert self._items is not None
        return list(self._items)

    def iter_items(self) -> Iterator[ParsedItem]:
        """Backward-compatible alias for `iter_flat_items()`."""
        return self.iter_flat_items()

    def items_list(self) -> list[ParsedItem]:
        """Backward-compatible alias for `flat_items_list()`."""
        return self.flat_items_list()

    def class_groups(self, min_organ_count: int = 2) -> dict[str, list[ParsedItem]]:
        """Group by class keys and keep groups with at least `min_organ_count` organs."""
        grouped: dict[str, list[ParsedItem]] = {}
        for item in self.iter_flat_items():
            group_id = make_group_id(item.meta, self.group_keys)
            grouped.setdefault(group_id, []).append(item)

        filtered: dict[str, list[ParsedItem]] = {}
        for group_id, items in grouped.items():
            organ_ids = {
                str(item.meta["organ_id"])
                for item in items
                if item.meta.get("organ_id") not in (None, "")
            }
            if len(organ_ids) >= min_organ_count:
                filtered[group_id] = items
        return filtered

    def iter_class_groups(self, min_organ_count: int = 2) -> Iterator[tuple[str, list[ParsedItem]]]:
        """Yield comparison class groups with multiple `organ_id` values."""
        yield from self.class_groups(min_organ_count=min_organ_count).items()

    def groups(self) -> dict[str, list[ParsedItem]]:
        """Backward-compatible alias for `class_groups()`."""
        return self.class_groups()

    def sample(self, n: int, seed: int | None = None) -> list[ParsedItem]:
        """Randomly sample `n` items (or fewer if dataset is smaller)."""
        items = self.flat_items_list()
        if n <= 0:
            return []
        if n >= len(items):
            return items
        rng = random.Random(seed)
        return rng.sample(items, n)
