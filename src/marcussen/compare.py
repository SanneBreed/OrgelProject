"""Pairwise comparison runner with streaming CSV output."""

from __future__ import annotations

from collections import Counter
import csv
from itertools import combinations
import logging
from pathlib import Path
from typing import Any, Iterator

import librosa
from tqdm.auto import tqdm

from .dataset import MarcussenDataset
from .parsing import ParsedItem

logger = logging.getLogger(__name__)


def _placeholder_distance(item_a: ParsedItem, item_b: ParsedItem) -> float:
    """Return a constant placeholder score.

    # TODO: Replace this placeholder with the real distance implementation.
    # TODO: Before running a real metric, we may need to trim clips and/or
    #       write preprocessed snippets to temporary files.
    """
    y_a, sr_a = librosa.load(item_a.path, sr=None, mono=True)
    y_b, sr_b = librosa.load(item_b.path, sr=None, mono=True)
    _ = (y_a, sr_a, y_b, sr_b)
    return 999.0


def compare_pair(item_a: ParsedItem, item_b: ParsedItem, metric: str = "placeholder") -> float:
    """Compute one pairwise comparison score."""
    family = item_a.meta.get("family", "_")
    organ_a = item_a.meta.get("organ_id", "_")
    organ_b = item_b.meta.get("organ_id", "_")
    logger.debug(
        "compare_pair family=%s differences=(organ_id:%s vs organ_id:%s)",
        family,
        organ_a,
        organ_b,
    )
    if metric != "placeholder":
        raise ValueError(f"Unsupported metric: {metric!r}")
    return _placeholder_distance(item_a, item_b)


def _is_cross_organ_pair(item_a: ParsedItem, item_b: ParsedItem) -> bool:
    organ_a = item_a.meta.get("organ_id")
    organ_b = item_b.meta.get("organ_id")
    return organ_a not in (None, "") and organ_b not in (None, "") and organ_a != organ_b


def _iter_pairs(
    items: list[ParsedItem],
) -> Iterator[tuple[ParsedItem, ParsedItem]]:
    yield from (
        (item_a, item_b)
        for item_a, item_b in combinations(items, 2)
        if _is_cross_organ_pair(item_a, item_b)
    )


def _cross_organ_pair_count(items: list[ParsedItem]) -> int:
    counts = Counter(
        str(item.meta["organ_id"])
        for item in items
        if item.meta.get("organ_id") not in (None, "")
    )
    total = sum(counts.values())
    same_organ_pairs = sum(count * count for count in counts.values())
    return (total * total - same_organ_pairs) // 2


def run_within_group(
    dataset: MarcussenDataset,
    out_csv_path: str | Path,
    metric: str = "placeholder",
    max_pairs: int | None = None,
) -> dict[str, Any]:
    """Compare cross-organ pairs within each metadata class and stream CSV rows."""
    if max_pairs is not None and max_pairs < 0:
        raise ValueError("max_pairs must be >= 0")

    out_path = Path(out_csv_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    grouped = dataset.class_groups()

    fieldnames = [
        "family",
        "division",
        "registration_raw",
        "pitch",
        "mic_location",
        "organ_a",
        "organ_b",
        "metric",
        "score",
        "status",
        "error",
        "path_a",
        "path_b",
        "group_id",
    ]

    rows_written = 0
    error_rows = 0
    considered_groups = 0
    total_pairs_available = sum(
        _cross_organ_pair_count(items)
        for items in grouped.values()
        if len(items) >= 2
    )
    total_pairs = (
        total_pairs_available
        if max_pairs is None
        else min(total_pairs_available, max_pairs)
    )

    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        with tqdm(total=total_pairs, desc="Comparing pairs", unit="pair") as progress:
            for group_id, items in grouped.items():
                if len(items) < 2:
                    continue
                if max_pairs is not None and rows_written >= max_pairs:
                    break

                considered_groups += 1
                for item_a, item_b in _iter_pairs(items):
                    if max_pairs is not None and rows_written >= max_pairs:
                        break
                    row = {
                        "family": item_a.meta.get("family", ""),
                        "division": item_a.meta.get("division", ""),
                        "registration_raw": item_a.meta.get("registration_raw", ""),
                        "pitch": item_a.meta.get("pitch", ""),
                        "mic_location": item_a.meta.get("mic_location", ""),
                        "organ_a": item_a.meta.get("organ_id", ""),
                        "organ_b": item_b.meta.get("organ_id", ""),
                        "metric": metric,
                        "score": "",
                        "status": "ok",
                        "error": "",
                        "path_a": item_a.path,
                        "path_b": item_b.path,
                        "group_id": group_id,
                    }

                    try:
                        score = compare_pair(item_a, item_b, metric=metric)
                        row["score"] = f"{score:.8f}"
                    except Exception as exc:
                        row["status"] = "error"
                        row["error"] = f"{type(exc).__name__}: {exc}"
                        error_rows += 1

                    writer.writerow(row)
                    rows_written += 1
                    progress.update(1)
                    if rows_written % 50 == 0:
                        handle.flush()

        handle.flush()

    logger.info(
        "Finished compare run: groups=%d rows=%d errors=%d out=%s",
        considered_groups,
        rows_written,
        error_rows,
        out_path,
    )
    return {
        "groups": considered_groups,
        "rows": rows_written,
        "errors": error_rows,
        "out_csv": str(out_path),
    }
