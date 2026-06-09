#!/usr/bin/env python3
"""One-off right join of Hans and Noah paired-distance batch 00 results."""

from __future__ import annotations

import csv
from pathlib import Path


BATCH_DIR = Path("outputs/webmushra_pi_results_2026-06-08_17-19/results/marcussen_batch_00")
HANS_CSV = BATCH_DIR / "paired_distance_hans_resolved.csv"
NOAH_CSV = BATCH_DIR / "paired_distance_noah.csv"
OUTPUT_CSV = BATCH_DIR / "paired_distance_hans_noah_joined.csv"

OUTPUT_FIELDS = [
    "trial_id",
    "pair_id",
    "session_test_id",
    "stimulus_1",
    "stimulus_2",
    "hans_distance",
    "noah_distance",
    "hans_distance_comment",
    "noah_distance_comment",
    "hans_distance_time",
    "noah_distance_time",
    "original_family",
    "original_division",
    "original_registration",
    "original_pitch",
    "original_organ_a",
    "original_organ_b",
]


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no CSV header")
        return list(reader)


def main() -> int:
    hans_by_trial_id = {row["trial_id"]: row for row in _read_rows(HANS_CSV)}
    noah_rows = _read_rows(NOAH_CSV)

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()

        for noah_row in noah_rows:
            trial_id = noah_row["trial_id"]
            hans_row = hans_by_trial_id.get(trial_id, {})
            writer.writerow(
                {
                    "trial_id": trial_id,
                    "pair_id": noah_row.get("pair_id") or hans_row.get("pair_id", ""),
                    "session_test_id": noah_row.get("session_test_id") or hans_row.get("session_test_id", ""),
                    "stimulus_1": noah_row.get("stimulus_1") or hans_row.get("stimulus_1", ""),
                    "stimulus_2": noah_row.get("stimulus_2") or hans_row.get("stimulus_2", ""),
                    "hans_distance": hans_row.get("distance", ""),
                    "noah_distance": noah_row.get("distance", ""),
                    "hans_distance_comment": hans_row.get("distance_comment", ""),
                    "noah_distance_comment": noah_row.get("distance_comment", ""),
                    "hans_distance_time": hans_row.get("distance_time", ""),
                    "noah_distance_time": noah_row.get("distance_time", ""),
                    "original_family": hans_row.get("original_family", ""),
                    "original_division": hans_row.get("original_division", ""),
                    "original_registration": hans_row.get("original_registration", ""),
                    "original_pitch": hans_row.get("original_pitch", ""),
                    "original_organ_a": hans_row.get("original_organ_a", ""),
                    "original_organ_b": hans_row.get("original_organ_b", ""),
                }
            )

    missing_hans = [row["trial_id"] for row in noah_rows if row["trial_id"] not in hans_by_trial_id]
    print(f"Wrote {len(noah_rows)} row(s) to {OUTPUT_CSV}")
    if missing_hans:
        print(f"No Hans row for {len(missing_hans)} Noah trial_id(s): {', '.join(missing_hans)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
