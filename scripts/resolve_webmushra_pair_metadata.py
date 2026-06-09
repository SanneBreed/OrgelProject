#!/usr/bin/env python3
"""Append original Marcussen pair metadata to webMUSHRA paired-distance CSVs."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


DEFAULT_PAIRS_CSV = Path("outputs/listening_experiment_pairs/pairs.csv")
DEFAULT_TRIAL_MANIFEST_CSV = Path("src/webMUSHRA/configs/trial_manifest.csv")

JOINED_FIELDS = [
    "original_family",
    "original_division",
    "original_registration",
    "original_pitch",
    "original_organ_a",
    "original_organ_b",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Left join a webMUSHRA paired-distance result CSV with the original "
            "Marcussen pairs CSV by pair_id."
        )
    )
    parser.add_argument("results_csv", type=Path, help="webMUSHRA paired_distance*.csv file")
    parser.add_argument(
        "--pairs-csv",
        type=Path,
        default=DEFAULT_PAIRS_CSV,
        help=f"Original Marcussen pairs CSV (default: {DEFAULT_PAIRS_CSV})",
    )
    parser.add_argument(
        "--trial-manifest-csv",
        type=Path,
        default=DEFAULT_TRIAL_MANIFEST_CSV,
        help=(
            "Generated webMUSHRA trial manifest used as a fallback for page IDs "
            f"(default: {DEFAULT_TRIAL_MANIFEST_CSV})"
        ),
    )
    parser.add_argument(
        "-o",
        "--output-csv",
        type=Path,
        help="Output CSV path (default: <results_csv stem>_resolved.csv beside input)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if any result row cannot be resolved to a pair metadata row",
    )
    return parser.parse_args()


def _read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no CSV header")
        return list(reader.fieldnames), list(reader)


def _load_pairs(path: Path) -> dict[str, dict[str, str]]:
    _, rows = _read_rows(path)
    pairs: dict[str, dict[str, str]] = {}
    for row in rows:
        pair_id = row.get("pair_id", "").strip()
        if not pair_id:
            continue
        if pair_id in pairs:
            raise ValueError(f"{path} contains duplicate pair_id {pair_id!r}")
        pairs[pair_id] = row
    return pairs


def _load_manifest_pair_ids(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}

    _, rows = _read_rows(path)
    page_to_pair: dict[str, str] = {}
    for row in rows:
        page_id = row.get("page_id", "").strip()
        pair_id = row.get("pair_id", "").strip()
        if page_id and pair_id:
            page_to_pair[page_id] = pair_id
    return page_to_pair


def _pair_id_for_result(row: dict[str, str], manifest_pair_ids: dict[str, str]) -> str:
    pair_id = row.get("pair_id", "").strip()
    if pair_id:
        return pair_id

    trial_id = row.get("trial_id", "").strip()
    if trial_id in manifest_pair_ids:
        return manifest_pair_ids[trial_id]
    return trial_id if trial_id.isdecimal() else ""


def _metadata_for_pair(pair_row: dict[str, str] | None) -> dict[str, str]:
    if pair_row is None:
        return {field: "" for field in JOINED_FIELDS}
    return {
        "original_family": pair_row.get("family", ""),
        "original_division": pair_row.get("division", ""),
        "original_registration": pair_row.get("registration_raw", ""),
        "original_pitch": pair_row.get("pitch", ""),
        "original_organ_a": pair_row.get("organ_a", ""),
        "original_organ_b": pair_row.get("organ_b", ""),
    }


def _output_path_for(results_csv: Path, explicit_output: Path | None) -> Path:
    if explicit_output is not None:
        return explicit_output
    return results_csv.with_name(f"{results_csv.stem}_resolved{results_csv.suffix}")


def main() -> int:
    args = _parse_args()
    output_csv = _output_path_for(args.results_csv, args.output_csv)

    input_fields, result_rows = _read_rows(args.results_csv)
    pairs = _load_pairs(args.pairs_csv)
    manifest_pair_ids = _load_manifest_pair_ids(args.trial_manifest_csv)

    output_fields = [field for field in input_fields if field not in JOINED_FIELDS] + JOINED_FIELDS
    unresolved: list[tuple[int, str, str]] = []

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fields)
        writer.writeheader()

        for line_number, row in enumerate(result_rows, start=2):
            resolved_pair_id = _pair_id_for_result(row, manifest_pair_ids)
            pair_row = pairs.get(resolved_pair_id)
            if pair_row is None:
                unresolved.append((line_number, row.get("trial_id", ""), resolved_pair_id))
            writer.writerow({**row, **_metadata_for_pair(pair_row)})

    print(f"Wrote {len(result_rows)} row(s) to {output_csv}")
    if unresolved:
        print(
            f"Warning: {len(unresolved)} row(s) could not be resolved from {args.pairs_csv}:",
            file=sys.stderr,
        )
        for line_number, trial_id, resolved_pair_id in unresolved[:10]:
            print(
                f"  line {line_number}: trial_id={trial_id!r}, resolved_pair_id={resolved_pair_id!r}",
                file=sys.stderr,
            )
        if len(unresolved) > 10:
            print(f"  ... {len(unresolved) - 10} more", file=sys.stderr)
        if args.strict:
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
