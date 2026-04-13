#!/usr/bin/env python3
"""Generate webMUSHRA paired-comparison batch configs from Marcussen pair CSVs.

This script reads the listening-dataset `pairs.csv` produced by the Marcussen
pipeline and writes one webMUSHRA YAML config per batch. The generated configs
assume the exported `wav/` directory is copied under the webMUSHRA asset root,
which defaults to `configs/resources/audio/`.

The main experiment rows are the cross-organ pairs (`same_organ_pair == False`).
Each batch also receives a small number of randomly selected same-organ control
pairs to act as sanity checks.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable


DEFAULT_INPUT_CSV = Path("outputs/listening_experiment_pairs/pairs.csv")
DEFAULT_OUTPUT_DIR = Path("src/webMUSHRA/configs/marcussen_batches")
DEFAULT_ASSET_PREFIX = "configs/resources/audio/marcussen_batches"
DEFAULT_BATCH_COUNT = 20
DEFAULT_CONTROL_MIN = 3
DEFAULT_CONTROL_MAX = 3
DEFAULT_SEED = 20260413


@dataclass(frozen=True, slots=True)
class PairRow:
    """One pair row from the Marcussen listening-dataset CSV."""

    row_index: int
    pair_id: int
    family: str
    division: str
    registration_raw: str
    pitch: str
    mic_location: str
    organ_a: str
    organ_b: str
    source_path_a: str
    source_path_b: str
    toot_wav_path_a: str
    toot_wav_path_b: str
    batch: str
    source_a_toot_index: str
    source_b_toot_index: str
    same_organ_pair: bool
    processing_chain: str
    group_id: str

    @property
    def trial_group_key(self) -> str:
        return self.group_id or "|".join(
            (
                self.family,
                self.division,
                self.registration_raw,
                self.pitch,
                self.mic_location,
            )
        )


@dataclass(frozen=True, slots=True)
class TrialAssignment:
    """A batch-local trial with a chosen reference orientation."""

    batch_index: int
    trial_index: int
    pair_row: PairRow
    reference_member: str
    reference_path: str
    nonreference_path: str
    reference_source_path: str
    nonreference_source_path: str
    reference_organ: str
    nonreference_organ: str


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate webMUSHRA paired-comparison YAML batches from Marcussen pairs.csv",
    )
    parser.add_argument(
        "--pairs-csv",
        type=Path,
        default=DEFAULT_INPUT_CSV,
        help=f"Input pair CSV path (default: {DEFAULT_INPUT_CSV})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for generated YAML and manifest files (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--asset-prefix",
        type=str,
        default=DEFAULT_ASSET_PREFIX,
        help=(
            "webMUSHRA path prefix prepended to each CSV `toot_wav_path_*` value. "
            f"Default: {DEFAULT_ASSET_PREFIX}"
        ),
    )
    parser.add_argument(
        "--batch-count",
        type=int,
        default=DEFAULT_BATCH_COUNT,
        help=f"Number of webMUSHRA config batches to create (default: {DEFAULT_BATCH_COUNT})",
    )
    parser.add_argument(
        "--control-min",
        type=int,
        default=DEFAULT_CONTROL_MIN,
        help=f"Minimum same-organ control pairs per batch (default: {DEFAULT_CONTROL_MIN})",
    )
    parser.add_argument(
        "--control-max",
        type=int,
        default=DEFAULT_CONTROL_MAX,
        help=f"Maximum same-organ control pairs per batch (default: {DEFAULT_CONTROL_MAX})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed for batching and orientation (default: {DEFAULT_SEED})",
    )
    return parser.parse_args()


def _load_pairs(csv_path: Path) -> list[PairRow]:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = [
            PairRow(
                row_index=index,
                pair_id=int(row["pair_id"]),
                family=row["family"],
                division=row["division"],
                registration_raw=row["registration_raw"],
                pitch=row["pitch"],
                mic_location=row["mic_location"],
                organ_a=row["organ_a"],
                organ_b=row["organ_b"],
                source_path_a=row["source_path_a"],
                source_path_b=row["source_path_b"],
                toot_wav_path_a=row["toot_wav_path_a"],
                toot_wav_path_b=row["toot_wav_path_b"],
                batch=row["batch"],
                source_a_toot_index=row["source_a_toot_index"],
                source_b_toot_index=row["source_b_toot_index"],
                same_organ_pair=row["same_organ_pair"].strip().lower() == "true",
                processing_chain=row["processing_chain"],
                group_id=row["group_id"],
            )
            for index, row in enumerate(reader, start=1)
        ]
    pair_ids = [row.pair_id for row in rows]
    if len(pair_ids) != len(set(pair_ids)):
        raise ValueError("Input CSV contains duplicate pair_id values")
    rows.sort(key=lambda row: row.pair_id)
    return rows


def _batch_targets(total_items: int, batch_count: int, rng: random.Random) -> list[int]:
    base, remainder = divmod(total_items, batch_count)
    targets = [base] * batch_count
    indices = list(range(batch_count))
    rng.shuffle(indices)
    for index in indices[:remainder]:
        targets[index] += 1
    return targets


def _control_targets(
    available_items: int,
    batch_count: int,
    minimum: int,
    maximum: int,
    rng: random.Random,
) -> list[int]:
    if minimum < 0 or maximum < minimum:
        raise ValueError("control bounds must satisfy 0 <= minimum <= maximum")
    desired_total = round(batch_count * ((minimum + maximum) / 2.0))
    total_to_allocate = min(available_items, desired_total, batch_count * maximum)
    return _batch_targets(total_to_allocate, batch_count, rng)


def _distribute_rows(
    rows: Iterable[PairRow],
    batch_targets: list[int],
    rng: random.Random,
    *,
    group_key: Callable[[PairRow], str],
) -> list[list[PairRow]]:
    total_required = sum(batch_targets)
    grouped_rows: dict[str, list[PairRow]] = defaultdict(list)
    for row in rows:
        grouped_rows[group_key(row)].append(row)

    groups = list(grouped_rows.values())
    rng.shuffle(groups)
    for group in groups:
        rng.shuffle(group)

    batches: list[list[PairRow]] = [[] for _ in batch_targets]
    group_counts: list[Counter[str]] = [Counter() for _ in batch_targets]
    placed = 0

    for group in groups:
        for row in group:
            if placed >= total_required:
                break

            candidates = [index for index, target in enumerate(batch_targets) if len(batches[index]) < target]
            if not candidates:
                break

            rng.shuffle(candidates)
            chosen_batch = min(
                candidates,
                key=lambda index: (group_counts[index][group_key(row)], len(batches[index])),
            )
            batches[chosen_batch].append(row)
            group_counts[chosen_batch][group_key(row)] += 1
            placed += 1

        if placed >= total_required:
            break

    if placed != total_required:
        raise ValueError(f"Could not allocate all required rows: placed {placed} of {total_required}")

    return batches


def _join_asset_path(asset_prefix: str, wav_path: str) -> str:
    return str(Path(asset_prefix) / Path(wav_path))


def _yaml_scalar(value: object) -> str:
    return json.dumps(value, ensure_ascii=False)


def _batch_asset_relpath(asset_prefix: str, batch_number: int, assignment: TrialAssignment, role: str) -> str:
    return str(
        Path(asset_prefix)
        / f"batch_{batch_number:02d}"
        / f"trial_{assignment.trial_index:03d}_pair_{assignment.pair_row.pair_id}_{role}.wav"
    )


def _copy_batch_audio_assets(
    assignments_by_batch: list[list[TrialAssignment]],
    *,
    pairs_csv_path: Path,
    webmushra_root: Path,
    asset_prefix: str,
) -> int:
    source_root = pairs_csv_path.parent
    asset_base_dir = webmushra_root / Path(asset_prefix)
    asset_base_dir.mkdir(parents=True, exist_ok=True)

    for stale_batch_dir in asset_base_dir.glob("batch_*"):
        if stale_batch_dir.is_dir():
            shutil.rmtree(stale_batch_dir)

    copied_files = 0
    for batch_number, assignments in enumerate(assignments_by_batch, start=1):
        for assignment in assignments:
            copies = [
                (
                    source_root / assignment.reference_path,
                    webmushra_root / _batch_asset_relpath(asset_prefix, batch_number, assignment, "reference"),
                ),
                (
                    source_root / assignment.nonreference_path,
                    webmushra_root / _batch_asset_relpath(asset_prefix, batch_number, assignment, "candidate"),
                ),
            ]
            for source_path, destination_path in copies:
                destination_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_path, destination_path)
                copied_files += 1

    return copied_files


def _trial_content(row: PairRow) -> str:
    details = [
        f"Family: {row.family}",
        f"Division: {row.division}",
        f"Registration: {row.registration_raw}",
        f"Pitch: {row.pitch}",
        f"Microphone: {row.mic_location}",
    ]
    return "<br/>".join(details)


def _trial_id(batch_number: int, assignment: TrialAssignment) -> str:
    return f"batch{batch_number:02d}_pair{assignment.pair_row.pair_id}_trial{assignment.trial_index:03d}"


def _intro_content(batch_number: int, trial_count: int) -> str:
    return (
        f"Marcussen listening test batch {batch_number:02d}.<br/><br/>"
        "Each trial provides one visible reference and two hidden options, A and B. "
        "Exactly one of A or B matches the reference. "
        "Use as many replays as needed, then select which hidden item matches the reference.<br/><br/>"
        f"This batch contains {trial_count} trials."
    )


def _assign_trials(
    batch_rows: list[PairRow],
    batch_index: int,
    rng: random.Random,
) -> list[TrialAssignment]:
    shuffled_rows = list(batch_rows)
    rng.shuffle(shuffled_rows)

    assignments: list[TrialAssignment] = []
    for trial_index, row in enumerate(shuffled_rows, start=1):
        reference_member = "a" if rng.random() < 0.5 else "b"
        if reference_member == "a":
            reference_path = row.toot_wav_path_a
            nonreference_path = row.toot_wav_path_b
            reference_source = row.source_path_a
            nonreference_source = row.source_path_b
            reference_organ = row.organ_a
            nonreference_organ = row.organ_b
        else:
            reference_path = row.toot_wav_path_b
            nonreference_path = row.toot_wav_path_a
            reference_source = row.source_path_b
            nonreference_source = row.source_path_a
            reference_organ = row.organ_b
            nonreference_organ = row.organ_a

        assignments.append(
            TrialAssignment(
                batch_index=batch_index,
                trial_index=trial_index,
                pair_row=row,
                reference_member=reference_member,
                reference_path=reference_path,
                nonreference_path=nonreference_path,
                reference_source_path=reference_source,
                nonreference_source_path=nonreference_source,
                reference_organ=reference_organ,
                nonreference_organ=nonreference_organ,
            )
        )
    return assignments


def _render_batch_yaml(
    assignments: list[TrialAssignment],
    batch_index: int,
    asset_prefix: str,
) -> str:
    batch_number = batch_index + 1
    batch_id = f"marcussen_batch_{batch_number:02d}"
    if not assignments:
        raise ValueError(f"Batch {batch_number:02d} has no trials")

    volume_stimulus = _batch_asset_relpath(asset_prefix, batch_number, assignments[0], "reference")
    lines = [
        f"testname: {_yaml_scalar(f'Marcussen AB Batch {batch_number:02d}')}",
        f"testId: {_yaml_scalar(batch_id)}",
        "bufferSize: 2048",
        "stopOnErrors: true",
        "showButtonPreviousPage: false",
        "language: \"en\"",
        "remoteService: \"service/write.php\"",
        "",
        "pages:",
        "  - type: generic",
        "    id: \"intro\"",
        "    name: \"Instructions\"",
        f"    content: {_yaml_scalar(_intro_content(batch_number, len(assignments)))}",
        "  - type: volume",
        "    id: \"volume\"",
        "    name: \"Volume\"",
        "    content: \"Adjust to a comfortable listening level before starting the trials.\"",
        f"    stimulus: {_yaml_scalar(volume_stimulus)}",
        "    defaultVolume: 0.5",
    ]

    for assignment in assignments:
        row = assignment.pair_row
        trial_id = _trial_id(batch_number, assignment)
        lines.extend(
            [
                "  - type: paired_comparison",
                f"    id: {_yaml_scalar(trial_id)}",
                f"    name: {_yaml_scalar(f'Trial {assignment.trial_index:03d}')}",
                f"    content: {_yaml_scalar(_trial_content(row))}",
                "    showWaveform: false",
                "    enableLooping: true",
                f"    reference: {_yaml_scalar(_batch_asset_relpath(asset_prefix, batch_number, assignment, 'reference'))}",
                "    stimuli:",
                f"      C1: {_yaml_scalar(_batch_asset_relpath(asset_prefix, batch_number, assignment, 'candidate'))}",
            ]
        )

    lines.extend(
        [
            "  - type: finish",
            "    name: \"Thank you\"",
            "    content: \"Thank you for completing this batch.\"",
            "    showResults: false",
            "    writeResults: true",
        ]
    )
    return "\n".join(lines) + "\n"


def _write_batch_manifest(output_dir: Path, assignments_by_batch: list[list[TrialAssignment]]) -> Path:
    manifest_path = output_dir / "batch_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "batch_number",
                "yaml_file",
                "trial_count",
                "main_pair_count",
                "same_organ_sanity_check_count",
                "pair_ids",
            ],
        )
        writer.writeheader()
        for batch_index, assignments in enumerate(assignments_by_batch, start=1):
            sanity_check_count = sum(assignment.pair_row.same_organ_pair for assignment in assignments)
            writer.writerow(
                {
                    "batch_number": batch_index,
                    "yaml_file": f"marcussen_batch_{batch_index:02d}.yaml",
                    "trial_count": len(assignments),
                    "main_pair_count": len(assignments) - sanity_check_count,
                    "same_organ_sanity_check_count": sanity_check_count,
                    "pair_ids": ";".join(str(assignment.pair_row.pair_id) for assignment in assignments),
                }
            )
    return manifest_path


def _write_trial_manifest(output_dir: Path, assignments_by_batch: list[list[TrialAssignment]], asset_prefix: str) -> Path:
    manifest_path = output_dir / "trial_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "batch_number",
                "yaml_file",
                "trial_index",
                "trial_id",
                "pair_id",
            ],
        )
        writer.writeheader()

        for batch_index, assignments in enumerate(assignments_by_batch, start=1):
            yaml_file = f"marcussen_batch_{batch_index:02d}.yaml"
            for assignment in assignments:
                row = assignment.pair_row
                writer.writerow(
                    {
                        "batch_number": batch_index,
                        "yaml_file": yaml_file,
                        "trial_index": assignment.trial_index,
                        "trial_id": _trial_id(batch_index, assignment),
                        "pair_id": row.pair_id,
                    }
                )
    return manifest_path


def main() -> int:
    args = _parse_args()
    if args.batch_count <= 0:
        raise ValueError("--batch-count must be > 0")

    rng = random.Random(args.seed)
    rows = _load_pairs(args.pairs_csv)
    main_rows = [row for row in rows if not row.same_organ_pair]
    control_rows = [row for row in rows if row.same_organ_pair]

    main_targets = _batch_targets(len(main_rows), args.batch_count, rng)
    control_targets = _control_targets(
        len(control_rows),
        args.batch_count,
        args.control_min,
        args.control_max,
        rng,
    )

    main_batches = _distribute_rows(main_rows, main_targets, rng, group_key=lambda row: row.trial_group_key)
    control_batches = _distribute_rows(control_rows, control_targets, rng, group_key=lambda row: row.trial_group_key)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for stale_yaml_path in args.output_dir.glob("marcussen_batch_*.yaml"):
        stale_yaml_path.unlink()

    assignments_by_batch: list[list[TrialAssignment]] = []
    for batch_index in range(args.batch_count):
        batch_rows = [*main_batches[batch_index], *control_batches[batch_index]]
        assignments = _assign_trials(batch_rows, batch_index, rng)
        assignments_by_batch.append(assignments)

    webmushra_root = args.output_dir.parent.parent
    copied_file_count = _copy_batch_audio_assets(
        assignments_by_batch,
        pairs_csv_path=args.pairs_csv,
        webmushra_root=webmushra_root,
        asset_prefix=args.asset_prefix,
    )

    for batch_index in range(args.batch_count):
        assignments = assignments_by_batch[batch_index]
        yaml_text = _render_batch_yaml(assignments, batch_index, args.asset_prefix)
        yaml_path = args.output_dir / f"marcussen_batch_{batch_index + 1:02d}.yaml"
        yaml_path.write_text(yaml_text, encoding="utf-8")

    batch_manifest = _write_batch_manifest(args.output_dir, assignments_by_batch)
    trial_manifest = _write_trial_manifest(args.output_dir, assignments_by_batch, args.asset_prefix)

    main_count = sum(len(batch) for batch in main_batches)
    control_count = sum(len(batch) for batch in control_batches)
    staged_asset_root = webmushra_root / Path(args.asset_prefix)

    print(
        f"Generated {args.batch_count} webMUSHRA batch configs in {args.output_dir} "
        f"using seed={args.seed}."
    )
    print(
        f"Main pairs allocated: {main_count} across {args.batch_count} batches "
        f"({min(main_targets)}-{max(main_targets)} per batch)."
    )
    print(
        f"Same-organ sanity-check pairs allocated: {control_count} across {args.batch_count} batches "
        f"({min(control_targets)}-{max(control_targets)} per batch)."
    )
    if control_count < args.batch_count * args.control_min:
        print(
            "Warning: the input CSV does not contain enough same-organ rows to satisfy "
            f"{args.control_min} controls per batch; the generator used all available control rows instead."
        )
    print(f"Copied {copied_file_count} WAV files into {staged_asset_root}")
    print(f"Batch manifest: {batch_manifest}")
    print(f"Trial manifest: {trial_manifest}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
