#!/usr/bin/env python3
"""Generate webMUSHRA paired-distance batch configs from Marcussen pairs.csv.

The input CSV is expected to contain explicit pair roles:

- `cross_organ_main`
- `same_pipe_reference`
- `same_organ_anchor`

Each generated batch contains:

- one non-saved same-pipe practice example after the volume page
- one non-saved same-organ anchor practice example after that
- the scored main trials
- two scored same-pipe hidden references
- two scored same-organ hidden anchors
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
DEFAULT_OUTPUT_DIR = Path("src/webMUSHRA/configs")
DEFAULT_ASSET_PREFIX = "configs/resources/audio/marcussen_batches"
DEFAULT_BATCH_COUNT = 20
DEFAULT_SAME_PIPE_PER_BATCH = 2
DEFAULT_SAME_ORGAN_ANCHOR_PER_BATCH = 2
DEFAULT_DEMO_SAME_PIPE_PER_BATCH = 1
DEFAULT_DEMO_SAME_ORGAN_ANCHOR_PER_BATCH = 1
DEFAULT_SEED = 20260413
DEFAULT_SHORT_DEMO_TRIAL_COUNT = 10
DEFAULT_ALLOWED_PASSWORDS = ("marcussen", "hans", "timbre")
CONSENT_INFO_CONTENT = (
    "<p>I hereby declare that I have been clearly informed about the research "
    "Perceptual Similarity of Organ Timbre: Linking Listener Judgments to Computational Metrics "
    "at the University of Amsterdam, Music Cognition Group, conducted by Noah Jaffe under "
    "supervision of Ashley Burgoyne at the University of Amsterdam and Hans Fidom at the Vrij "
    "Universiteit Amsterdam as described in the information brochure. My questions have been "
    "answered to my satisfaction.</p>"
    "<p>I realise that participation in this research is on an entirely voluntary basis. I retain "
    "the right to revoke this consent without having to provide any reasons for my decision. I am "
    "aware that I am entitled to discontinue the research at any time, and that I can always "
    "withdraw my consent after the research has ended. If I decide to stop or withdraw my consent, "
    "all the information gathered up until then will be permanently deleted.</p>"
    "<p>If my research results are used in scientific publications or made public in any other way, "
    "they will be fully anonymised. My personal information may not be viewed by third parties.</p>"
    "<p>If I need any further information on the research, now or in the future, I can contact "
    "Noah Jaffe (phone number: +31 06 29 38 89 41; e-mail: n.jaffe@uva.nl; Spuistraat 134, "
    "1012 VB Amsterdam, The Netherlands.</p>"
    "<p>If I have any complaints regarding this research, I can contact the secretary of the Ethics "
    "Committee of the Faculty of Humanities of the University of Amsterdam: commissie-ethiek-fgw@uva.nl; Binnengasthuisstraat 9, 1012 ZA Amsterdam, The Netherlands.</p>"
    "<p><strong>By clicking next, I consent to participate in this research</strong></p>"
)
SPECIAL_BATCH_ZERO_INDEX = -1
SPECIAL_BATCH_ZERO_PITCH_ORDER = ("C", "c0", "c1", "c2", "c3")
SPECIAL_BATCH_ZERO_MAIN_FAMILY = "Principals"
SPECIAL_BATCH_ZERO_DIVISION = "Upper Division"
SPECIAL_BATCH_ZERO_REGISTRATION = "P8"
SPECIAL_BATCH_ZERO_MIC_LOCATION = "Close"
SPECIAL_BATCH_ZERO_ANCHOR_FAMILY = "Strings"
SPECIAL_BATCH_ZERO_ANCHOR_REGISTRATION = "VdG8"

PAIR_ROLE_CROSS_ORGAN_MAIN = "cross_organ_main"
PAIR_ROLE_SAME_PIPE_REFERENCE = "same_pipe_reference"
PAIR_ROLE_SAME_ORGAN_ANCHOR = "same_organ_anchor"


def _parse_bool(value: str) -> bool:
    return str(value or "").strip().lower() == "true"


def _infer_pair_role(row: dict[str, str]) -> str:
    explicit = str(row.get("pair_role", "")).strip()
    if explicit:
        return explicit
    if _parse_bool(row.get("same_organ_anchor", "")) or row.get("batch", "") == "same_organ_anchor":
        return PAIR_ROLE_SAME_ORGAN_ANCHOR
    if _parse_bool(row.get("same_pipe_reference", "")) or _parse_bool(row.get("same_organ_pair", "")):
        return PAIR_ROLE_SAME_PIPE_REFERENCE
    return PAIR_ROLE_CROSS_ORGAN_MAIN


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
    family_a: str
    family_b: str
    division_a: str
    division_b: str
    registration_raw_a: str
    registration_raw_b: str
    pitch_a: str
    pitch_b: str
    mic_location_a: str
    mic_location_b: str
    foot_length_a: str
    foot_length_b: str
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
    same_organ: bool
    same_pipe_reference: bool
    same_organ_anchor: bool
    pair_role: str
    processing_chain: str
    group_id: str
    pair_group_id: str

    @property
    def trial_group_key(self) -> str:
        return self.pair_group_id or self.group_id or "|".join(
            (
                self.pair_role,
                self.family_a,
                self.family_b,
                self.division_a,
                self.division_b,
                self.registration_raw_a,
                self.registration_raw_b,
                self.pitch_a,
                self.pitch_b,
                self.mic_location_a,
                self.mic_location_b,
                self.organ_a,
                self.organ_b,
            )
        )


@dataclass(frozen=True, slots=True)
class PageAssignment:
    """A batch-local page with a deterministic left/right item order."""

    batch_index: int
    page_kind: str
    page_index: int
    pair_row: PairRow
    first_member: str
    first_path: str
    second_path: str
    first_source_path: str
    second_source_path: str
    first_organ: str
    second_organ: str
    store_results: bool

    @property
    def page_name(self) -> str:
        prefix = "Trial" if self.store_results else "Example"
        return f"{prefix} {self.page_index:03d}"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate webMUSHRA paired-distance YAML batches from Marcussen pairs.csv",
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
        "--same-pipe-per-batch",
        type=int,
        default=DEFAULT_SAME_PIPE_PER_BATCH,
        help=f"Scored same-pipe hidden references per batch (default: {DEFAULT_SAME_PIPE_PER_BATCH})",
    )
    parser.add_argument(
        "--same-organ-anchor-per-batch",
        type=int,
        default=DEFAULT_SAME_ORGAN_ANCHOR_PER_BATCH,
        help=f"Scored same-organ hidden anchors per batch (default: {DEFAULT_SAME_ORGAN_ANCHOR_PER_BATCH})",
    )
    parser.add_argument(
        "--demo-same-pipe-per-batch",
        type=int,
        default=DEFAULT_DEMO_SAME_PIPE_PER_BATCH,
        help=f"Non-saved same-pipe examples per batch (default: {DEFAULT_DEMO_SAME_PIPE_PER_BATCH})",
    )
    parser.add_argument(
        "--demo-same-organ-anchor-per-batch",
        type=int,
        default=DEFAULT_DEMO_SAME_ORGAN_ANCHOR_PER_BATCH,
        help=(
            "Non-saved same-organ anchor examples per batch "
            f"(default: {DEFAULT_DEMO_SAME_ORGAN_ANCHOR_PER_BATCH})"
        ),
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
                family=row.get("family", ""),
                division=row.get("division", ""),
                registration_raw=row.get("registration_raw", ""),
                pitch=row.get("pitch", ""),
                mic_location=row.get("mic_location", ""),
                family_a=row.get("family_a", row.get("family", "")),
                family_b=row.get("family_b", row.get("family", "")),
                division_a=row.get("division_a", row.get("division", "")),
                division_b=row.get("division_b", row.get("division", "")),
                registration_raw_a=row.get("registration_raw_a", row.get("registration_raw", "")),
                registration_raw_b=row.get("registration_raw_b", row.get("registration_raw", "")),
                pitch_a=row.get("pitch_a", row.get("pitch", "")),
                pitch_b=row.get("pitch_b", row.get("pitch", "")),
                mic_location_a=row.get("mic_location_a", row.get("mic_location", "")),
                mic_location_b=row.get("mic_location_b", row.get("mic_location", "")),
                foot_length_a=row.get("foot_length_a", ""),
                foot_length_b=row.get("foot_length_b", ""),
                organ_a=row.get("organ_a", ""),
                organ_b=row.get("organ_b", ""),
                source_path_a=row.get("source_path_a", ""),
                source_path_b=row.get("source_path_b", ""),
                toot_wav_path_a=row.get("toot_wav_path_a", ""),
                toot_wav_path_b=row.get("toot_wav_path_b", ""),
                batch=row.get("batch", ""),
                source_a_toot_index=row.get("source_a_toot_index", ""),
                source_b_toot_index=row.get("source_b_toot_index", ""),
                same_organ_pair=_parse_bool(row.get("same_organ_pair", "")),
                same_organ=_parse_bool(row.get("same_organ", row.get("same_organ_pair", ""))),
                same_pipe_reference=_parse_bool(row.get("same_pipe_reference", row.get("same_organ_pair", ""))),
                same_organ_anchor=_parse_bool(row.get("same_organ_anchor", "")),
                pair_role=_infer_pair_role(row),
                processing_chain=row.get("processing_chain", ""),
                group_id=row.get("group_id", ""),
                pair_group_id=row.get("pair_group_id", row.get("group_id", "")),
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


def _fixed_targets(
    *,
    available_items: int,
    batch_count: int,
    per_batch: int,
    label: str,
    allow_reduce: bool = False,
) -> list[int]:
    if per_batch < 0:
        raise ValueError(f"{label} per-batch count must be >= 0")
    required = batch_count * per_batch
    if available_items < required:
        if allow_reduce:
            reduced_per_batch = available_items // batch_count
            if per_batch > 0 and reduced_per_batch < per_batch:
                print(
                    f"Warning: reducing {label} from {per_batch} to {reduced_per_batch} per batch "
                    f"because only {available_items} row(s) are available for {batch_count} batches."
                )
            return [reduced_per_batch] * batch_count
        raise ValueError(
            f"Not enough {label} rows for {batch_count} batches: need {required}, found {available_items}"
        )
    return [per_batch] * batch_count


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


def _remaining_rows(rows: list[PairRow], *, used_pair_ids: set[int]) -> list[PairRow]:
    return [row for row in rows if row.pair_id not in used_pair_ids]


def _prepare_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale_yaml in output_dir.glob("marcussen_batch_*.yaml"):
        stale_yaml.unlink()
    for manifest_name in ("batch_manifest.csv", "trial_manifest.csv"):
        manifest_path = output_dir / manifest_name
        if manifest_path.exists():
            manifest_path.unlink()


def _webmushra_root_for_output_dir(output_dir: Path) -> Path:
    if output_dir.name == "configs":
        return output_dir.parent
    if output_dir.parent.name == "configs":
        return output_dir.parent.parent
    return output_dir.parent.parent


def _batch_number_from_index(batch_index: int) -> int:
    if batch_index == SPECIAL_BATCH_ZERO_INDEX:
        return 0
    return batch_index + 1


def _yaml_scalar(value: object) -> str:
    return json.dumps(value, ensure_ascii=False)


def _batch_asset_relpath(asset_prefix: str, batch_number: int, assignment: PageAssignment, role: str) -> str:
    return str(
        Path(asset_prefix)
        / f"batch_{batch_number:02d}"
        / f"{assignment.page_kind}_{assignment.page_index:03d}_pair_{_pair_file_id(assignment.pair_row)}_{role}.wav"
    )


def _copy_batch_audio_assets(
    pages_by_batch: list[list[PageAssignment]],
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
    for assignments in pages_by_batch:
        if not assignments:
            continue
        batch_number = _batch_number_from_index(assignments[0].batch_index)
        for assignment in assignments:
            copies = [
                (
                    source_root / assignment.first_path,
                    webmushra_root / _batch_asset_relpath(asset_prefix, batch_number, assignment, "item_1"),
                ),
                (
                    source_root / assignment.second_path,
                    webmushra_root / _batch_asset_relpath(asset_prefix, batch_number, assignment, "item_2"),
                ),
            ]
            for source_path, destination_path in copies:
                destination_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_path, destination_path)
                copied_files += 1

    return copied_files


def _ordered_value(assignment: PageAssignment, a_value: str, b_value: str) -> tuple[str, str]:
    if assignment.first_member == "a":
        return a_value, b_value
    return b_value, a_value


def _append_shared_or_ordered(details: list[str], label: str, first_value: str, second_value: str) -> None:
    if first_value == second_value:
        if first_value:
            details.append(f"{label}: {first_value}")
        return
    if first_value:
        details.append(f"{label} item 1: {first_value}")
    if second_value:
        details.append(f"{label} item 2: {second_value}")


def _trial_content(assignment: PageAssignment) -> str:
    row = assignment.pair_row
    family_1, family_2 = _ordered_value(assignment, row.family_a, row.family_b)
    division_1, division_2 = _ordered_value(assignment, row.division_a, row.division_b)
    registration_1, registration_2 = _ordered_value(assignment, row.registration_raw_a, row.registration_raw_b)
    pitch_1, pitch_2 = _ordered_value(assignment, row.pitch_a, row.pitch_b)
    mic_1, mic_2 = _ordered_value(assignment, row.mic_location_a, row.mic_location_b)

    details: list[str] = []
    _append_shared_or_ordered(details, "Family", family_1, family_2)
    _append_shared_or_ordered(details, "Division", division_1, division_2)
    _append_shared_or_ordered(details, "Registration", registration_1, registration_2)
    _append_shared_or_ordered(details, "Pitch", pitch_1, pitch_2)
    _append_shared_or_ordered(details, "Microphone", mic_1, mic_2)
    return "<br/>".join(details)


def _page_id(assignment: PageAssignment) -> str:
    if assignment.store_results:
        return _pair_file_id(assignment.pair_row)
    return f"{assignment.page_kind}_{assignment.page_index:03d}_pair_{assignment.pair_row.pair_id}"


def _page_content(assignment: PageAssignment) -> str:
    if assignment.store_results:
        return ""

    if assignment.pair_row.pair_role == PAIR_ROLE_SAME_PIPE_REFERENCE:
        return (
            "Practice example: these two sounds come from the same pipe, "
            "so a rating near 1 is expected."
        )
    if assignment.pair_row.pair_role == PAIR_ROLE_SAME_ORGAN_ANCHOR:
        return (
            "Practice example: these two sounds come from the same organ but contrasting stops, "
            "so a high distance rating is expected."
        )
    return "Practice example."


def _append_paired_distance_page(
    lines: list[str],
    assignment: PageAssignment,
    batch_number: int,
    asset_prefix: str,
    *,
    indent: str = "  ",
) -> None:
    lines.extend(
        [
            f"{indent}- type: paired_distance",
            f"{indent}  id: {_yaml_scalar(_page_id(assignment))}",
            f"{indent}  name: {_yaml_scalar(assignment.page_name)}",
            f"{indent}  content: {_yaml_scalar(_page_content(assignment))}",
        ]
    )
    if not assignment.store_results:
        lines.append(f"{indent}  storeResults: false")
    lines.extend(
        [
            f"{indent}  stimuli:",
            f"{indent}    1: {_yaml_scalar(_batch_asset_relpath(asset_prefix, batch_number, assignment, 'item_1'))}",
            f"{indent}    2: {_yaml_scalar(_batch_asset_relpath(asset_prefix, batch_number, assignment, 'item_2'))}",
        ]
    )


def _intro_content(batch_number: int, *, trial_count: int, demo_count: int) -> str:
    if batch_number == 0:
        return (
            "Marcussen listening test batch 00.<br/><br/>"
            "Each trial presents two audio files, item 1 and item 2. "
            "Rate how different they sound on a scale from 1 to 7, where 1 means most similar and 7 means incredibly different.<br/><br/>"
            "Keyboard shortcuts: <strong>Q</strong> plays item 1, <strong>W</strong> plays item 2, "
            "<strong>E</strong> plays item 1 then item 2, <strong>Space</strong> pauses playback, "
            "<strong>1-7</strong> selects the distance rating, and <strong>Cmd/Ctrl+Enter</strong> advances while the comment box is focused.<br/><br/>"
            f"This batch contains {trial_count} scored trials: all available "
            "P8 Upper Division Close cross-organ pairs across the selected pitches, "
            "plus one same-pipe hidden reference and one same-pitch different-timbre hidden anchor."
        )
    return (
        f"Marcussen listening test batch {batch_number:02d}.<br/><br/>"
        "Each trial presents two audio files, item 1 and item 2. "
        "Rate how different they sound on a scale from 1 to 7, where 1 means most similar and 7 means incredibly different.<br/><br/>"
        "Keyboard shortcuts: <strong>Q</strong> plays item 1, <strong>W</strong> plays item 2, "
        "<strong>E</strong> plays item 1 then item 2, <strong>Space</strong> pauses playback, "
        "<strong>1-7</strong> selects the distance rating, and <strong>Cmd/Ctrl+Enter</strong> advances while the comment box is focused.<br/><br/>"
        f"This batch begins with {demo_count} practice example(s) that are not saved, followed by {trial_count} scored trials."
    )


def _make_assignment(
    row: PairRow,
    *,
    batch_index: int,
    page_kind: str,
    page_index: int,
    store_results: bool,
    rng: random.Random,
) -> PageAssignment:
    first_member = "a" if rng.random() < 0.5 else "b"
    if first_member == "a":
        first_path = row.toot_wav_path_a
        second_path = row.toot_wav_path_b
        first_source = row.source_path_a
        second_source = row.source_path_b
        first_organ = row.organ_a
        second_organ = row.organ_b
    else:
        first_path = row.toot_wav_path_b
        second_path = row.toot_wav_path_a
        first_source = row.source_path_b
        second_source = row.source_path_a
        first_organ = row.organ_b
        second_organ = row.organ_a

    return PageAssignment(
        batch_index=batch_index,
        page_kind=page_kind,
        page_index=page_index,
        pair_row=row,
        first_member=first_member,
        first_path=first_path,
        second_path=second_path,
        first_source_path=first_source,
        second_source_path=second_source,
        first_organ=first_organ,
        second_organ=second_organ,
        store_results=store_results,
    )


def _assign_batch_pages(
    *,
    demo_rows: list[PairRow],
    trial_rows: list[PairRow],
    batch_index: int,
    rng: random.Random,
) -> list[PageAssignment]:
    pages: list[PageAssignment] = []

    for demo_index, row in enumerate(demo_rows, start=1):
        pages.append(
            _make_assignment(
                row,
                batch_index=batch_index,
                page_kind="example",
                page_index=demo_index,
                store_results=False,
                rng=rng,
            )
        )

    shuffled_trials = list(trial_rows)
    rng.shuffle(shuffled_trials)
    for trial_index, row in enumerate(shuffled_trials, start=1):
        pages.append(
            _make_assignment(
                row,
                batch_index=batch_index,
                page_kind="trial",
                page_index=trial_index,
                store_results=True,
                rng=rng,
            )
        )

    return pages


def _pair_file_id(row: PairRow) -> str:
    if row.pair_role == PAIR_ROLE_SAME_ORGAN_ANCHOR and row.pair_id == 0:
        return "anchor_001"
    return str(row.pair_id)


def _is_batch_zero_main_row(row: PairRow) -> bool:
    return (
        row.pair_role == PAIR_ROLE_CROSS_ORGAN_MAIN
        and row.family == SPECIAL_BATCH_ZERO_MAIN_FAMILY
        and row.division == SPECIAL_BATCH_ZERO_DIVISION
        and row.registration_raw == SPECIAL_BATCH_ZERO_REGISTRATION
        and row.mic_location == SPECIAL_BATCH_ZERO_MIC_LOCATION
        and row.pitch in SPECIAL_BATCH_ZERO_PITCH_ORDER
    )


def _batch_zero_sort_key(row: PairRow) -> tuple[int, str, str, int]:
    try:
        pitch_index = SPECIAL_BATCH_ZERO_PITCH_ORDER.index(row.pitch)
    except ValueError:
        pitch_index = len(SPECIAL_BATCH_ZERO_PITCH_ORDER)
    return (pitch_index, row.organ_a, row.organ_b, row.pair_id)


def _select_batch_zero_same_pipe(rows: list[PairRow]) -> PairRow:
    candidates = [
        row
        for row in rows
        if row.pair_role == PAIR_ROLE_SAME_PIPE_REFERENCE
        and row.family == SPECIAL_BATCH_ZERO_MAIN_FAMILY
        and row.division == SPECIAL_BATCH_ZERO_DIVISION
        and row.registration_raw == SPECIAL_BATCH_ZERO_REGISTRATION
        and row.mic_location == SPECIAL_BATCH_ZERO_MIC_LOCATION
        and row.pitch in SPECIAL_BATCH_ZERO_PITCH_ORDER
    ]
    if not candidates:
        raise ValueError("Could not find a P8 Upper Division Close same-pipe reference for batch 00")
    return sorted(candidates, key=_batch_zero_sort_key)[0]


def _select_batch_zero_anchor_source(rows: list[PairRow], *, reference: PairRow) -> PairRow:
    candidates = [
        row
        for row in rows
        if row.family == SPECIAL_BATCH_ZERO_ANCHOR_FAMILY
        and row.division == SPECIAL_BATCH_ZERO_DIVISION
        and row.registration_raw == SPECIAL_BATCH_ZERO_ANCHOR_REGISTRATION
        and row.pitch == reference.pitch
        and row.mic_location == SPECIAL_BATCH_ZERO_MIC_LOCATION
        and row.organ_a == reference.organ_a
        and row.organ_b == reference.organ_a
        and row.pair_role == PAIR_ROLE_SAME_PIPE_REFERENCE
    ]
    if not candidates:
        raise ValueError(
            "Could not find a same-organ VdG8 Upper Division Close anchor source for batch 00"
        )
    return sorted(candidates, key=lambda row: (row.organ_a, row.pair_id))[0]


def _make_batch_zero_anchor(reference: PairRow, anchor_source: PairRow) -> PairRow:
    return PairRow(
        row_index=0,
        pair_id=0,
        family=reference.family,
        division=reference.division,
        registration_raw=reference.registration_raw,
        pitch=reference.pitch,
        mic_location=reference.mic_location,
        family_a=reference.family,
        family_b=anchor_source.family,
        division_a=reference.division,
        division_b=anchor_source.division,
        registration_raw_a=reference.registration_raw,
        registration_raw_b=anchor_source.registration_raw,
        pitch_a=reference.pitch,
        pitch_b=anchor_source.pitch,
        mic_location_a=reference.mic_location,
        mic_location_b=anchor_source.mic_location,
        foot_length_a=reference.foot_length_a,
        foot_length_b=anchor_source.foot_length_a,
        organ_a=reference.organ_a,
        organ_b=anchor_source.organ_a,
        source_path_a=reference.source_path_a,
        source_path_b=anchor_source.source_path_a,
        toot_wav_path_a=reference.toot_wav_path_a,
        toot_wav_path_b=anchor_source.toot_wav_path_a,
        batch="batch_00_anchor",
        source_a_toot_index=reference.source_a_toot_index,
        source_b_toot_index=anchor_source.source_a_toot_index,
        same_organ_pair=False,
        same_organ=True,
        same_pipe_reference=False,
        same_organ_anchor=True,
        pair_role=PAIR_ROLE_SAME_ORGAN_ANCHOR,
        processing_chain=reference.processing_chain or anchor_source.processing_chain,
        group_id=reference.group_id,
        pair_group_id="batch_00_same_pitch_different_timbre_anchor",
    )


def _build_batch_zero_pages(rows: list[PairRow]) -> list[PageAssignment]:
    main_rows = sorted([row for row in rows if _is_batch_zero_main_row(row)], key=_batch_zero_sort_key)
    if not main_rows:
        return []

    same_pipe_reference = _select_batch_zero_same_pipe(rows)
    anchor = _make_batch_zero_anchor(
        same_pipe_reference,
        _select_batch_zero_anchor_source(rows, reference=same_pipe_reference),
    )
    trial_rows = [*main_rows, same_pipe_reference, anchor]

    pages: list[PageAssignment] = []
    for trial_index, row in enumerate(trial_rows, start=1):
        pages.append(
            _make_assignment(
                row,
                batch_index=SPECIAL_BATCH_ZERO_INDEX,
                page_kind="trial",
                page_index=trial_index,
                store_results=True,
                rng=random.Random(1),
            )
        )
    return pages


def _render_batch_yaml(
    pages: list[PageAssignment],
    batch_index: int,
    asset_prefix: str,
    *,
    test_name: str | None = None,
    test_id: str | None = None,
) -> str:
    batch_number = _batch_number_from_index(batch_index)
    batch_id = test_id or f"marcussen_batch_{batch_number:02d}"
    batch_name = test_name or f"Marcussen AB Batch {batch_number:02d}"
    if not pages:
        raise ValueError(f"Batch {batch_number:02d} has no pages")

    scored_pages = [page for page in pages if page.store_results]
    demo_pages = [page for page in pages if not page.store_results]
    volume_stimulus = _batch_asset_relpath(asset_prefix, batch_number, pages[0], "item_1")
    lines = [
        f"testname: {_yaml_scalar(batch_name)}",
        f"testId: {_yaml_scalar(batch_id)}",
        "bufferSize: 2048",
        "stopOnErrors: true",
        "showButtonPreviousPage: true",
        "language: \"en\"",
        "remoteService: \"service/write.php\"",
        "allowedPasswords:",
        *[f"  - {_yaml_scalar(password)}" for password in DEFAULT_ALLOWED_PASSWORDS],
        "",
        "pages:",
        "  - type: generic",
        "    id: \"consent_info\"",
        "    name: \"Consent\"",
        f"    content: {_yaml_scalar(CONSENT_INFO_CONTENT)}",
        "  - type: generic",
        "    id: \"intro\"",
        "    name: \"Instructions\"",
        f"    content: {_yaml_scalar(_intro_content(batch_number, trial_count=len(scored_pages), demo_count=len(demo_pages)))}",
        "  - type: volume",
        "    id: \"volume\"",
        "    name: \"Volume\"",
        "    content: \"Adjust to a comfortable listening level before starting the examples and trials.\"",
        f"    stimulus: {_yaml_scalar(volume_stimulus)}",
        "    defaultVolume: 0.5",
    ]

    for assignment in demo_pages:
        _append_paired_distance_page(lines, assignment, batch_number, asset_prefix)

    if scored_pages:
        lines.extend(["  -", "    - random"])
        for assignment in scored_pages:
            _append_paired_distance_page(lines, assignment, batch_number, asset_prefix, indent="    ")

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


def _write_batch_manifest(output_dir: Path, pages_by_batch: list[list[PageAssignment]]) -> Path:
    manifest_path = output_dir / "batch_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "batch_number",
                "yaml_file",
                "total_page_count",
                "demo_page_count",
                "trial_count",
                "main_pair_count",
                "same_pipe_reference_count",
                "same_organ_anchor_count",
                "demo_same_pipe_count",
                "demo_same_organ_anchor_count",
                "scored_pair_ids",
            ],
        )
        writer.writeheader()
        for pages in pages_by_batch:
            if not pages:
                continue
            batch_number = _batch_number_from_index(pages[0].batch_index)
            scored_pages = [page for page in pages if page.store_results]
            demo_pages = [page for page in pages if not page.store_results]
            writer.writerow(
                {
                    "batch_number": batch_number,
                    "yaml_file": f"marcussen_batch_{batch_number:02d}.yaml",
                    "total_page_count": len(pages),
                    "demo_page_count": len(demo_pages),
                    "trial_count": len(scored_pages),
                    "main_pair_count": sum(page.pair_row.pair_role == PAIR_ROLE_CROSS_ORGAN_MAIN for page in scored_pages),
                    "same_pipe_reference_count": sum(
                        page.pair_row.pair_role == PAIR_ROLE_SAME_PIPE_REFERENCE for page in scored_pages
                    ),
                    "same_organ_anchor_count": sum(
                        page.pair_row.pair_role == PAIR_ROLE_SAME_ORGAN_ANCHOR for page in scored_pages
                    ),
                    "demo_same_pipe_count": sum(
                        page.pair_row.pair_role == PAIR_ROLE_SAME_PIPE_REFERENCE for page in demo_pages
                    ),
                    "demo_same_organ_anchor_count": sum(
                        page.pair_row.pair_role == PAIR_ROLE_SAME_ORGAN_ANCHOR for page in demo_pages
                    ),
                    "scored_pair_ids": ";".join(str(page.pair_row.pair_id) for page in scored_pages),
                }
            )
    return manifest_path


def _write_trial_manifest(output_dir: Path, pages_by_batch: list[list[PageAssignment]]) -> Path:
    manifest_path = output_dir / "trial_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "batch_number",
                "yaml_file",
                "page_kind",
                "page_index",
                "page_name",
                "store_results",
                "page_id",
                "pair_id",
                "pair_role",
            ],
        )
        writer.writeheader()

        for pages in pages_by_batch:
            if not pages:
                continue
            batch_number = _batch_number_from_index(pages[0].batch_index)
            yaml_file = f"marcussen_batch_{batch_number:02d}.yaml"
            for assignment in pages:
                writer.writerow(
                    {
                        "batch_number": batch_number,
                        "yaml_file": yaml_file,
                        "page_kind": assignment.page_kind,
                        "page_index": assignment.page_index,
                        "page_name": assignment.page_name,
                        "store_results": "True" if assignment.store_results else "False",
                        "page_id": _page_id(assignment),
                        "pair_id": assignment.pair_row.pair_id,
                        "pair_role": assignment.pair_row.pair_role,
                    }
                )
    return manifest_path


def _short_demo_pages(pages: list[PageAssignment], *, scored_trial_count: int) -> list[PageAssignment]:
    short_pages: list[PageAssignment] = []
    scored_kept = 0
    for page in pages:
        if not page.store_results:
            short_pages.append(page)
            continue
        if scored_kept >= scored_trial_count:
            continue
        short_pages.append(page)
        scored_kept += 1
    return short_pages


def main() -> int:
    args = _parse_args()
    if args.batch_count <= 0:
        raise ValueError("--batch-count must be > 0")

    rng = random.Random(args.seed)
    rows = _load_pairs(args.pairs_csv)

    main_rows = [row for row in rows if row.pair_role == PAIR_ROLE_CROSS_ORGAN_MAIN]
    same_pipe_rows = [row for row in rows if row.pair_role == PAIR_ROLE_SAME_PIPE_REFERENCE]
    same_organ_anchor_rows = [row for row in rows if row.pair_role == PAIR_ROLE_SAME_ORGAN_ANCHOR]
    batch_00_pages = _build_batch_zero_pages(rows)

    main_targets = _batch_targets(len(main_rows), args.batch_count, rng)
    same_pipe_demo_targets = _fixed_targets(
        available_items=len(same_pipe_rows),
        batch_count=args.batch_count,
        per_batch=args.demo_same_pipe_per_batch,
        label="same-pipe demo",
        allow_reduce=True,
    )
    same_organ_demo_targets = _fixed_targets(
        available_items=len(same_organ_anchor_rows),
        batch_count=args.batch_count,
        per_batch=args.demo_same_organ_anchor_per_batch,
        label="same-organ demo anchor",
        allow_reduce=True,
    )

    same_pipe_demo_batches = _distribute_rows(
        same_pipe_rows,
        same_pipe_demo_targets,
        rng,
        group_key=lambda row: row.trial_group_key,
    )
    used_same_pipe_ids = {
        row.pair_id
        for batch in same_pipe_demo_batches
        for row in batch
    }
    remaining_same_pipe_rows = _remaining_rows(same_pipe_rows, used_pair_ids=used_same_pipe_ids)
    same_pipe_trial_targets = _fixed_targets(
        available_items=len(remaining_same_pipe_rows),
        batch_count=args.batch_count,
        per_batch=args.same_pipe_per_batch,
        label="same-pipe hidden reference",
        allow_reduce=True,
    )
    same_pipe_trial_batches = _distribute_rows(
        remaining_same_pipe_rows,
        same_pipe_trial_targets,
        rng,
        group_key=lambda row: row.trial_group_key,
    )

    same_organ_demo_batches = _distribute_rows(
        same_organ_anchor_rows,
        same_organ_demo_targets,
        rng,
        group_key=lambda row: row.trial_group_key,
    )
    used_same_organ_anchor_ids = {
        row.pair_id
        for batch in same_organ_demo_batches
        for row in batch
    }
    remaining_same_organ_anchor_rows = _remaining_rows(
        same_organ_anchor_rows,
        used_pair_ids=used_same_organ_anchor_ids,
    )
    same_organ_anchor_targets = _fixed_targets(
        available_items=len(remaining_same_organ_anchor_rows),
        batch_count=args.batch_count,
        per_batch=args.same_organ_anchor_per_batch,
        label="same-organ hidden anchor",
        allow_reduce=True,
    )
    same_organ_anchor_batches = _distribute_rows(
        remaining_same_organ_anchor_rows,
        same_organ_anchor_targets,
        rng,
        group_key=lambda row: row.trial_group_key,
    )

    main_batches = _distribute_rows(main_rows, main_targets, rng, group_key=lambda row: row.trial_group_key)

    _prepare_output_dir(args.output_dir)

    pages_by_batch: list[list[PageAssignment]] = []
    for batch_index in range(args.batch_count):
        demo_rows = [*same_pipe_demo_batches[batch_index], *same_organ_demo_batches[batch_index]]
        trial_rows = [
            *main_batches[batch_index],
            *same_pipe_trial_batches[batch_index],
            *same_organ_anchor_batches[batch_index],
        ]
        pages_by_batch.append(
            _assign_batch_pages(
                demo_rows=demo_rows,
                trial_rows=trial_rows,
                batch_index=batch_index,
                rng=rng,
            )
        )

    all_pages_by_batch = [*([batch_00_pages] if batch_00_pages else []), *pages_by_batch]
    webmushra_root = _webmushra_root_for_output_dir(args.output_dir)
    copied_file_count = _copy_batch_audio_assets(
        all_pages_by_batch,
        pairs_csv_path=args.pairs_csv,
        webmushra_root=webmushra_root,
        asset_prefix=args.asset_prefix,
    )

    batch_00_path: Path | None = None
    if batch_00_pages:
        batch_00_yaml = _render_batch_yaml(batch_00_pages, SPECIAL_BATCH_ZERO_INDEX, args.asset_prefix)
        batch_00_path = args.output_dir / "marcussen_batch_00.yaml"
        batch_00_path.write_text(batch_00_yaml, encoding="utf-8")

    for batch_index, pages in enumerate(pages_by_batch):
        yaml_text = _render_batch_yaml(pages, batch_index, args.asset_prefix)
        yaml_path = args.output_dir / f"marcussen_batch_{batch_index + 1:02d}.yaml"
        yaml_path.write_text(yaml_text, encoding="utf-8")

    batch_01_short_pages = _short_demo_pages(
        pages_by_batch[0],
        scored_trial_count=DEFAULT_SHORT_DEMO_TRIAL_COUNT,
    )
    batch_01_short_yaml = _render_batch_yaml(
        batch_01_short_pages,
        0,
        args.asset_prefix,
        test_name=(
            f"Marcussen AB Batch 01 (Short Demo - "
            f"{sum(page.store_results for page in batch_01_short_pages)} scored trials)"
        ),
        test_id="marcussen_batch_01_short",
    )
    batch_01_short_path = args.output_dir / "marcussen_batch_01_short.yaml"
    batch_01_short_path.write_text(batch_01_short_yaml, encoding="utf-8")

    batch_manifest = _write_batch_manifest(args.output_dir, all_pages_by_batch)
    trial_manifest = _write_trial_manifest(args.output_dir, all_pages_by_batch)

    main_count = sum(len(batch) for batch in main_batches)
    same_pipe_trial_count = sum(len(batch) for batch in same_pipe_trial_batches)
    same_pipe_demo_count = sum(len(batch) for batch in same_pipe_demo_batches)
    same_organ_anchor_count = sum(len(batch) for batch in same_organ_anchor_batches)
    same_organ_demo_count = sum(len(batch) for batch in same_organ_demo_batches)
    staged_asset_root = webmushra_root / Path(args.asset_prefix)

    if batch_00_path is not None:
        print(
            f"Generated special batch 00: {batch_00_path} "
            f"({sum(page.store_results for page in batch_00_pages)} scored trials)"
        )
    print(
        f"Generated {args.batch_count} webMUSHRA batch configs in {args.output_dir} "
        f"using seed={args.seed}."
    )
    print(
        f"Also generated short demo version: {batch_01_short_path} "
        f"({sum(page.store_results for page in batch_01_short_pages)} scored trials plus "
        f"{sum(not page.store_results for page in batch_01_short_pages)} examples)"
    )
    print(
        f"Main pairs allocated: {main_count} across {args.batch_count} batches "
        f"({min(main_targets)}-{max(main_targets)} per batch)."
    )
    print(
        f"Same-pipe hidden references allocated: {same_pipe_trial_count} total "
        f"({args.same_pipe_per_batch} per batch) plus {same_pipe_demo_count} demo examples."
    )
    print(
        f"Same-organ hidden anchors allocated: {same_organ_anchor_count} total "
        f"({args.same_organ_anchor_per_batch} per batch) plus {same_organ_demo_count} demo examples."
    )
    print(f"Copied {copied_file_count} WAV files into {staged_asset_root}")
    print(f"Batch manifest: {batch_manifest}")
    print(f"Trial manifest: {trial_manifest}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
