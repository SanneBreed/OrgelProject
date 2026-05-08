"""Build listening-test outputs from grouped dataset items.

This module prepares processed audio files, organizes them into listening
dataset batches, and writes the pair CSV that points to those exported clips.
It sits after parsing/grouping and uses the shared pairing rules to define
which prepared items appear together in the output dataset.

The listening-dataset workflow is designed around the fact that each source
recording contains three organ "toots". One goal is to split each original file
into those three toot-level excerpts so they can be exported as separate batch
items. Another goal is to include control comparisons within the listening
dataset, such as pairing two different toots from the same original recording
in addition to cross-organ pairs.
"""

from __future__ import annotations

from dataclasses import dataclass
import csv
import logging
from pathlib import Path
import re
import shutil
from statistics import median
from typing import Any

from tqdm.auto import tqdm

from .audio_prep import estimate_wrapped_pitch_cents_offset, prepare_source_audio
from .constants import ALLOWED_ORGAN_IDS
from .dataset import MarcussenDataset, make_group_id
from .pairs import is_cross_organ_pair
from .parsing import ParsedItem

logger = logging.getLogger(__name__)

LISTENING_DATASET_TOOT_INDICES = (1, 2)
CROSS_ORGAN_TOOT_INDEX = 1
SAME_ORGAN_CONTROL_TOOT_PAIR = (1, 2)
PAIR_ROLE_CROSS_ORGAN_MAIN = "cross_organ_main"
PAIR_ROLE_SAME_PIPE_REFERENCE = "same_pipe_reference"
PAIR_ROLE_SAME_ORGAN_ANCHOR = "same_organ_anchor"
TUNING_SAMPLE_COUNT = 5
TUNING_REPORT_NAME = "organ_tuning_offsets.csv"
TUNING_REFERENCE_REGISTRATION = "P8"
TUNING_REFERENCE_DIVISION = "main_division"
TUNING_REFERENCE_MIC_LOCATION = "close"
TUNING_REFERENCE_PITCHES = ("C", "c0", "c1", "c2", "c3")
_FOOT_LENGTH_RE = re.compile(r"(\d+)")


@dataclass(slots=True, frozen=True)
class ListeningDatasetItem:
    """Prepared listening-dataset clip plus source metadata."""

    source_item: ParsedItem
    source_path_rel: str
    toot_wav_path_rel: str
    batch: str
    toot_index: int | None
    processing_chain: str

    @property
    def organ_id(self) -> str | None:
        value = self.source_item.meta.get("organ_id")
        return None if value in (None, "") else str(value)

    @property
    def meta(self) -> dict[str, Any]:
        return self.source_item.meta


@dataclass(slots=True, frozen=True)
class OrganTuningMeasurement:
    """One file-level tuning estimate used to derive an organ-wide offset."""

    pitch: str
    source_path_rel: str
    fundamental_hz: float
    wrapped_offset_cents: float


@dataclass(slots=True, frozen=True)
class OrganTuningProfile:
    """Per-organ tuning offset plus the sample measurements behind it."""

    organ_id: str
    global_offset_cents: float
    applied_shift_cents: float
    measurements: tuple[OrganTuningMeasurement, ...]


SAME_ORGAN_CONTROL_BATCH = "same_organ_control"
SAME_ORGAN_ANCHOR_BATCH = "same_organ_anchor"


def _path_relative_to_root(path_str: str, root: Path) -> str:
    path = Path(path_str)
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _prepare_source_items(
    item: ParsedItem,
    *,
    dataset_root: Path,
    wav_root: Path,
    organ_tuning_profiles: dict[str, OrganTuningProfile],
    expand_toots: bool,
    trim: bool,
    normalize: bool,
    steady_state_seconds: float | None,
) -> list[ListeningDatasetItem]:
    prepared_items: list[ListeningDatasetItem] = []
    rel_source_path = _path_relative_to_root(item.path, dataset_root)
    organ_id = item.meta.get("organ_id")
    organ_profile = None if organ_id in (None, "") else organ_tuning_profiles.get(str(organ_id))
    prepared_files = prepare_source_audio(
        item.path,
        output_root=wav_root.parent,
        relative_source_path=rel_source_path,
        expand_toots=expand_toots,
        toot_indices=LISTENING_DATASET_TOOT_INDICES if expand_toots else None,
        trim=trim,
        normalize=normalize,
        steady_state_seconds=steady_state_seconds,
        pitch_correct=bool(organ_tuning_profiles),
        pitch_shift_steps=0.0 if organ_profile is None else organ_profile.applied_shift_cents / 100.0,
    )
    for prepared in prepared_files:
        prepared_items.append(
            ListeningDatasetItem(
                source_item=item,
                source_path_rel=rel_source_path,
                toot_wav_path_rel=prepared.output_relpath,
                batch=prepared.batch,
                toot_index=prepared.toot_index,
                processing_chain=prepared.processing_chain,
            )
        )
    return prepared_items


def _target_batches(
    *,
    expand_toots: bool,
):
    if not expand_toots:
        return ["full"]
    return [
        f"toot_{CROSS_ORGAN_TOOT_INDEX}",
        SAME_ORGAN_CONTROL_BATCH,
        SAME_ORGAN_ANCHOR_BATCH,
    ]


def _all_batch_caps_reached(
    rows_written_by_batch: dict[str, int],
    *,
    target_batches: list[str],
    debug_first_n_per_batch: int | None,
) -> bool:
    if debug_first_n_per_batch is None:
        return False
    return all(rows_written_by_batch.get(batch, 0) >= debug_first_n_per_batch for batch in target_batches)


def _iter_incremental_pairs(
    existing_items: list[ListeningDatasetItem],
    new_item: ListeningDatasetItem,
    *,
    expand_toots: bool,
):
    for existing_item in existing_items:
        if expand_toots:
            if _is_cross_organ_listening_pair(existing_item, new_item):
                yield existing_item, new_item
            continue
        if is_cross_organ_pair(existing_item, new_item):
            yield existing_item, new_item


def _is_cross_organ_listening_pair(item_a: ListeningDatasetItem, item_b: ListeningDatasetItem) -> bool:
    return (
        item_a.toot_index == CROSS_ORGAN_TOOT_INDEX
        and item_b.toot_index == CROSS_ORGAN_TOOT_INDEX
        and is_cross_organ_pair(item_a, item_b)
    )


def _is_same_organ_control_pair(item_a: ListeningDatasetItem, item_b: ListeningDatasetItem) -> bool:
    return (
        item_a.organ_id not in (None, "")
        and item_a.organ_id == item_b.organ_id
        and {item_a.toot_index, item_b.toot_index} == set(SAME_ORGAN_CONTROL_TOOT_PAIR)
    )


def _extract_single_foot_length(registration_raw: Any) -> str:
    values = _FOOT_LENGTH_RE.findall(str(registration_raw or ""))
    if len(values) != 1:
        return ""
    return values[0]


def _pair_role_same_organ_value(pair_role: str) -> str:
    return "True" if pair_role in {PAIR_ROLE_SAME_PIPE_REFERENCE, PAIR_ROLE_SAME_ORGAN_ANCHOR} else "False"


def _pair_role_same_pipe_value(pair_role: str) -> str:
    return "True" if pair_role == PAIR_ROLE_SAME_PIPE_REFERENCE else "False"


def _pair_role_anchor_value(pair_role: str) -> str:
    return "True" if pair_role == PAIR_ROLE_SAME_ORGAN_ANCHOR else "False"


def _pair_group_id(
    item_a: ListeningDatasetItem,
    item_b: ListeningDatasetItem,
    *,
    pair_role: str,
    dataset: MarcussenDataset,
) -> str:
    if pair_role != PAIR_ROLE_SAME_ORGAN_ANCHOR:
        return make_group_id(_output_meta(item_a), dataset.group_keys)

    families = sorted(
        value
        for value in (_display_value("family", item_a.meta.get("family", "")), _display_value("family", item_b.meta.get("family", "")))
        if value not in (None, "")
    )
    registrations = sorted(
        value
        for value in (item_a.meta.get("registration_raw", ""), item_b.meta.get("registration_raw", ""))
        if value not in (None, "")
    )
    return "|".join(
        [
            f"pair_role={pair_role}",
            f"organ_id={item_a.organ_id or ''}",
            f"pitch={item_a.meta.get('pitch', '')}",
            f"division={_display_value('division', item_a.meta.get('division', ''))}",
            f"mic_location={_display_value('mic_location', item_a.meta.get('mic_location', ''))}",
            f"foot_length={_extract_single_foot_length(item_a.meta.get('registration_raw', ''))}",
            f"family_pair={'+'.join(families)}",
            f"registration_pair={'+'.join(registrations)}",
        ]
    )


def _same_organ_anchor_key(item: ListeningDatasetItem) -> tuple[str, str, str, str, str] | None:
    if item.toot_index != CROSS_ORGAN_TOOT_INDEX:
        return None
    organ_id = item.organ_id
    pitch = item.meta.get("pitch")
    division = item.meta.get("division")
    mic_location = item.meta.get("mic_location")
    foot_length = _extract_single_foot_length(item.meta.get("registration_raw", ""))
    if organ_id in (None, "") or pitch in (None, "") or division in (None, "") or mic_location in (None, "") or foot_length == "":
        return None
    return (
        str(organ_id),
        str(pitch),
        str(division),
        str(mic_location),
        foot_length,
    )


def _iter_same_organ_anchor_pairs(items: list[ListeningDatasetItem]):
    items_by_key: dict[tuple[str, str, str, str, str], list[ListeningDatasetItem]] = {}
    for item in items:
        key = _same_organ_anchor_key(item)
        if key is None:
            continue
        items_by_key.setdefault(key, []).append(item)

    for group_items in items_by_key.values():
        for index_a, item_a in enumerate(group_items):
            family_a = item_a.meta.get("family")
            if family_a in (None, ""):
                continue
            for item_b in group_items[index_a + 1 :]:
                family_b = item_b.meta.get("family")
                if family_b in (None, "") or family_a == family_b:
                    continue
                if item_a.source_path_rel == item_b.source_path_rel:
                    continue
                yield item_a, item_b


def _make_csv_row(
    item_a: ListeningDatasetItem,
    item_b: ListeningDatasetItem,
    *,
    pair_id: int,
    batch: str,
    pair_role: str,
    dataset: MarcussenDataset,
) -> dict[str, Any]:
    output_meta_a = _output_meta(item_a)
    output_meta_b = _output_meta(item_b)
    return {
        "pair_id": pair_id,
        "family": output_meta_a.get("family", ""),
        "division": output_meta_a.get("division", ""),
        "registration_raw": item_a.meta.get("registration_raw", ""),
        "pitch": item_a.meta.get("pitch", ""),
        "mic_location": output_meta_a.get("mic_location", ""),
        "family_a": output_meta_a.get("family", ""),
        "family_b": output_meta_b.get("family", ""),
        "division_a": output_meta_a.get("division", ""),
        "division_b": output_meta_b.get("division", ""),
        "registration_raw_a": item_a.meta.get("registration_raw", ""),
        "registration_raw_b": item_b.meta.get("registration_raw", ""),
        "pitch_a": item_a.meta.get("pitch", ""),
        "pitch_b": item_b.meta.get("pitch", ""),
        "mic_location_a": output_meta_a.get("mic_location", ""),
        "mic_location_b": output_meta_b.get("mic_location", ""),
        "foot_length_a": _extract_single_foot_length(item_a.meta.get("registration_raw", "")),
        "foot_length_b": _extract_single_foot_length(item_b.meta.get("registration_raw", "")),
        "organ_a": item_a.organ_id or "",
        "organ_b": item_b.organ_id or "",
        "source_path_a": item_a.source_path_rel,
        "source_path_b": item_b.source_path_rel,
        "toot_wav_path_a": item_a.toot_wav_path_rel,
        "toot_wav_path_b": item_b.toot_wav_path_rel,
        "batch": batch,
        "source_a_toot_index": "" if item_a.toot_index is None else item_a.toot_index,
        "source_b_toot_index": "" if item_b.toot_index is None else item_b.toot_index,
        "same_organ_pair": _pair_role_same_pipe_value(pair_role),
        "same_organ": _pair_role_same_organ_value(pair_role),
        "same_pipe_reference": _pair_role_same_pipe_value(pair_role),
        "same_organ_anchor": _pair_role_anchor_value(pair_role),
        "pair_role": pair_role,
        "processing_chain": item_a.processing_chain,
        "group_id": make_group_id(output_meta_a, dataset.group_keys),
        "pair_group_id": _pair_group_id(item_a, item_b, pair_role=pair_role, dataset=dataset),
    }


def _display_value(key: str, value: Any) -> Any:
    if key == "division" and value not in (None, ""):
        return str(value).replace("_", " ").title()
    if key == "mic_location" and value not in (None, ""):
        return str(value).title()
    if key == "family" and value not in (None, ""):
        return str(value).title()
    return value


def _output_meta(
    item: ListeningDatasetItem,
) -> dict[str, Any]:
    meta = dict(item.meta)
    meta["family"] = _display_value("family", meta.get("family", ""))
    meta["division"] = _display_value("division", meta.get("division", ""))
    meta["mic_location"] = _display_value("mic_location", meta.get("mic_location", ""))
    return meta


def _organ_sort_key(organ_id: str) -> tuple[int, int | str]:
    if organ_id in ALLOWED_ORGAN_IDS:
        return (0, ALLOWED_ORGAN_IDS.index(organ_id))
    return (1, organ_id)


def _select_tuning_reference_items(dataset: MarcussenDataset) -> dict[str, list[ParsedItem]]:
    selected_by_organ: dict[str, dict[str, ParsedItem]] = {}
    for item in dataset.iter_flat_items():
        meta = item.meta
        organ_id = meta.get("organ_id")
        pitch = meta.get("pitch")
        if organ_id in (None, "") or pitch in (None, ""):
            continue
        if meta.get("registration_raw") != TUNING_REFERENCE_REGISTRATION:
            continue
        if meta.get("division") != TUNING_REFERENCE_DIVISION:
            continue
        if str(meta.get("mic_location", "")).lower() != TUNING_REFERENCE_MIC_LOCATION:
            continue

        organ_key = str(organ_id)
        pitch_key = str(pitch)
        if pitch_key not in TUNING_REFERENCE_PITCHES:
            continue
        selected = selected_by_organ.setdefault(organ_key, {})
        if len(selected) >= TUNING_SAMPLE_COUNT or pitch_key in selected:
            continue
        selected[pitch_key] = item

    selected_items: dict[str, list[ParsedItem]] = {}
    for organ_id, selected in selected_by_organ.items():
        missing = [pitch for pitch in TUNING_REFERENCE_PITCHES if pitch not in selected]
        if missing:
            logger.warning(
                "Organ %s is missing tuning reference samples for %s %s %s: %s",
                organ_id,
                TUNING_REFERENCE_REGISTRATION,
                TUNING_REFERENCE_DIVISION,
                TUNING_REFERENCE_MIC_LOCATION,
                ", ".join(missing),
            )
        ordered = [selected[pitch] for pitch in TUNING_REFERENCE_PITCHES if pitch in selected]
        if ordered:
            selected_items[organ_id] = ordered
    return selected_items


def _build_organ_tuning_profiles(dataset: MarcussenDataset) -> dict[str, OrganTuningProfile]:
    selected_by_organ = _select_tuning_reference_items(dataset)
    if not selected_by_organ:
        return {}

    measurements_by_organ: dict[str, list[OrganTuningMeasurement]] = {}
    global_offsets: dict[str, float] = {}

    for organ_id in sorted(selected_by_organ, key=_organ_sort_key):
        measurements: list[OrganTuningMeasurement] = []
        for item in selected_by_organ[organ_id]:
            try:
                fundamental_hz, cents_offset = estimate_wrapped_pitch_cents_offset(
                    item.path,
                    expected_pitch=str(item.meta.get("pitch", "")),
                )
            except ValueError as exc:
                logger.warning("Skipping tuning estimate for %s: %s", item.path, exc)
                continue

            measurements.append(
                OrganTuningMeasurement(
                    pitch=str(item.meta.get("pitch", "")),
                    source_path_rel=_path_relative_to_root(item.path, dataset.root),
                    fundamental_hz=fundamental_hz,
                    wrapped_offset_cents=cents_offset,
                )
            )

        if not measurements:
            logger.warning("No usable tuning samples found for organ %s", organ_id)
            continue

        measurements_by_organ[organ_id] = measurements
        global_offsets[organ_id] = float(median(measurement.wrapped_offset_cents for measurement in measurements))

    if not global_offsets:
        return {}

    logger.info("Applying direct tuning correction from each organ's median wrapped pitch offset")

    profiles: dict[str, OrganTuningProfile] = {}
    for organ_id in sorted(global_offsets, key=_organ_sort_key):
        global_offset = global_offsets[organ_id]
        profiles[organ_id] = OrganTuningProfile(
            organ_id=organ_id,
            global_offset_cents=global_offset,
            applied_shift_cents=-global_offset,
            measurements=tuple(measurements_by_organ[organ_id]),
        )
    return profiles


def _write_tuning_report(report_path: Path, profiles: dict[str, OrganTuningProfile]) -> None:
    fieldnames = [
        "organ_id",
        "pitch",
        "source_path",
        "fundamental_hz",
        "wrapped_offset_cents",
        "global_offset_cents",
        "applied_shift_cents",
    ]
    with report_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for organ_id in sorted(profiles, key=_organ_sort_key):
            profile = profiles[organ_id]
            for measurement in profile.measurements:
                writer.writerow(
                    {
                        "organ_id": profile.organ_id,
                        "pitch": measurement.pitch,
                        "source_path": measurement.source_path_rel,
                        "fundamental_hz": f"{measurement.fundamental_hz:.6f}",
                        "wrapped_offset_cents": f"{measurement.wrapped_offset_cents:.6f}",
                        "global_offset_cents": f"{profile.global_offset_cents:.6f}",
                        "applied_shift_cents": f"{profile.applied_shift_cents:.6f}",
                    }
                )


def _prune_unreferenced_wavs(wav_root: Path, referenced_relpaths: set[str]) -> int:
    kept = 0
    if not wav_root.exists():
        return kept

    for wav_path in sorted(wav_root.rglob("*.wav")):
        relpath = str(wav_path.relative_to(wav_root.parent))
        if relpath in referenced_relpaths:
            kept += 1
            continue
        wav_path.unlink()

    for directory in sorted((path for path in wav_root.rglob("*") if path.is_dir()), reverse=True):
        try:
            directory.rmdir()
        except OSError:
            pass

    return kept


def _remove_stray_output_dirs(out_root: Path) -> None:
    stray_dir = out_root / "wav 2"
    if stray_dir.exists():
        shutil.rmtree(stray_dir)


def prepare_listening_dataset(
    dataset: MarcussenDataset,
    out_dir: str | Path,
    csv_name: str = "pairs.csv",
    max_pairs: int | None = None,
    *,
    expand_toots: bool = True,
    trim: bool = False,
    normalize: bool = False,
    steady_state_seconds: float | None = None,
    debug_first_n_per_batch: int | None = None,
) -> dict[str, Any]:
    """Write pair metadata CSV and processed toot WAV exports for listening tests."""
    if max_pairs is not None and max_pairs < 0:
        raise ValueError("max_pairs must be >= 0")
    if steady_state_seconds is not None and steady_state_seconds <= 0:
        raise ValueError("steady_state_seconds must be > 0")
    if debug_first_n_per_batch is not None and debug_first_n_per_batch <= 0:
        raise ValueError("debug_first_n_per_batch must be > 0")

    out_root = Path(out_dir)
    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    _remove_stray_output_dirs(out_root)
    organ_tuning_profiles = _build_organ_tuning_profiles(dataset)
    tuning_report_path = out_root / TUNING_REPORT_NAME
    _write_tuning_report(tuning_report_path, organ_tuning_profiles)
    csv_path = out_root / csv_name
    grouped = dataset.class_groups()

    fieldnames = [
        "pair_id",
        "family",
        "division",
        "registration_raw",
        "pitch",
        "mic_location",
        "family_a",
        "family_b",
        "division_a",
        "division_b",
        "registration_raw_a",
        "registration_raw_b",
        "pitch_a",
        "pitch_b",
        "mic_location_a",
        "mic_location_b",
        "foot_length_a",
        "foot_length_b",
        "organ_a",
        "organ_b",
        "source_path_a",
        "source_path_b",
        "toot_wav_path_a",
        "toot_wav_path_b",
        "batch",
        "source_a_toot_index",
        "source_b_toot_index",
        "same_organ_pair",
        "same_organ",
        "same_pipe_reference",
        "same_organ_anchor",
        "pair_role",
        "processing_chain",
        "group_id",
        "pair_group_id",
    ]

    rows_written = 0
    rows_written_by_batch: dict[str, int] = {}
    considered_groups = 0
    referenced_wav_paths: set[str] = set()
    control_rows: list[dict[str, Any]] = []
    anchor_candidate_items: list[ListeningDatasetItem] = []
    total_pairs = None

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        with tqdm(total=total_pairs, desc="Preparing listening dataset", unit="pair") as progress:
            target_batches = _target_batches(expand_toots=expand_toots)
            for _, source_items in grouped.items():
                if max_pairs is not None and rows_written >= max_pairs:
                    break
                if _all_batch_caps_reached(
                    rows_written_by_batch,
                    target_batches=target_batches,
                    debug_first_n_per_batch=debug_first_n_per_batch,
                ):
                    break

                group_has_rows = False
                items_by_batch: dict[str, list[ListeningDatasetItem]] = {}
                items_by_source: dict[str, list[ListeningDatasetItem]] = {}
                for source_item in source_items:
                    if max_pairs is not None and rows_written >= max_pairs:
                        break
                    if _all_batch_caps_reached(
                        rows_written_by_batch,
                        target_batches=target_batches,
                        debug_first_n_per_batch=debug_first_n_per_batch,
                    ):
                        break

                    prepared_items = _prepare_source_items(
                        source_item,
                        dataset_root=dataset.root,
                        wav_root=out_root / "wav",
                        organ_tuning_profiles=organ_tuning_profiles,
                        expand_toots=expand_toots,
                        trim=trim,
                        normalize=normalize,
                        steady_state_seconds=steady_state_seconds,
                    )

                    for prepared_item in prepared_items:
                        batch = prepared_item.batch
                        if (
                            debug_first_n_per_batch is not None
                            and rows_written_by_batch.get(batch, 0) >= debug_first_n_per_batch
                        ):
                            continue

                        existing_items = items_by_batch.setdefault(batch, [])
                        for item_a, item_b in _iter_incremental_pairs(
                            existing_items,
                            prepared_item,
                            expand_toots=expand_toots,
                        ):
                            if max_pairs is not None and rows_written >= max_pairs:
                                break
                            if (
                                debug_first_n_per_batch is not None
                                and rows_written_by_batch.get(batch, 0) >= debug_first_n_per_batch
                            ):
                                break
                            if not group_has_rows:
                                considered_groups += 1
                                group_has_rows = True

                            writer.writerow(
                                _make_csv_row(
                                    item_a,
                                    item_b,
                                    pair_id=rows_written + 1,
                                    batch=batch,
                                    pair_role=PAIR_ROLE_CROSS_ORGAN_MAIN,
                                    dataset=dataset,
                                )
                            )
                            referenced_wav_paths.add(item_a.toot_wav_path_rel)
                            referenced_wav_paths.add(item_b.toot_wav_path_rel)
                            rows_written += 1
                            rows_written_by_batch[batch] = rows_written_by_batch.get(batch, 0) + 1
                            progress.update(1)
                            if rows_written % 50 == 0:
                                handle.flush()

                        if debug_first_n_per_batch is None or rows_written_by_batch.get(batch, 0) < debug_first_n_per_batch:
                            existing_items.append(prepared_item)
                            items_by_source.setdefault(prepared_item.source_path_rel, []).append(prepared_item)
                            anchor_candidate_items.append(prepared_item)

                if expand_toots:
                    for source_path, source_prepared_items in items_by_source.items():
                        if max_pairs is not None and rows_written >= max_pairs:
                            break
                        for index_a, item_a in enumerate(source_prepared_items):
                            for item_b in source_prepared_items[index_a + 1 :]:
                                if max_pairs is not None and rows_written >= max_pairs:
                                    break
                                if not _is_same_organ_control_pair(item_a, item_b):
                                    continue
                                if (
                                    debug_first_n_per_batch is not None
                                    and rows_written_by_batch.get(SAME_ORGAN_CONTROL_BATCH, 0)
                                    >= debug_first_n_per_batch
                                ):
                                    break
                                control_rows.append(
                                    _make_csv_row(
                                        item_a,
                                        item_b,
                                        pair_id=rows_written + 1,
                                        batch=SAME_ORGAN_CONTROL_BATCH,
                                        pair_role=PAIR_ROLE_SAME_PIPE_REFERENCE,
                                        dataset=dataset,
                                    )
                                )
                                referenced_wav_paths.add(item_a.toot_wav_path_rel)
                                referenced_wav_paths.add(item_b.toot_wav_path_rel)
                                rows_written += 1
                                rows_written_by_batch[SAME_ORGAN_CONTROL_BATCH] = (
                                    rows_written_by_batch.get(SAME_ORGAN_CONTROL_BATCH, 0) + 1
                                )
                                progress.update(1)
                                if rows_written % 50 == 0:
                                    handle.flush()

            if expand_toots:
                for item_a, item_b in _iter_same_organ_anchor_pairs(anchor_candidate_items):
                    if max_pairs is not None and rows_written >= max_pairs:
                        break
                    if (
                        debug_first_n_per_batch is not None
                        and rows_written_by_batch.get(SAME_ORGAN_ANCHOR_BATCH, 0) >= debug_first_n_per_batch
                    ):
                        break
                    control_rows.append(
                        _make_csv_row(
                            item_a,
                            item_b,
                            pair_id=rows_written + 1,
                            batch=SAME_ORGAN_ANCHOR_BATCH,
                            pair_role=PAIR_ROLE_SAME_ORGAN_ANCHOR,
                            dataset=dataset,
                        )
                    )
                    referenced_wav_paths.add(item_a.toot_wav_path_rel)
                    referenced_wav_paths.add(item_b.toot_wav_path_rel)
                    rows_written += 1
                    rows_written_by_batch[SAME_ORGAN_ANCHOR_BATCH] = (
                        rows_written_by_batch.get(SAME_ORGAN_ANCHOR_BATCH, 0) + 1
                    )
                    progress.update(1)
                    if rows_written % 50 == 0:
                        handle.flush()

            for row in control_rows:
                writer.writerow(row)

        handle.flush()

    wav_files = (
        _prune_unreferenced_wavs(out_root / "wav", referenced_wav_paths)
        if debug_first_n_per_batch is not None
        else len(referenced_wav_paths)
    )
    _remove_stray_output_dirs(out_root)

    logger.info(
        "Finished listening dataset run: groups=%d rows=%d wav_files=%d out=%s expand_toots=%s",
        considered_groups,
        rows_written,
        wav_files,
        out_root,
        expand_toots,
    )
    return {
        "groups": considered_groups,
        "rows": rows_written,
        "wav_files": wav_files,
        "out_dir": str(out_root),
        "out_csv": str(csv_path),
        "tuning_report": str(tuning_report_path),
        "expand_toots": expand_toots,
        "debug_first_n_per_batch": debug_first_n_per_batch,
    }
