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
import shutil
from typing import Any

from tqdm.auto import tqdm

from .audio_prep import prepare_source_audio
from .dataset import MarcussenDataset, make_group_id
from .pairs import is_cross_organ_pair
from .parsing import ParsedItem

logger = logging.getLogger(__name__)

LISTENING_DATASET_TOOT_INDICES = (1, 2)
CROSS_ORGAN_TOOT_INDEX = 1
SAME_ORGAN_CONTROL_TOOT_PAIR = (1, 2)


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


SAME_ORGAN_CONTROL_BATCH = "same_organ_control"


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
    expand_toots: bool,
    trim: bool,
    normalize: bool,
    steady_state_seconds: float | None,
) -> list[ListeningDatasetItem]:
    prepared_items: list[ListeningDatasetItem] = []
    rel_source_path = _path_relative_to_root(item.path, dataset_root)
    prepared_files = prepare_source_audio(
        item.path,
        output_root=wav_root.parent,
        relative_source_path=rel_source_path,
        expand_toots=expand_toots,
        toot_indices=LISTENING_DATASET_TOOT_INDICES if expand_toots else None,
        trim=trim,
        normalize=normalize,
        steady_state_seconds=steady_state_seconds,
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
    return [f"toot_{CROSS_ORGAN_TOOT_INDEX}", SAME_ORGAN_CONTROL_BATCH]


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


def _make_csv_row(
    item_a: ListeningDatasetItem,
    item_b: ListeningDatasetItem,
    *,
    pair_id: int,
    batch: str,
    dataset: MarcussenDataset,
) -> dict[str, Any]:
    output_meta = _output_meta(item_a)
    return {
        "pair_id": pair_id,
        "family": output_meta.get("family", ""),
        "division": output_meta.get("division", ""),
        "registration_raw": item_a.meta.get("registration_raw", ""),
        "pitch": item_a.meta.get("pitch", ""),
        "mic_location": output_meta.get("mic_location", ""),
        "organ_a": item_a.organ_id or "",
        "organ_b": item_b.organ_id or "",
        "source_path_a": item_a.source_path_rel,
        "source_path_b": item_b.source_path_rel,
        "toot_wav_path_a": item_a.toot_wav_path_rel,
        "toot_wav_path_b": item_b.toot_wav_path_rel,
        "batch": batch,
        "source_a_toot_index": "" if item_a.toot_index is None else item_a.toot_index,
        "source_b_toot_index": "" if item_b.toot_index is None else item_b.toot_index,
        "same_organ_pair": "True" if _is_same_organ_control_pair(item_a, item_b) else "False",
        "processing_chain": item_a.processing_chain,
        "group_id": make_group_id(output_meta, dataset.group_keys),
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
    csv_path = out_root / csv_name
    grouped = dataset.class_groups()

    fieldnames = [
        "pair_id",
        "family",
        "division",
        "registration_raw",
        "pitch",
        "mic_location",
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
        "processing_chain",
        "group_id",
    ]

    rows_written = 0
    rows_written_by_batch: dict[str, int] = {}
    considered_groups = 0
    referenced_wav_paths: set[str] = set()
    control_rows: list[dict[str, Any]] = []
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
        "expand_toots": expand_toots,
        "debug_first_n_per_batch": debug_first_n_per_batch,
    }
