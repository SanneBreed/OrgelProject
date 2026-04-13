"""Reusable audio-processing functions for listening-dataset generation.

This module takes source recordings and applies export-oriented preparation
steps such as toot detection, trimming, normalization, steady-state extraction,
and WAV writing. It is the low-level processing layer used by the listening
dataset pipeline.

The preparation logic assumes that each original dataset recording contains
three organ "toots". A central purpose of this module is to detect those toot
regions and split a single source file into separate toot-level outputs, which
can then be used downstream both for cross-organ comparisons and for
same-recording control checks between different toots.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path

import numpy as np

from .audio import load_audio, write_wav

logger = logging.getLogger(__name__)

DEFAULT_EXPECTED_TOOTS = 3
DEFAULT_TOP_DB = 40.0
DEFAULT_MIN_SILENCE_SECONDS = 0.08
DEFAULT_MIN_CLIP_SECONDS = 0.08
DEFAULT_NORMALIZE_PEAK_DBFS = -1.0


@dataclass(slots=True, frozen=True)
class AudioSegment:
    """Detected toot region within a source file."""

    index: int
    start_sample: int
    end_sample: int
    start_s: float
    end_s: float


@dataclass(slots=True, frozen=True)
class PreparedAudioFile:
    """Written listening-dataset file plus source/toot metadata."""

    output_relpath: str
    output_path: str
    batch: str
    toot_index: int | None
    start_s: float | None
    end_s: float | None
    processing_chain: str


def _smoothed_power(y_mono: np.ndarray, sr: int) -> np.ndarray:
    window_samples = max(1, int(round(0.02 * sr)))
    power = np.square(np.asarray(y_mono, dtype=np.float32))
    kernel = np.ones(window_samples, dtype=np.float32) / float(window_samples)
    return np.convolve(power, kernel, mode="same")


def _active_intervals_from_power(
    power: np.ndarray,
    sr: int,
    *,
    top_db: float,
    min_silence_seconds: float,
    min_clip_seconds: float,
) -> list[tuple[int, int]]:
    if power.size == 0:
        return []

    max_power = float(np.max(power))
    if max_power <= 0.0:
        return []

    threshold = max_power / float(10 ** (top_db / 10.0))
    active_mask = power >= threshold
    changes = np.diff(active_mask.astype(np.int8), prepend=0, append=0)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)

    min_gap_samples = max(1, int(round(min_silence_seconds * sr)))
    min_clip_samples = max(1, int(round(min_clip_seconds * sr)))
    merged: list[tuple[int, int]] = []
    for start, end in zip(starts.tolist(), ends.tolist()):
        if end - start < min_clip_samples:
            continue
        if merged and start - merged[-1][1] <= min_gap_samples:
            prev_start, _ = merged[-1]
            merged[-1] = (prev_start, end)
            continue
        merged.append((start, end))
    return merged


def _find_toot_intervals(
    y_mono: np.ndarray,
    sr: int,
    *,
    expected_toots: int,
    top_db: float,
    min_silence_seconds: float,
    min_clip_seconds: float,
) -> list[tuple[int, int]]:
    power = _smoothed_power(y_mono, sr)
    rise = np.diff(power, prepend=power[0])
    min_spacing_samples = max(1, int(round(2.0 * sr)))

    onset_samples: list[int] = []
    for idx in np.argsort(rise)[::-1]:
        if rise[idx] <= 0:
            break
        if any(abs(int(idx) - existing) < min_spacing_samples for existing in onset_samples):
            continue
        onset_samples.append(int(idx))
        if len(onset_samples) == expected_toots:
            break

    onset_samples.sort()
    if len(onset_samples) == expected_toots:
        boundaries = [0]
        for start_idx, end_idx in zip(onset_samples, onset_samples[1:]):
            valley = int(np.argmin(power[start_idx:end_idx]) + start_idx)
            boundaries.append(valley)
        boundaries.append(len(power))
        return list(zip(boundaries[:-1], boundaries[1:]))

    intervals = _active_intervals_from_power(
        power,
        sr,
        top_db=top_db,
        min_silence_seconds=min_silence_seconds,
        min_clip_seconds=min_clip_seconds,
    )
    if len(intervals) > expected_toots:
        intervals = sorted(intervals, key=lambda pair: pair[1] - pair[0], reverse=True)[:expected_toots]
        intervals.sort(key=lambda pair: pair[0])
    return intervals


def detect_toot_segments(
    source_path: str | Path,
    *,
    expected_toots: int = DEFAULT_EXPECTED_TOOTS,
    top_db: float = DEFAULT_TOP_DB,
    min_silence_seconds: float = DEFAULT_MIN_SILENCE_SECONDS,
    min_clip_seconds: float = DEFAULT_MIN_CLIP_SECONDS,
) -> list[AudioSegment]:
    """Detect up to `expected_toots` toot regions in one source file."""
    y_mono, sr = load_audio(source_path, sr=None, mono=True)
    intervals = _find_toot_intervals(
        y_mono,
        sr,
        expected_toots=expected_toots,
        top_db=top_db,
        min_silence_seconds=min_silence_seconds,
        min_clip_seconds=min_clip_seconds,
    )
    if len(intervals) != expected_toots:
        logger.warning(
            "Detected %d toot segment(s) in %s, expected %d",
            len(intervals),
            source_path,
            expected_toots,
        )
    return [
        AudioSegment(
            index=idx,
            start_sample=start,
            end_sample=end,
            start_s=start / sr,
            end_s=end / sr,
        )
        for idx, (start, end) in enumerate(intervals, start=1)
    ]


def trim_audio(
    y: np.ndarray,
    sr: int,
    *,
    top_db: float = DEFAULT_TOP_DB,
    min_silence_seconds: float = DEFAULT_MIN_SILENCE_SECONDS,
    min_clip_seconds: float = DEFAULT_MIN_CLIP_SECONDS,
) -> np.ndarray:
    """Trim leading and trailing silence while preserving internal gaps."""
    y_mono = y if y.ndim == 1 else y.mean(axis=1)
    intervals = _active_intervals_from_power(
        _smoothed_power(y_mono, sr),
        sr,
        top_db=top_db,
        min_silence_seconds=min_silence_seconds,
        min_clip_seconds=min_clip_seconds,
    )
    if not intervals:
        return y
    start = intervals[0][0]
    end = intervals[-1][1]
    return y[start:end]


def normalize_audio(y: np.ndarray, *, peak_dbfs: float = DEFAULT_NORMALIZE_PEAK_DBFS) -> np.ndarray:
    """Peak-normalize audio to the requested dBFS target."""
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak <= 0.0:
        return y
    target_peak = float(10 ** (peak_dbfs / 20.0))
    normalized = np.asarray(y, dtype=np.float32) * np.float32(target_peak / peak)
    return np.clip(normalized, -1.0, 1.0)


def extract_steady_state(y: np.ndarray, sr: int, *, seconds: float) -> np.ndarray:
    """Extract a centered steady-state window from the active portion."""
    target_samples = max(1, int(round(seconds * sr)))
    if y.shape[0] <= target_samples:
        return y
    start = max(0, (y.shape[0] - target_samples) // 2)
    return y[start : start + target_samples]


def _processing_chain_label(
    *,
    trim: bool,
    normalize: bool,
    steady_state_seconds: float | None,
) -> str:
    steps: list[str] = []
    if trim:
        steps.append("trim")
    if steady_state_seconds is not None:
        steps.append(f"steady_state_{format(steady_state_seconds, 'g')}s")
    if normalize:
        steps.append("normalize")
    return "raw" if not steps else ";".join(steps)


def _output_relpath(
    relative_source_path: str | Path,
    *,
    batch: str,
    toot_index: int | None,
    processing_chain: str,
) -> str:
    rel_source = Path(relative_source_path)
    stem = rel_source.stem
    if toot_index is not None:
        stem = f"{stem}__toot{toot_index}"
    if processing_chain != "raw":
        stem = f"{stem}__{processing_chain.replace(';', '__')}"
    rel_output = rel_source.with_name(f"{stem}.wav")
    if batch == "full":
        return str(Path("wav") / rel_output)
    return str(Path("wav") / batch / rel_output)


def prepare_source_audio(
    source_path: str | Path,
    *,
    output_root: str | Path,
    relative_source_path: str | Path,
    expand_toots: bool = False,
    toot_indices: tuple[int, ...] | None = None,
    trim: bool = False,
    normalize: bool = False,
    steady_state_seconds: float | None = None,
    expected_toots: int = DEFAULT_EXPECTED_TOOTS,
    top_db: float = DEFAULT_TOP_DB,
    min_silence_seconds: float = DEFAULT_MIN_SILENCE_SECONDS,
    min_clip_seconds: float = DEFAULT_MIN_CLIP_SECONDS,
    normalize_peak_dbfs: float = DEFAULT_NORMALIZE_PEAK_DBFS,
) -> list[PreparedAudioFile]:
    """Process one source file and write one or more WAV exports."""
    y, sr = load_audio(source_path, sr=None, mono=False)
    processing_chain = _processing_chain_label(
        trim=trim,
        normalize=normalize,
        steady_state_seconds=steady_state_seconds,
    )
    segments: list[AudioSegment | None]
    if expand_toots:
        segments = detect_toot_segments(
            source_path,
            expected_toots=expected_toots,
            top_db=top_db,
            min_silence_seconds=min_silence_seconds,
            min_clip_seconds=min_clip_seconds,
        )
        if toot_indices is not None:
            allowed_toots = set(toot_indices)
            segments = [segment for segment in segments if segment.index in allowed_toots]
    else:
        segments = [None]

    prepared: list[PreparedAudioFile] = []
    output_root_path = Path(output_root)
    for segment in segments:
        batch = "full" if segment is None else f"toot_{segment.index}"
        relpath = _output_relpath(
            relative_source_path,
            batch=batch,
            toot_index=None if segment is None else segment.index,
            processing_chain=processing_chain,
        )
        clip = y if segment is None else y[segment.start_sample : segment.end_sample]
        if trim:
            clip = trim_audio(
                clip,
                sr,
                top_db=top_db,
                min_silence_seconds=min_silence_seconds,
                min_clip_seconds=min_clip_seconds,
            )
        if steady_state_seconds is not None:
            clip = extract_steady_state(clip, sr, seconds=steady_state_seconds)
        if normalize:
            clip = normalize_audio(clip, peak_dbfs=normalize_peak_dbfs)

        output_path = output_root_path / relpath
        write_wav(output_path, clip, sr)
        prepared.append(
            PreparedAudioFile(
                output_relpath=relpath,
                output_path=str(output_path),
                batch=batch,
                toot_index=None if segment is None else segment.index,
                start_s=None if segment is None else segment.start_s,
                end_s=None if segment is None else segment.end_s,
                processing_chain=processing_chain,
            )
        )

    return prepared
