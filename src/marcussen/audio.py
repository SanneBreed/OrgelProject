"""Audio loading helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _require_soundfile():
    try:
        import soundfile as sf
    except ModuleNotFoundError as exc:  # pragma: no cover - env-specific
        raise RuntimeError("soundfile is required for audio operations. Install dependencies first.") from exc
    return sf


def get_info(path: str | Path) -> tuple[int, int, float, int]:
    """Return `(samplerate, frames, duration_s, channels)` for an audio file."""
    sf = _require_soundfile()
    info = sf.info(str(path))
    duration_s = float(info.frames) / float(info.samplerate)
    return info.samplerate, info.frames, duration_s, info.channels


def _resample_linear(y: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample waveform with linear interpolation (dependency-light fallback)."""
    if orig_sr == target_sr:
        return y

    frames = y.shape[0]
    target_frames = max(1, int(round(frames * target_sr / orig_sr)))

    old_x = np.linspace(0.0, 1.0, num=frames, endpoint=False)
    new_x = np.linspace(0.0, 1.0, num=target_frames, endpoint=False)

    if y.ndim == 1:
        return np.interp(new_x, old_x, y).astype(np.float32, copy=False)

    resampled = np.empty((target_frames, y.shape[1]), dtype=np.float32)
    for channel in range(y.shape[1]):
        resampled[:, channel] = np.interp(new_x, old_x, y[:, channel]).astype(np.float32, copy=False)
    return resampled


def load_audio(path: str | Path, sr: int | None = None, mono: bool = False) -> tuple[np.ndarray, int]:
    """Load audio with optional resampling and mono fold-down."""
    sf = _require_soundfile()
    y, native_sr = sf.read(str(path), dtype="float32", always_2d=False)
    y = np.asarray(y, dtype=np.float32)

    if mono and y.ndim == 2:
        y = y.mean(axis=1)

    out_sr = native_sr
    if sr is not None and sr != native_sr:
        y = _resample_linear(y, orig_sr=native_sr, target_sr=sr)
        out_sr = sr

    return y, out_sr


def write_wav(path: str | Path, y: np.ndarray, sr: int) -> None:
    """Write waveform data to a WAV file."""
    sf = _require_soundfile()
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), np.asarray(y, dtype=np.float32), sr, format="WAV")
