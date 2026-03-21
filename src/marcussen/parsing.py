"""Filename parsing for Marcussen organ recordings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

from .constants import (
    DIVISION_NORMALIZATION,
    detect_organ_id,
    expand_registration,
    expand_tokens,
    is_pitch_token,
    looks_like_registration,
    normalize_pitch,
)

_NOTE_DISTANCE_RE = re.compile(
    r"^(?P<prefix>.+?)\s+(?P<note>[^\s]+)\s+(?P<distance>CLOSE|DISTANT)$",
    re.IGNORECASE,
)
_DISTANCE_RE = re.compile(r"^(?P<prefix>.+?)\s+(?P<distance>CLOSE|DISTANT)$", re.IGNORECASE)


@dataclass(slots=True)
class ParsedItem:
    """Parsed representation of a filename and its extracted metadata."""

    path: str
    ext: str
    raw_stem: str
    meta: dict[str, Any]
    extras: list[str]
    warnings: list[str]


def _tokenize(text: str) -> list[str]:
    return [token for token in re.split(r"\s+", text.strip()) if token]


def _infer_family(path_obj: Path) -> str | None:
    parts = path_obj.parts
    for idx, part in enumerate(parts):
        if part.lower() == "sounds" and idx > 0:
            return parts[idx - 1]
    if path_obj.parent.name:
        return path_obj.parent.name
    return None


def _parse_with_patterns(raw_stem: str, warnings: list[str]) -> dict[str, Any]:
    for pattern_name, pattern in (("note_distance", _NOTE_DISTANCE_RE), ("distance_only", _DISTANCE_RE)):
        match = pattern.match(raw_stem)
        if not match:
            continue

        result: dict[str, Any] = {
            "prefix": match.group("prefix"),
            "pattern": pattern_name,
            "distance": match.group("distance").lower(),
        }

        note = match.groupdict().get("note")
        if note is not None:
            if is_pitch_token(note):
                result["pitch"] = normalize_pitch(note)
            else:
                warnings.append(f"Pattern matched but pitch token looked unusual: {note!r}")
        return result

    warnings.append("No regex pattern matched; using full stem as prefix")
    return {"prefix": raw_stem, "pattern": "fallback"}


def _extract_division(tokens: list[str]) -> tuple[str | None, int, int]:
    lowered = [token.lower() for token in tokens]
    for width in range(len(tokens), 0, -1):
        for start in range(0, len(tokens) - width + 1):
            phrase = " ".join(lowered[start : start + width])
            if phrase in DIVISION_NORMALIZATION:
                return DIVISION_NORMALIZATION[phrase], start, width
    return None, -1, 0


def parse_filename(path_like: str) -> ParsedItem:
    """Parse one filename into canonical metadata, extras, and warnings.

    Raises:
        ValueError: if an unknown organ ID is detected.
    """

    warnings: list[str] = []
    extras: list[str] = []
    meta: dict[str, Any] = {}

    path_obj = Path(path_like)
    ext = path_obj.suffix.lower()
    raw_stem = path_obj.stem

    if ext != ".flac":
        warnings.append(f"Unexpected extension {ext!r}; parser tuned for .flac")

    family = _infer_family(path_obj)
    if family:
        meta["family"] = family

    parsed = _parse_with_patterns(raw_stem, warnings)
    prefix = str(parsed.get("prefix", raw_stem)).strip()
    tokens = _tokenize(prefix)

    if "distance" in parsed:
        meta["mic_location"] = str(parsed["distance"])
    if "pitch" in parsed:
        meta["pitch"] = str(parsed["pitch"])
        
    if "normalised" in str(path_obj).lower():
        meta["normalisation"] = "yes"
    else:
        meta["normalisation"] = "no"

    if not tokens:
        warnings.append("No tokens found in filename stem")
        return ParsedItem(
            path=str(path_like),
            ext=ext,
            raw_stem=raw_stem,
            meta=meta,
            extras=extras,
            warnings=warnings,
        )

    expanded_all = expand_tokens(tokens)
    for key, value in expanded_all.items():
        meta.setdefault(key, value)

    registration_idx: int | None = None
    for idx, token in enumerate(tokens):
        if looks_like_registration(token):
            registration_idx = idx
            break

    if registration_idx is None:
        warnings.append("Could not identify registration token")
        return ParsedItem(
            path=str(path_like),
            ext=ext,
            raw_stem=raw_stem,
            meta=meta,
            extras=extras,
            warnings=warnings,
        )

    registration = tokens[registration_idx]
    meta.update(expand_registration(registration))

    organ_id_raw = " ".join(tokens[:registration_idx]).strip()
    if not organ_id_raw:
        raise ValueError(f"Missing organ_id in filename: {path_like}")
    try:
        meta["organ_id"] = detect_organ_id(organ_id_raw)
    except ValueError as exc:
        raise ValueError(f"{path_like}: {exc}") from exc

    remaining_tokens = tokens[registration_idx + 1 :]
    if remaining_tokens:
        division, div_start, div_width = _extract_division(remaining_tokens)
        if division:
            meta["division"] = division
        leftovers = [
            token
            for idx, token in enumerate(remaining_tokens)
            if not (div_start <= idx < div_start + div_width)
        ]

        for token in leftovers:
            if token.upper() == "BEATING":
                meta["beating"] = True
                continue
            token_expanded = expand_tokens([token])
            before = set(meta.keys())
            for key, value in token_expanded.items():
                meta.setdefault(key, value)
            if set(meta.keys()) == before:
                extras.append(token)
    elif "division" not in meta:
        warnings.append("Division token(s) not detected")

    return ParsedItem(
        path=str(path_like),
        ext=ext,
        raw_stem=raw_stem,
        meta=meta,
        extras=extras,
        warnings=warnings,
    )
