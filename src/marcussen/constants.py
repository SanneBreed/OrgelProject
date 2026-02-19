"""Minimal token helpers and constants for Marcussen filename parsing."""

from __future__ import annotations

import re
from typing import Any

ALLOWED_ORGAN_IDS: tuple[str, ...] = (
    "Copenhagen Grundtvigs",
    "Rotterdam",
    "Linz Dom",
    "Stockholm Oscar",
)
_ALLOWED_ORGAN_ID_SET = set(ALLOWED_ORGAN_IDS)

DIVISION_NORMALIZATION: dict[str, str] = {
    "main division": "main_division",
    "upper division": "upper_division",
    "positive back": "positive_back",
    "back positive": "positive_back",
    "positive console": "positive_console",
    "pedal": "pedal",
}

MIC_LOCATION_NORMALIZATION: dict[str, str] = {
    "close": "close",
    "distant": "distant",
}

REGISTRATION_STOP_PREFIX: dict[str, str] = {
    "F": "Flute",
    "P": "Principal",
    "O": "Octave",
    "Q": "Quint",
    "N": "Nazard",
    "G": "Gedackt",
    "SG": "String",
    "RQ": "Rauschquint",
    "VDG": "Viola da Gamba",
    "VC": "Voix Celeste",
    "UM": "Unda Maris",
    "OB": "Oboe",
}

REGISTRATION_STOP_TOKEN: dict[str, str] = {
    "MIX": "Mixture",
    "CYMBEL": "Cymbel",
    "SESQ": "Sesquialtera",
    "SCH": "Scharf",
    "RQ": "Rauschquint",
    "BAARPIJP": "Baarpijp",
}

_PITCH_RE = re.compile(r"^(?:[A-Ga-g](?:#|b)?\d?|[cC]\d?)$")
_REG_COMPONENT_RE = re.compile(r"^([A-Za-z]+)(\d+)?$")
_REG_TOKEN_RE = re.compile(r"^[A-Za-z][A-Za-z0-9]*(?:\+[A-Za-z][A-Za-z0-9]*)*$")


def detect_organ_id(organ_id: str) -> str:
    """Validate and return the canonical organ ID.

    Raises:
        ValueError: if the provided organ ID is not one of the allowed values.
    """
    normalized = " ".join(organ_id.split()).strip()
    if normalized in _ALLOWED_ORGAN_ID_SET:
        return normalized
    allowed = ", ".join(ALLOWED_ORGAN_IDS)
    raise ValueError(f"Unknown organ_id {normalized!r}. Allowed values: {allowed}")


def normalize_pitch(pitch: str) -> str:
    """Normalize pitch tokens while preserving symbolic uppercase C."""
    token = pitch.strip()
    if not token:
        return token
    if token == "C":
        return "C"
    if len(token) == 1:
        return token.lower()
    return token[0].lower() + token[1:]


def is_pitch_token(token: str) -> bool:
    """Return True when a token looks like a pitch annotation."""
    return bool(_PITCH_RE.match(token.strip()))


def looks_like_registration(token: str) -> bool:
    """Return True when a token appears to encode stop registration shorthand."""
    cleaned = token.strip()
    if not cleaned or not _REG_TOKEN_RE.match(cleaned):
        return False
    if "+" in cleaned:
        return True
    upper = cleaned.upper()
    if upper in REGISTRATION_STOP_TOKEN:
        return True
    match = _REG_COMPONENT_RE.match(cleaned)
    if not match:
        return False
    prefix, foot = match.groups()
    return prefix.upper() in REGISTRATION_STOP_PREFIX or upper in REGISTRATION_STOP_TOKEN or foot is not None


def _expand_registration_component(component: str) -> str:
    part = component.strip()
    upper = part.upper()
    if upper in REGISTRATION_STOP_TOKEN:
        return REGISTRATION_STOP_TOKEN[upper]

    match = _REG_COMPONENT_RE.match(part)
    if not match:
        return part

    prefix, foot = match.groups()
    family = REGISTRATION_STOP_PREFIX.get(prefix.upper(), prefix)
    if foot is None:
        return family
    return f"{family} {foot}'"


def expand_registration(registration: str) -> dict[str, Any]:
    """Expand a registration shorthand token into structured fields."""
    parts = [p for p in registration.split("+") if p]
    expanded = [_expand_registration_component(part) for part in parts]
    return {
        "registration_raw": registration,
        "registration_parts": parts,
        "registration_expanded": " + ".join(expanded),
    }


def _extract_phrase(tokens: list[str], mapping: dict[str, str]) -> str | None:
    if not tokens:
        return None
    lowered = [t.lower() for t in tokens]
    for width in range(len(tokens), 0, -1):
        for start in range(0, len(tokens) - width + 1):
            phrase = " ".join(lowered[start : start + width])
            if phrase in mapping:
                return mapping[phrase]
    return None


def expand_tokens(tokens: list[str]) -> dict[str, Any]:
    """Expand shorthand tokens into canonical fields when recognized."""
    fields: dict[str, Any] = {}
    cleaned = [t.strip() for t in tokens if t.strip()]
    if not cleaned:
        return fields

    division = _extract_phrase(cleaned, DIVISION_NORMALIZATION)
    if division:
        fields["division"] = division

    for token in cleaned:
        lower = token.lower()

        if "mic_location" not in fields and lower in MIC_LOCATION_NORMALIZATION:
            fields["mic_location"] = MIC_LOCATION_NORMALIZATION[lower]
            continue

        if "pitch" not in fields and is_pitch_token(token):
            fields["pitch"] = normalize_pitch(token)
            continue

        if "registration_raw" not in fields and looks_like_registration(token):
            fields.update(expand_registration(token))

    return fields
