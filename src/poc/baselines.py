"""Lineage-free deterministic baselines, showing that some tasks genuinely require lineage evidence.

* ``name_similarity``   — matches the source name to a final entity name (names only).
* ``strong_id``         — matches a namespaced strong id shared with a final record.
* ``source_final_only`` — strong id, then name similarity.

None reads the intermediate artifacts, so each answers identically at every observability level.
"""

from __future__ import annotations

import difflib
import re
from collections.abc import Callable

from .models import Label, ObservableBundle, Prediction, final_strong_keys, source_strong_ids

NAME_MATCH_THRESHOLD = 0.80


def _normalize(name: str) -> str:
    n = name.lower()
    n = re.sub(r"[.,]", " ", n)
    n = re.sub(r"\b(incorporated|inc|ltd|limited|llc|plc|partners|group|co|corp)\b", " ", n)
    return re.sub(r"\s+", " ", n).strip()


def _display_name(src: dict) -> str:
    for key in ("legal_name", "business_name", "org_mention"):
        if src.get(key):
            return src[key]
    return ""


def _present(source_id: str, final_id: str) -> Prediction:
    return Prediction(source_id=source_id, status=Label.PRESENT_UNDER_OTHER_ID, final_id=final_id)


def _undeterminable(source_id: str) -> Prediction:
    return Prediction(source_id=source_id, status=Label.UNDETERMINABLE)


# --------------------------------------------------------------------------------------
# The three baselines
# --------------------------------------------------------------------------------------
def name_similarity(bundle: ObservableBundle, source_id: str) -> Prediction:
    """Predict the final entity whose name best matches the source name.

    Args:
        bundle — Observable evidence; only sources and the final dataset are read.
        source_id — Probe source record.

    Returns:
        PRESENT_UNDER_OTHER_ID for the best match at or above ``NAME_MATCH_THRESHOLD``,
        else UNDETERMINABLE.
    """
    src = bundle.source(source_id)
    if not src:
        return _undeterminable(source_id)
    target = _normalize(_display_name(src))
    best_id, best_score = None, 0.0
    for row in bundle.final_dataset:
        score = difflib.SequenceMatcher(None, target, _normalize(row["canonical_name"])).ratio()
        if score > best_score:
            best_id, best_score = row["final_id"], score
    if best_id is not None and best_score >= NAME_MATCH_THRESHOLD:
        return _present(source_id, best_id)
    return _undeterminable(source_id)


def strong_id(bundle: ObservableBundle, source_id: str) -> Prediction:
    """Predict the final entity that shares a namespaced strong id with the source.

    Args:
        bundle — Observable evidence; only sources and the final dataset are read.
        source_id — Probe source record.

    Returns:
        PRESENT_UNDER_OTHER_ID for a strong-id match, else UNDETERMINABLE.
    """
    src = bundle.source(source_id)
    if not src:
        return _undeterminable(source_id)
    index = {key: row["final_id"] for row in bundle.final_dataset for key in final_strong_keys(row)}
    for sid in source_strong_ids(src):
        if sid.key() in index:
            return _present(source_id, index[sid.key()])
    return _undeterminable(source_id)


def source_final_only(bundle: ObservableBundle, source_id: str) -> Prediction:
    """Predict by strong id, falling back to name similarity.

    Args:
        bundle — Observable evidence; only sources and the final dataset are read.
        source_id — Probe source record.

    Returns:
        The strong-id prediction if it finds a match, else the name-similarity prediction.
    """
    by_id = strong_id(bundle, source_id)
    if by_id.status is Label.PRESENT_UNDER_OTHER_ID:
        return by_id
    return name_similarity(bundle, source_id)


BASELINES: dict[str, Callable[[ObservableBundle, str], Prediction]] = {
    "name_similarity": name_similarity,
    "strong_id": strong_id,
    "source_final_only": source_final_only,
}
