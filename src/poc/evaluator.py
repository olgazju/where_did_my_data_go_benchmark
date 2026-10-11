"""Deterministic evaluator: parse a model's JSON answer and score it against the reference.

Two scores are reported, status correctness and full-answer correctness; malformed responses
are tracked separately and count as incorrect.
"""

from __future__ import annotations

import json

from pydantic import BaseModel

from .models import Label, Observability, Prediction, ReferenceAnswer


class ParseResult(BaseModel):
    """Outcome of parsing a raw model response."""

    ok: bool
    prediction: Prediction | None = None
    evidence_refs: list = []
    error: str | None = None


def parse_prediction(raw: str) -> ParseResult:
    """Extract the outermost JSON object from a response and validate it as a Prediction.

    Args:
        raw — Raw model response text; surrounding prose or code fences are tolerated.

    Returns:
        A successful result with the prediction and its ``evidence_refs`` list (empty when absent
        or not a list), or a failed (malformed) result with an error when no JSON object is
        found, ``source_id`` is missing or empty, or ``status`` is not a defined label.
    """
    text = raw.strip()
    # Allow a fenced code block or surrounding prose: take the outermost {...}.
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return ParseResult(ok=False, error="no JSON object found")
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError as e:
        return ParseResult(ok=False, error=f"invalid JSON: {e}")
    if not isinstance(data, dict):
        return ParseResult(ok=False, error="JSON is not an object")

    source_id = _none_or_str(data.get("source_id"))
    if source_id is None:
        return ParseResult(ok=False, error="missing source_id")

    status_raw = data.get("status")
    try:
        status = Label(status_raw)
    except ValueError:
        return ParseResult(ok=False, error=f"invalid status: {status_raw!r}")

    pred = Prediction(
        source_id=source_id,
        status=status,
        final_id=_none_or_str(data.get("final_id")),
        merge_target_id=_none_or_str(data.get("merge_target_id")),
    )
    refs = data.get("evidence_refs")
    return ParseResult(ok=True, prediction=pred, evidence_refs=refs if isinstance(refs, list) else [])


def _none_or_str(v: object) -> str | None:
    if v is None:
        return None
    s = str(v).strip()
    return s or None


class EvalRecord(BaseModel):
    """Scored outcome of one model answer for one task."""

    task_id: str
    source_id: str
    level: Observability
    reference_status: Label
    predicted_status: Label | None = None
    status_correct: bool = False
    full_correct: bool = False
    malformed: bool = False
    error: str | None = None


def evaluate_one(task_id: str, reference: ReferenceAnswer, raw: str) -> EvalRecord:
    """Parse one raw model answer and score it against the reference.

    Args:
        task_id — Identifier recorded on the result.
        reference — The solver's reference answer for the task.
        raw — Raw model response text.

    Returns:
        The scored record; unparseable answers and answers about a different ``source_id``
        are marked malformed and incorrect.
    """
    parsed = parse_prediction(raw)
    rec = EvalRecord(
        task_id=task_id,
        source_id=reference.source_id,
        level=reference.level,
        reference_status=reference.status,
    )
    if not parsed.ok:
        rec.malformed = True
        rec.error = parsed.error
        return rec

    pred = parsed.prediction
    assert pred is not None
    # A prediction about a different record is not a valid answer to this task.
    if pred.source_id != reference.source_id:
        rec.malformed = True
        rec.error = f"source_id mismatch: {pred.source_id!r} != {reference.source_id!r}"
        return rec

    rec.predicted_status = pred.status
    rec.status_correct = pred.status == reference.status
    rec.full_correct = rec.status_correct and _fields_valid_and_match(reference, pred)
    return rec


def _fields_valid_and_match(reference: ReferenceAnswer, pred: Prediction) -> bool:
    """Check the status-specific id fields of a prediction whose status matches the reference.

    Args:
        reference — The solver's reference answer.
        pred — The parsed prediction.

    Returns:
        True if every required id matches the reference and every non-applicable id is null.
    """
    if pred.status is Label.PRESENT_UNDER_OTHER_ID:
        return pred.final_id == reference.final_id and pred.merge_target_id is None
    if pred.status is Label.WRONG_MERGE:
        if pred.final_id is not None:
            return False
        # An unobservable target must be null: asserting one without evidence is a guess.
        if reference.merge_target_observable:
            return pred.merge_target_id == reference.merge_target_id
        return pred.merge_target_id is None
    return pred.final_id is None and pred.merge_target_id is None
