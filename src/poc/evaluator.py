"""STEP 6/8 — the deterministic evaluator.

Parses a model's JSON answer, validates it, and compares it with the reference answer.
No LLM-as-a-judge. Two scores are reported: status correctness and full-answer
correctness. Malformed responses are tracked separately (and count as incorrect).
"""

from __future__ import annotations

import json

from pydantic import BaseModel, ValidationError

from .models import Label, Observability, Prediction, ReferenceAnswer


class ParseResult(BaseModel):
    ok: bool
    prediction: Prediction | None = None
    error: str | None = None


def parse_prediction(raw: str) -> ParseResult:
    """Tolerantly extract the JSON object and validate it into a :class:`Prediction`.

    Structural requirements (a failure here marks the answer *malformed*):
      * a JSON object is present,
      * ``source_id`` is present and non-empty, and
      * ``status`` is one of the defined labels.
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

    try:
        pred = Prediction(
            source_id=source_id,
            status=status,
            final_id=_none_or_str(data.get("final_id")),
            merge_target_id=_none_or_str(data.get("merge_target_id")),
        )
    except ValidationError as e:  # pragma: no cover - defensive
        return ParseResult(ok=False, error=str(e))
    return ParseResult(ok=True, prediction=pred)


def _none_or_str(v) -> str | None:
    if v is None:
        return None
    s = str(v).strip()
    return s or None


class EvalRecord(BaseModel):
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
    """Enforce status-specific identifier requirements AND that non-applicable id fields
    are null. Called only when the status already matches the reference."""
    if pred.status is Label.PRESENT_UNDER_OTHER_ID:
        # final_id is required and must match; merge_target_id is non-applicable -> null.
        return pred.final_id == reference.final_id and pred.merge_target_id is None
    if pred.status is Label.WRONG_MERGE:
        # final_id is non-applicable -> null.
        if pred.final_id is not None:
            return False
        # merge_target_id required and matched only when observable; otherwise it must be
        # null (a target asserted without supporting evidence is a guess, not an answer).
        if reference.merge_target_observable:
            return pred.merge_target_id == reference.merge_target_id
        return pred.merge_target_id is None
    # INGESTION_LOSS / EXTRACTION_LOSS / FILTERED_OUT / UNDETERMINABLE: both ids null.
    return pred.final_id is None and pred.merge_target_id is None


class EvalSummary(BaseModel):
    total: int
    scored: int  # non-malformed
    malformed: int
    status_correct: int
    full_correct: int
    status_accuracy: float
    full_accuracy: float
    by_level: dict[str, dict[str, float]]
    records: list[EvalRecord]


def summarize(records: list[EvalRecord]) -> EvalSummary:
    total = len(records)
    malformed = sum(r.malformed for r in records)
    status_correct = sum(r.status_correct for r in records)
    full_correct = sum(r.full_correct for r in records)

    by_level: dict[str, dict[str, float]] = {}
    for lvl in {r.level for r in records}:
        rs = [r for r in records if r.level == lvl]
        by_level[lvl.value] = {
            "total": len(rs),
            "status_accuracy": _safe_div(sum(r.status_correct for r in rs), len(rs)),
            "full_accuracy": _safe_div(sum(r.full_correct for r in rs), len(rs)),
            "malformed": sum(r.malformed for r in rs),
        }

    return EvalSummary(
        total=total,
        scored=total - malformed,
        malformed=malformed,
        status_correct=status_correct,
        full_correct=full_correct,
        status_accuracy=_safe_div(status_correct, total),
        full_accuracy=_safe_div(full_correct, total),
        by_level=by_level,
        records=records,
    )


def _safe_div(a: int, b: int) -> float:
    return round(a / b, 4) if b else 0.0
