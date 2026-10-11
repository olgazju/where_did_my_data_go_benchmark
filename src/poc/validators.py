"""The official evidence-chain validator.

Two independent axes, plus the identifier check:

  * citation_integrity   — every cited ref names a REAL observable row (or a genuine
                           completeness guarantee), never a hallucinated or MISSING one.
  * evidence_sufficiency — the GROUNDED refs actually establish the claimed status.
  * identifier_evidence_validity — the reported final_id / merge_target_id is a PUBLIC
                           `ENT-…` id (never an internal canonical id), is cited, and matches
                           the reference (or is correctly null when not applicable/observable).

status_evidence_validity = integrity ∧ sufficiency ∧ status==reference.
full_evidence_validity   = status_evidence_validity ∧ identifier_evidence_validity.

Structured citations: {"artifact": "<name>", "match": {"<field>": "<value>"}}. `source_records`
and `final_dataset` support a synthetic "strong_id" field; list-valued fields (`strong_ids`,
`members`) match by membership or whole-list equality; absence-based labels may cite a
completeness guarantee via {"guaranteed_complete_for": "<scope>"}.
"""
from __future__ import annotations

from .models import (
    ArtifactState,
    Label,
    ObservableBundle,
    Prediction,
    ReferenceAnswer,
    final_strong_keys,
    source_strong_ids,
)

ALLOWED_ARTIFACTS = {"source_records", "final_dataset", "ingestion_manifest", "extraction_log",
                     "normalized_records", "resolution_map", "merge_decision_log", "filter_log"}


def _rows_for(bundle: ObservableBundle, artifact: str) -> list[dict] | None:
    if artifact == "source_records":
        return bundle.sources
    if artifact == "final_dataset":
        return bundle.final_dataset
    if artifact not in ALLOWED_ARTIFACTS:
        return None
    art = bundle.artifact(artifact)
    if art.state is ArtifactState.MISSING:
        return None
    return art.rows


def _text(v) -> str:
    """Return the comparable text of a stored or cited value; a strong id becomes ``namespace/value``."""
    if isinstance(v, dict) and {"namespace", "value"} <= v.keys():
        return f"{v['namespace']}/{v['value']}"
    return str(v)


def _value_matches(row_val, want) -> bool:
    if isinstance(row_val, bool):  # a JSON citation may spell it "false"
        return str(want).lower() == str(row_val).lower()
    if isinstance(row_val, list):
        members = [_text(x) for x in row_val]
        if isinstance(want, list):
            return [_text(x) for x in want] == members
        return _text(want) in members or str(want) == str(row_val)
    return str(row_val) == str(want)


def _field_matches(artifact: str, row: dict, field: str, want) -> bool:
    if field == "strong_id":
        if not isinstance(want, str):
            return False
        if artifact == "source_records":
            return want in {s.key() for s in source_strong_ids(row)}
        if artifact == "final_dataset":
            return want in final_strong_keys(row)
        return want in {_text(x) for x in row.get("strong_ids", [])}
    return _value_matches(row.get(field), want)


def _match_of(ref: dict) -> dict | None:
    """Return a ref's match fields, reading a strong id cited as ``namespace`` + ``value`` as one id.

    Args:
        ref — One structured citation from the model.

    Returns:
        The normalized ``match`` dict, or None if the ref is not a well-formed citation.
    """
    if not isinstance(ref, dict) or not isinstance(ref.get("artifact"), str):
        return None
    match = ref.get("match")
    if not isinstance(match, dict) or not match:
        return None
    if {"namespace", "value"} <= match.keys():
        rest = {k: v for k, v in match.items() if k not in ("namespace", "value")}
        return rest | {"strong_id": f"{match['namespace']}/{match['value']}"}
    return match


def _matched_rows(bundle: ObservableBundle, ref: dict) -> tuple[list[dict], bool]:
    """Return a ref's matched rows and whether it cites a real completeness guarantee."""
    match = _match_of(ref)
    if match is None:
        return [], False
    artifact = ref["artifact"]
    if "guaranteed_complete_for" in match:
        scope = match["guaranteed_complete_for"]
        return [], artifact in ALLOWED_ARTIFACTS and bundle.artifact(artifact).is_complete_for(scope)
    rows = _rows_for(bundle, artifact)
    if rows is None:
        return [], False
    hits = [r for r in rows if all(_field_matches(artifact, r, f, v) for f, v in match.items())]
    return hits, False


def _grounded(bundle: ObservableBundle, ref: dict) -> bool:
    rows, completeness = _matched_rows(bundle, ref)
    return bool(rows) or completeness


def _cited_artifacts(bundle: ObservableBundle, refs: list[dict]) -> set[str]:
    return {r.get("artifact") for r in refs if _grounded(bundle, r)}


# --------------------------------------------------------------------------------------
# Integrity + sufficiency
# --------------------------------------------------------------------------------------
def citation_integrity(bundle: ObservableBundle, refs: list[dict]) -> bool:
    """Check that every cited ref names a real observable row or completeness guarantee.

    Args:
        bundle — Observable evidence the model saw.
        refs — The model's structured ``evidence_refs``.

    Returns:
        True if ``refs`` is non-empty and every ref is grounded.
    """
    return bool(refs) and all(_grounded(bundle, r) for r in refs)


def _present_sufficient(bundle: ObservableBundle, refs: list[dict]) -> bool:
    """Check PRESENT support: a cited shared strong id, or cited resolution_map + final_dataset."""
    src_keys: set[str] = set()
    fin_keys: set[str] = set()
    for r in refs:
        rows, _ = _matched_rows(bundle, r)
        if not rows:
            continue
        fields = set(_match_of(r))
        if r["artifact"] == "source_records" and {"strong_id", "registration_number", "duns"} & fields:
            for row in rows:
                src_keys |= {s.key() for s in source_strong_ids(row)}
        if r["artifact"] == "final_dataset" and {"strong_id", "strong_ids"} & fields:
            for row in rows:
                fin_keys |= final_strong_keys(row)
    if src_keys & fin_keys:
        return True
    return {"resolution_map", "final_dataset"} <= _cited_artifacts(bundle, refs)


def evidence_sufficiency(bundle: ObservableBundle, reference: ReferenceAnswer,
                         claimed_status: Label, refs: list[dict]) -> bool:
    """Check that the grounded refs establish the claimed status.

    Args:
        bundle — Observable evidence the model saw.
        reference — The solver's reference answer.
        claimed_status — The status the model reported.
        refs — The model's structured ``evidence_refs``.

    Returns:
        True for ``UNDETERMINABLE`` (needs no evidence); otherwise True only if the status
        matches the reference and the grounded refs cite the artifacts that establish it.
    """
    if claimed_status is Label.UNDETERMINABLE:
        return True
    if claimed_status is not reference.status:
        return False
    if claimed_status is Label.PRESENT_UNDER_OTHER_ID:
        return _present_sufficient(bundle, refs)
    cited = _cited_artifacts(bundle, refs)
    if claimed_status is Label.INGESTION_LOSS:
        return "ingestion_manifest" in cited
    if claimed_status is Label.EXTRACTION_LOSS:
        return "extraction_log" in cited
    if claimed_status is Label.FILTERED_OUT:
        return "filter_log" in cited
    if claimed_status is Label.WRONG_MERGE:
        return bool({"merge_decision_log", "resolution_map"} & cited)
    return False


def status_evidence_validity(bundle: ObservableBundle, reference: ReferenceAnswer,
                             claimed_status: Label, refs: list[dict]) -> bool:
    """Check citation integrity and evidence sufficiency for the claimed status.

    Args:
        bundle — Observable evidence the model saw.
        reference — The solver's reference answer.
        claimed_status — The status the model reported.
        refs — The model's structured ``evidence_refs``.

    Returns:
        True if both checks pass; always True for ``UNDETERMINABLE``, so callers that score
        correctness must also check the status against the reference.
    """
    if claimed_status is Label.UNDETERMINABLE:
        return True
    return (citation_integrity(bundle, refs)
            and evidence_sufficiency(bundle, reference, claimed_status, refs))


# --------------------------------------------------------------------------------------
# Identifier
# --------------------------------------------------------------------------------------
def _cites_final_id(bundle: ObservableBundle, refs: list[dict], final_id: str) -> bool:
    return any(_grounded(bundle, r) and r.get("artifact") == "final_dataset"
               and str((r.get("match") or {}).get("final_id")) == str(final_id) for r in refs)


def identifier_evidence_validity(bundle: ObservableBundle, reference: ReferenceAnswer,
                                 claimed_final_id: str | None, claimed_merge_target_id: str | None,
                                 refs: list[dict]) -> bool:
    """Check the reported ids against the reference's status-specific id requirements.

    Args:
        bundle — Observable evidence the model saw.
        reference — The solver's reference answer.
        claimed_final_id — The model's ``final_id``.
        claimed_merge_target_id — The model's ``merge_target_id``.
        refs — The model's structured ``evidence_refs``.

    Returns:
        True if every required id is a public final-dataset id equal to the reference and cited
        via ``final_dataset``, and every non-applicable or unobservable id is null.
    """
    public_ids = {r["final_id"] for r in bundle.final_dataset}

    if reference.status is Label.PRESENT_UNDER_OTHER_ID:
        if claimed_final_id not in public_ids or claimed_final_id != reference.final_id:
            return False
        if claimed_merge_target_id is not None:
            return False
        return _cites_final_id(bundle, refs, claimed_final_id)

    if reference.status is Label.WRONG_MERGE:
        if claimed_final_id is not None:
            return False
        if reference.merge_target_observable:
            if (claimed_merge_target_id not in public_ids
                    or claimed_merge_target_id != reference.merge_target_id):
                return False
            return _cites_final_id(bundle, refs, claimed_merge_target_id)
        return claimed_merge_target_id is None

    return claimed_final_id is None and claimed_merge_target_id is None


def chain_validity(bundle: ObservableBundle, reference: ReferenceAnswer, prediction: Prediction,
                   refs: list[dict]) -> dict[str, bool]:
    """Run every evidence-chain check on one well-formed prediction.

    Args:
        bundle — Observable evidence the model saw.
        reference — The solver's reference answer.
        prediction — The model's parsed answer.
        refs — The model's structured ``evidence_refs``.

    Returns:
        ``citation_integrity``, ``evidence_sufficiency``, ``status_evidence_validity``,
        ``identifier_evidence_validity`` and ``full_evidence_validity`` (status and ids both valid).
    """
    status_ok = status_evidence_validity(bundle, reference, prediction.status, refs)
    ids_ok = identifier_evidence_validity(bundle, reference, prediction.final_id,
                                          prediction.merge_target_id, refs)
    return {
        "citation_integrity": citation_integrity(bundle, refs),
        "evidence_sufficiency": evidence_sufficiency(bundle, reference, prediction.status, refs),
        "status_evidence_validity": status_ok,
        "identifier_evidence_validity": ids_ok,
        "full_evidence_validity": status_ok and ids_ok,
    }
