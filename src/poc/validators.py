"""The official evidence-chain validator (formerly 'v2').

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

from .models import ArtifactState, Label, ObservableBundle, ReferenceAnswer

ALLOWED_ARTIFACTS = {"source_records", "final_dataset", "ingestion_manifest", "extraction_log",
                     "normalized_records", "resolution_map", "merge_decision_log", "filter_log"}

_STATUS_SHAPE = {
    Label.PRESENT_UNDER_OTHER_ID: "present",
    Label.INGESTION_LOSS: "ingestion",
    Label.EXTRACTION_LOSS: "extraction",
    Label.FILTERED_OUT: "filtered",
    Label.WRONG_MERGE: "wrong_merge",
}


def _source_strong_keys(src: dict) -> set[str]:
    t = src.get("source_type")
    if t == "registry":
        return {f"registry:{src['jurisdiction']}/{src['registration_number']}"}
    if t == "directory" and src.get("duns"):
        return {f"duns/{src['duns']}"}
    return set()


def _final_strong_keys(row: dict) -> set[str]:
    return {f"{s['namespace']}/{s['value']}" for s in row.get("strong_ids", [])}


def _rows_for(bundle: ObservableBundle, artifact: str):
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


def _value_matches(row_val, want) -> bool:
    if isinstance(row_val, list):
        members = [str(x) for x in row_val]
        if isinstance(want, list):
            return [str(x) for x in want] == members
        return str(want) in members or str(want) == str(row_val)
    return str(row_val) == str(want)


def _field_matches(artifact: str, row: dict, field: str, want) -> bool:
    if field == "strong_id":
        if artifact == "source_records":
            return want in _source_strong_keys(row)
        if artifact == "final_dataset":
            return want in _final_strong_keys(row)
        return False
    return _value_matches(row.get(field), want)


def _matched_rows(bundle: ObservableBundle, ref: dict):
    """Rows matched by a structured ref, plus a completeness-guarantee flag."""
    if not isinstance(ref, dict):
        return [], False
    artifact, match = ref.get("artifact"), ref.get("match") or {}
    if not isinstance(match, dict) or not match:
        return [], False
    if "guaranteed_complete_for" in match:
        art = bundle.artifact(artifact) if artifact in ALLOWED_ARTIFACTS else None
        scope = match["guaranteed_complete_for"]
        if art is not None and art.state is ArtifactState.COMPLETE and scope in art.complete_scopes:
            return [], True
        return [], False
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
    return bool(refs) and all(_grounded(bundle, r) for r in refs)


def _present_sufficient(bundle: ObservableBundle, refs: list[dict]) -> bool:
    src_keys, fin_keys = set(), set()
    for r in refs:
        if not _grounded(bundle, r):
            continue
        match = r.get("match") or {}
        rows, _ = _matched_rows(bundle, r)
        if r.get("artifact") == "source_records" and ({"strong_id", "registration_number", "duns"} & set(match)):
            for row in rows:
                src_keys |= _source_strong_keys(row)
        if r.get("artifact") == "final_dataset" and ({"strong_id", "strong_ids"} & set(match)):
            for row in rows:
                fin_keys |= _final_strong_keys(row)
    if src_keys & fin_keys:
        return True
    return {"resolution_map", "final_dataset"} <= _cited_artifacts(bundle, refs)


def evidence_sufficiency(bundle: ObservableBundle, reference: ReferenceAnswer,
                         claimed_status: Label, refs: list[dict]) -> bool:
    if claimed_status is Label.UNDETERMINABLE:
        return True
    if claimed_status is not reference.status:
        return False
    shape = _STATUS_SHAPE.get(claimed_status)
    cited = _cited_artifacts(bundle, refs)
    if shape == "present":
        return _present_sufficient(bundle, refs)
    if shape == "ingestion":
        return "ingestion_manifest" in cited
    if shape == "extraction":
        return "extraction_log" in cited
    if shape == "filtered":
        return "filter_log" in cited
    if shape == "wrong_merge":
        return bool({"merge_decision_log", "resolution_map"} & cited)
    return False


def status_evidence_validity(bundle: ObservableBundle, reference: ReferenceAnswer,
                             claimed_status: Label, refs: list[dict]) -> bool:
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
                                 claimed_status: Label, claimed_final_id: str | None,
                                 claimed_merge_target_id: str | None, refs: list[dict]) -> bool:
    final_ids = {r["final_id"] for r in bundle.final_dataset}
    canon_ids = {r["canonical_id"] for r in bundle.final_dataset}

    def is_public(x: str | None) -> bool:
        return x in final_ids and not (x in canon_ids and x not in final_ids)

    if reference.status is Label.PRESENT_UNDER_OTHER_ID:
        if not is_public(claimed_final_id) or claimed_final_id != reference.final_id:
            return False
        if claimed_merge_target_id is not None:
            return False
        return _cites_final_id(bundle, refs, claimed_final_id)

    if reference.status is Label.WRONG_MERGE:
        if claimed_final_id is not None:
            return False
        if reference.merge_target_observable:
            if not is_public(claimed_merge_target_id) or claimed_merge_target_id != reference.merge_target_id:
                return False
            return _cites_final_id(bundle, refs, claimed_merge_target_id)
        return claimed_merge_target_id is None

    return claimed_final_id is None and claimed_merge_target_id is None


def full_evidence_validity(bundle: ObservableBundle, reference: ReferenceAnswer,
                           claimed_status: Label, claimed_final_id: str | None,
                           claimed_merge_target_id: str | None, refs: list[dict]) -> bool:
    return (status_evidence_validity(bundle, reference, claimed_status, refs)
            and identifier_evidence_validity(bundle, reference, claimed_status,
                                             claimed_final_id, claimed_merge_target_id, refs))
