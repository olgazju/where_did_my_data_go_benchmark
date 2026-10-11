"""Deterministic reference solver.

Reads only the :class:`ObservableBundle` the model receives (never hidden ground truth) and
returns the label positively established by the evidence, with the ``evidence_refs`` that
license it; if more than one outcome remains consistent, the answer is ``UNDETERMINABLE``.

Every rule has an artifact/scope precondition, so removing evidence can only stop a rule from
firing, never start one (monotonicity).
"""

from __future__ import annotations

from .models import (
    ArtifactState,
    Label,
    ObservableBundle,
    ReferenceAnswer,
    final_strong_keys,
    source_strong_ids,
)


# --------------------------------------------------------------------------------------
# Small observable helpers (operate purely on the bundle)
# --------------------------------------------------------------------------------------
def _final_strong_index(bundle: ObservableBundle) -> dict[str, dict]:
    """Map each strong-id key to the final row that carries it."""
    index: dict[str, dict] = {}
    for row in bundle.final_dataset:
        for key in final_strong_keys(row):
            index[key] = row
    return index


def _final_by_canonical(bundle: ObservableBundle) -> dict[str, dict]:
    return {row["canonical_id"]: row for row in bundle.final_dataset}


def _resolution_canonical(bundle: ObservableBundle, source_id: str) -> str | None:
    art = bundle.artifact("resolution_map")
    if art.state is ArtifactState.MISSING:
        return None
    for row in art.rows:
        if row.get("source_id") == source_id:
            return row.get("canonical_id")
    return None


def _merge_entry_for(bundle: ObservableBundle, source_id: str) -> dict | None:
    """Return the merge-decision entry listing ``source_id`` as a member, if observable."""
    art = bundle.artifact("merge_decision_log")
    if art.state is ArtifactState.MISSING:
        return None
    for entry in art.rows:
        if source_id in entry.get("members", []):
            return entry
    return None


def _observable_members(bundle: ObservableBundle, canon: str, merge_entry: dict | None) -> set[str]:
    """Return the source ids tied to ``canon`` by the merge entry and/or the resolution map."""
    members: set[str] = set()
    if merge_entry is not None:
        members.update(merge_entry.get("members", []))
    res = bundle.artifact("resolution_map")
    if res.state is not ArtifactState.MISSING:
        members.update(r["source_id"] for r in res.rows if r.get("canonical_id") == canon)
    return members


def _canonical_has_observable_conflict(bundle: ObservableBundle, canon: str) -> bool:
    """Check whether two observable members of a canonical carry conflicting strong ids.

    Args:
        bundle — Observable evidence for one level.
        canon — Canonical id whose members are compared.

    Returns:
        True if two members carry different values in the same namespace, derived from the
        source records only (never from the ``conflicting_strong_ids`` flag).
    """
    entry = None
    art = bundle.artifact("merge_decision_log")
    if art.state is not ArtifactState.MISSING:
        for e in art.rows:
            if e.get("canonical_id") == canon:
                entry = e
                break
    members = _observable_members(bundle, canon, entry)
    seen: dict[str, str] = {}
    for m in members:
        for sid in source_strong_ids(bundle.source(m) or {}):
            if sid.namespace in seen and seen[sid.namespace] != sid.value:
                return True
            seen[sid.namespace] = sid.value
    return False


# --------------------------------------------------------------------------------------
# Solver
# --------------------------------------------------------------------------------------
def solve(bundle: ObservableBundle, source_id: str) -> ReferenceAnswer:
    """Derive the evidence-supported reference answer for one source record.

    Args:
        bundle — Observable evidence for one level.
        source_id — Id of the investigated source record.

    Returns:
        The first label established by rules R1–R7 with its ``evidence_refs``, else
        ``UNDETERMINABLE``.
    """
    src = bundle.source(source_id)
    if src is None:
        return ReferenceAnswer(source_id=source_id, level=bundle.level, status=Label.UNDETERMINABLE)

    src_strong = source_strong_ids(src)
    final_strong = _final_strong_index(bundle)
    final_by_canon = _final_by_canonical(bundle)

    # R1 — PRESENT via a shared strong identifier (works at every level; sources+final).
    for s in src_strong:
        row = final_strong.get(s.key())
        if row is not None:
            return ReferenceAnswer(
                source_id=source_id, level=bundle.level,
                status=Label.PRESENT_UNDER_OTHER_ID, final_id=row["final_id"],
                evidence_refs=[
                    f"source:{source_id}#strong_id={s.key()}",
                    f"final_dataset:{row['final_id']}#strong_id={s.key()}",
                ],
            )

    src_strong_keys = {s.key() for s in src_strong}

    # R3/R4 — WRONG_MERGE: S is a member of a canonical that mixed distinct entities AND
    # S's own identity is NOT represented in that canonical's final row (S was absorbed).
    wrong = _establish_wrong_merge(bundle, source_id, src_strong_keys, final_by_canon)
    if wrong is not None:
        return wrong

    # R2 — PRESENT via the resolution map (source -> canonical -> final row), only for
    # records without a strong id: a strong-id record that R1 missed is absent from every
    # final row, so the mapping cannot tell present from absorbed. A conflicting canonical
    # is not trusted either, since the record could belong to either merged entity.
    canon = _resolution_canonical(bundle, source_id)
    if canon is not None and canon in final_by_canon and not src_strong_keys:
        if not _canonical_has_observable_conflict(bundle, canon):
            row = final_by_canon[canon]
            return ReferenceAnswer(
                source_id=source_id, level=bundle.level,
                status=Label.PRESENT_UNDER_OTHER_ID, final_id=row["final_id"],
                evidence_refs=[
                    f"resolution_map:{source_id}->{canon}",
                    f"final_dataset:{row['final_id']}#canonical_id={canon}",
                ],
            )

    # R5 — INGESTION_LOSS: manifest guaranteed complete for S's batch and S not ingested.
    manifest = bundle.artifact("ingestion_manifest")
    ingested_ids = {r.get("source_id") for r in manifest.rows}
    batch = src.get("batch")
    if batch is not None and manifest.is_complete_for(batch) and source_id not in ingested_ids:
        return ReferenceAnswer(
            source_id=source_id, level=bundle.level, status=Label.INGESTION_LOSS,
            evidence_refs=[
                f"ingestion_manifest#complete_for={batch}",
                f"ingestion_manifest#absent={source_id}",
            ],
        )

    # R6 — EXTRACTION_LOSS: S ingested, extraction log complete, S failed/absent there.
    extraction = bundle.artifact("extraction_log")
    if source_id in ingested_ids and extraction.is_complete_for("extraction"):
        status_by_id = {r.get("source_id"): r.get("status") for r in extraction.rows}
        st = status_by_id.get(source_id)
        if st in (None, "failed", "empty"):
            return ReferenceAnswer(
                source_id=source_id, level=bundle.level, status=Label.EXTRACTION_LOSS,
                evidence_refs=[
                    f"ingestion_manifest#present={source_id}",
                    f"extraction_log#complete;status={st or 'absent'}",
                ],
            )

    # R7 — FILTERED_OUT: S -> canonical C (resolution present), filter log complete, C dropped.
    filter_log = bundle.artifact("filter_log")
    if canon is not None and filter_log.is_complete_for("filter"):
        for row in filter_log.rows:
            if row.get("canonical_id") == canon and row.get("kept") is False:
                return ReferenceAnswer(
                    source_id=source_id, level=bundle.level, status=Label.FILTERED_OUT,
                    evidence_refs=[
                        f"resolution_map:{source_id}->{canon}",
                        f"filter_log:{canon}#kept=False;rule={row.get('rule')}",
                    ],
                )

    return ReferenceAnswer(source_id=source_id, level=bundle.level, status=Label.UNDETERMINABLE)


def _establish_wrong_merge(
    bundle: ObservableBundle,
    source_id: str,
    src_strong_keys: set[str],
    final_by_canon: dict[str, dict],
) -> ReferenceAnswer | None:
    """Establish WRONG_MERGE for a record absorbed into a canonical that mixes distinct entities.

    Args:
        bundle — Observable evidence for one level.
        source_id — Id of the investigated source record (S).
        src_strong_keys — S's strong-id keys; WRONG_MERGE requires at least one.
        final_by_canon — Final rows keyed by canonical id.

    Returns:
        A WRONG_MERGE answer if an observable sibling contradicts S in a shared namespace and
        S's strong ids are absent from the canonical's final row, else None. The
        ``conflicting_strong_ids`` flag is only cited as corroboration, never decided on.
    """
    if not src_strong_keys:
        return None

    merge_entry = _merge_entry_for(bundle, source_id)
    canon = merge_entry["canonical_id"] if merge_entry is not None else _resolution_canonical(bundle, source_id)
    if canon is None:
        return None

    sibling = _contradicting_sibling(bundle, canon, source_id, src_strong_keys, merge_entry)
    if sibling is None:
        return None

    # S is the absorbed member iff the canonical's final row carries none of S's strong ids.
    final_row = final_by_canon.get(canon)
    if src_strong_keys & final_strong_keys(final_row):
        return None

    merge_target_id = final_row["final_id"] if final_row is not None else None
    sib_id, sib_key = sibling
    evidence = [
        f"resolution/merge#member={source_id}->{canon}",
        f"records#same_namespace_contradiction:{source_id} vs {sib_id}({sib_key})",
    ]
    if merge_entry is not None and merge_entry.get("conflicting_strong_ids"):
        evidence.append(f"merge_decision_log:{canon}#conflicting_strong_ids=true(corroborating)")
    if merge_target_id:
        evidence.append(f"final_dataset:{merge_target_id}#missing_strong_id_for={source_id}")
    return ReferenceAnswer(
        source_id=source_id, level=bundle.level, status=Label.WRONG_MERGE,
        merge_target_id=merge_target_id,
        merge_target_observable=merge_target_id is not None,
        evidence_refs=evidence,
    )


def _contradicting_sibling(
    bundle: ObservableBundle,
    canon: str,
    source_id: str,
    src_strong_keys: set[str],
    merge_entry: dict | None,
) -> tuple[str, str] | None:
    """Find an observable sibling member whose strong id contradicts S's in a shared namespace.

    Args:
        bundle — Observable evidence for one level.
        canon — Canonical id S resolved into.
        source_id — Id of the investigated source record (S).
        src_strong_keys — S's strong-id keys.
        merge_entry — Merge-decision entry listing S as a member, if observable.

    Returns:
        ``(sibling_id, sibling_strong_id_key)`` for the first contradicting sibling, else None.
    """
    src_ns = {k.split("/", 1)[0]: k for k in src_strong_keys}
    for m in _observable_members(bundle, canon, merge_entry):
        if m == source_id:
            continue
        for sid in source_strong_ids(bundle.source(m) or {}):
            if sid.namespace in src_ns and src_ns[sid.namespace] != sid.key():
                return m, sid.key()
    return None
