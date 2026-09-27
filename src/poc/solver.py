"""STEP 5 — the deterministic reference solver.

Consumes exactly the same :class:`ObservableBundle` the LLM receives (never the hidden
ground truth) and returns the label POSITIVELY ESTABLISHED by the available evidence,
together with the ``evidence_refs`` that license it. If nothing is positively
established — i.e. more than one outcome remains consistent with the evidence — the
answer is ``UNDETERMINABLE``.

Design guarantees (verified in the tests):

* The solver only reads ``bundle`` — there is no path to ``GroundTruthRecord``.
* Every rule has an artifact/scope *precondition*; removing an artifact can only stop a
  rule from firing, never start one. So removing evidence can turn a determinable answer
  into ``UNDETERMINABLE`` but can never invent a new, contradictory positive outcome
  (monotonicity).
"""

from __future__ import annotations

from .models import (
    ArtifactState,
    Label,
    ObservableBundle,
    Observability,
    ReferenceAnswer,
    StrongId,
)


# --------------------------------------------------------------------------------------
# Small observable helpers (operate purely on the bundle)
# --------------------------------------------------------------------------------------
def _source_strong_ids(src: dict) -> list[StrongId]:
    t = src.get("source_type")
    if t == "registry":
        return [StrongId(namespace=f"registry:{src['jurisdiction']}", value=src["registration_number"])]
    if t == "directory":
        return [StrongId(namespace="duns", value=src["duns"])] if src.get("duns") else []
    return []


def _final_strong_index(bundle: ObservableBundle) -> dict[str, dict]:
    """Map a strong-id key -> the final row that contains it."""
    index: dict[str, dict] = {}
    for row in bundle.final_dataset:
        for s in row.get("strong_ids", []):
            index[f"{s['namespace']}/{s['value']}"] = row
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
    """The merge-decision entry that lists S as a member, regardless of any flag."""
    art = bundle.artifact("merge_decision_log")
    if art.state is ArtifactState.MISSING:
        return None
    for entry in art.rows:
        if source_id in entry.get("members", []):
            return entry
    return None


def _observable_members(bundle: ObservableBundle, canon: str, merge_entry: dict | None) -> set[str]:
    """Source ids observably tied to a canonical, via the merge log and/or resolution map."""
    members: set[str] = set()
    if merge_entry is not None:
        members.update(merge_entry.get("members", []))
    res = bundle.artifact("resolution_map")
    if res.state is not ArtifactState.MISSING:
        members.update(r["source_id"] for r in res.rows if r.get("canonical_id") == canon)
    return members


def _strong_id_keys(bundle: ObservableBundle, source_id: str) -> set[str]:
    s = bundle.source(source_id)
    return {sid.key() for sid in _source_strong_ids(s)} if s else set()


def _canonical_has_observable_conflict(bundle: ObservableBundle, canon: str) -> bool:
    """True if two members of `canon` carry different values in the SAME namespace.

    Derived purely from the observable source records + membership evidence — it does NOT
    trust any precomputed `conflicting_strong_ids` flag on the merge-decision log.
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
        for sid in _source_strong_ids(bundle.source(m) or {}):
            if sid.namespace in seen and seen[sid.namespace] != sid.value:
                return True
            seen[sid.namespace] = sid.value
    return False


# --------------------------------------------------------------------------------------
# Solver
# --------------------------------------------------------------------------------------
def solve(bundle: ObservableBundle, source_id: str) -> ReferenceAnswer:
    src = bundle.source(source_id)
    if src is None:
        # No observable source record at all -> nothing can be established.
        return ReferenceAnswer(source_id=source_id, level=bundle.level, status=Label.UNDETERMINABLE)

    src_strong = _source_strong_ids(src)
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

    # R2 — PRESENT via the resolution map (source -> canonical -> final row).
    # Only for records with NO strong id (e.g. news mentions): the mapping is then the
    # only identity link. A record that DOES carry a strong id and is genuinely present
    # is already handled by R1; if R1 did not fire, its identity is absent from every
    # final row, so the mapping alone cannot prove it is present rather than absorbed.
    canon = _resolution_canonical(bundle, source_id)
    if canon is not None and canon in final_by_canon and not src_strong_keys:
        # Trust the mapping only if the canonical is not observably a conflicting (wrong)
        # merge — otherwise a no-strong-id member could belong to either absorbed entity.
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
    batch = src.get("batch")
    if batch is not None and manifest.is_complete_for(batch):
        ingested_ids = {r.get("source_id") for r in manifest.rows}
        if source_id not in ingested_ids:
            return ReferenceAnswer(
                source_id=source_id, level=bundle.level, status=Label.INGESTION_LOSS,
                evidence_refs=[
                    f"ingestion_manifest#complete_for={batch}",
                    f"ingestion_manifest#absent={source_id}",
                ],
            )

    # R6 — EXTRACTION_LOSS: S ingested, extraction log complete, S failed/absent there.
    extraction = bundle.artifact("extraction_log")
    ingested_ids = {r.get("source_id") for r in manifest.rows}
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

    # Nothing positively established.
    return ReferenceAnswer(source_id=source_id, level=bundle.level, status=Label.UNDETERMINABLE)


def _establish_wrong_merge(
    bundle: ObservableBundle,
    source_id: str,
    src_strong_keys: set[str],
    final_by_canon: dict[str, dict],
) -> ReferenceAnswer | None:
    """Establish WRONG_MERGE from OBSERVABLE records + mapping.

    The identity contradiction is always *rediscovered* from the members' own strong ids
    (via the merge-log membership and/or the resolution map); the precomputed
    ``conflicting_strong_ids`` flag is used only as a corroborating citation, never as the
    basis for the decision. WRONG_MERGE requires S to carry a strong id so we can tell it
    was the absorbed member (its identity is absent from the canonical's final row).
    """
    if not src_strong_keys:
        return None

    # Identify the canonical S resolved into (from merge-log membership or the map).
    merge_entry = _merge_entry_for(bundle, source_id)
    canon = merge_entry["canonical_id"] if merge_entry is not None else _resolution_canonical(bundle, source_id)
    if canon is None:
        return None

    # Rediscover a same-namespace contradiction between S and an observable sibling member.
    sibling = _contradicting_sibling(bundle, canon, source_id, src_strong_keys, merge_entry)
    if sibling is None:
        return None

    # S is absorbed iff the canonical's final row does NOT carry any of S's strong ids.
    final_row = final_by_canon.get(canon)
    if src_strong_keys & _final_keys(final_row):
        return None  # S's identity survives in the canonical -> not the absorbed one

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
    """Return (sibling_id, sibling_strong_id_key) that contradicts S in a shared namespace."""
    src_ns = {k.split("/", 1)[0]: k for k in src_strong_keys}
    for m in _observable_members(bundle, canon, merge_entry):
        if m == source_id:
            continue
        for sid in _source_strong_ids(bundle.source(m) or {}):
            if sid.namespace in src_ns and src_ns[sid.namespace] != sid.key():
                return m, sid.key()
    return None


def _final_keys(final_row: dict | None) -> set[str]:
    if not final_row:
        return set()
    return {f"{s['namespace']}/{s['value']}" for s in final_row.get("strong_ids", [])}


def solve_all(bundle: ObservableBundle, source_ids: list[str]) -> list[ReferenceAnswer]:
    return [solve(bundle, sid) for sid in source_ids]
