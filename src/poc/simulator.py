"""STEP 3 — the pipeline simulator.

Runs the hand-authored scenario through the six stages and produces, from that single
execution:

* the hidden :class:`GroundTruthRecord` history for every source record, and
* the *master* set of observable artifacts (every artifact COMPLETE, all rows present).

The observability layer (:mod:`poc.observability`) later redacts this master set into
Full / Partial / Minimal views. Because everything derives from here, the artifacts can
never be mutually inconsistent.
"""

from __future__ import annotations

import re

from pydantic import BaseModel

from .models import Artifact, ArtifactState, GroundTruthRecord, Label, StrongId
from .scenario import Fate, Scenario, ScenarioRecord


class SimulationResult(BaseModel):
    scenario: Scenario
    ground_truth: list[GroundTruthRecord]
    sources: list[dict]
    final_dataset: list[dict]
    master_artifacts: dict[str, Artifact]

    def gt(self, source_id: str) -> GroundTruthRecord:
        for r in self.ground_truth:
            if r.source_id == source_id:
                return r
        raise KeyError(source_id)


def _normalize(name: str) -> str:
    """A deliberately simple name normalizer (lowercase, drop legal suffixes/punct)."""
    n = name.lower()
    n = re.sub(r"[.,]", " ", n)
    n = re.sub(r"\b(incorporated|inc|ltd|limited|llc|plc|partners|group|co|corp)\b", " ", n)
    n = re.sub(r"\s+", " ", n).strip()
    return n


def _strong_ids_of(rec: ScenarioRecord) -> list[StrongId]:
    return rec.source.strong_ids()


def _conflicting_strong_ids(members_strong_ids: list[StrongId]) -> bool:
    """True if two members carry different values in the SAME namespace (a contradiction)."""
    seen: dict[str, str] = {}
    for s in members_strong_ids:
        if s.namespace in seen and seen[s.namespace] != s.value:
            return True
        seen[s.namespace] = s.value
    return False


def simulate(scenario: Scenario) -> SimulationResult:
    recs = scenario.records

    final_by_canonical = {fe.canonical_id: fe for fe in scenario.final_entities}
    # canonical_id -> source_ids resolved into it (any fate that assigns a canonical)
    members_by_canonical: dict[str, list[str]] = {}
    for r in recs:
        if r.canonical_id:
            members_by_canonical.setdefault(r.canonical_id, []).append(r.source.source_id)

    # ---- ground truth ---------------------------------------------------------------
    ground_truth: list[GroundTruthRecord] = []
    for r in recs:
        sid = r.source.source_id
        cid = r.canonical_id
        merged_with = [m for m in members_by_canonical.get(cid, []) if m != sid] if cid else []

        if r.fate is Fate.INGESTION_LOSS:
            gt = GroundTruthRecord(
                source_id=sid, true_entity_id=r.true_entity_id,
                ingested=False, extracted=False, normalized=False,
                actual_outcome=Label.INGESTION_LOSS,
                disappearance_reason="Ingestion failed; record never entered the pipeline.",
            )
        elif r.fate is Fate.EXTRACTION_LOSS:
            gt = GroundTruthRecord(
                source_id=sid, true_entity_id=r.true_entity_id,
                ingested=True, extracted=False, normalized=False,
                actual_outcome=Label.EXTRACTION_LOSS,
                disappearance_reason="Ingested but structured extraction failed.",
            )
        elif r.fate is Fate.WRONG_MERGE_ABSORBED:
            gt = GroundTruthRecord(
                source_id=sid, true_entity_id=r.true_entity_id,
                ingested=True, extracted=True, normalized=True,
                canonical_id=cid, merged_with=merged_with, is_wrong_merge=True,
                final_id=None,
                actual_outcome=Label.WRONG_MERGE,
                disappearance_reason=f"Distinct entity incorrectly merged into canonical {cid}.",
            )
        elif r.fate is Fate.FILTERED_OUT:
            gt = GroundTruthRecord(
                source_id=sid, true_entity_id=r.true_entity_id,
                ingested=True, extracted=True, normalized=True,
                canonical_id=cid, merged_with=merged_with,
                filtered_out=True, filter_rule=r.filter_rule, final_id=None,
                actual_outcome=Label.FILTERED_OUT,
                disappearance_reason=f"Excluded from final dataset by business rule '{r.filter_rule}'.",
            )
        else:  # RESOLVED_PRESENT
            fe = final_by_canonical[cid]
            gt = GroundTruthRecord(
                source_id=sid, true_entity_id=r.true_entity_id,
                ingested=True, extracted=True, normalized=True,
                canonical_id=cid, merged_with=merged_with, final_id=fe.final_id,
                actual_outcome=Label.PRESENT_UNDER_OTHER_ID,
                disappearance_reason=None,
            )
        ground_truth.append(gt)

    gt_by_id = {g.source_id: g for g in ground_truth}

    # ---- observable payloads (constant across levels) -------------------------------
    sources = [r.source.model_dump() for r in recs]
    final_dataset = [
        {
            "final_id": fe.final_id,
            "canonical_id": fe.canonical_id,
            "canonical_name": fe.canonical_name,
            "strong_ids": [s.model_dump() for s in fe.strong_ids],
        }
        for fe in scenario.final_entities
    ]

    # ---- master artifacts (all COMPLETE) --------------------------------------------
    ingestion_rows = [
        {"source_id": r.source.source_id, "batch": r.source.batch, "status": "ingested"}
        for r in recs if gt_by_id[r.source.source_id].ingested
    ]
    extraction_rows = [
        {
            "source_id": r.source.source_id,
            "status": "ok" if gt_by_id[r.source.source_id].extracted else "failed",
        }
        for r in recs if gt_by_id[r.source.source_id].ingested
    ]
    normalized_rows = [
        {
            "source_id": r.source.source_id,
            "normalized_name": _normalize(r.source.display_name()),
            "strong_ids": [s.key() for s in _strong_ids_of(r)],
        }
        for r in recs if gt_by_id[r.source.source_id].normalized
    ]
    resolution_rows = [
        {"source_id": r.source.source_id, "canonical_id": r.canonical_id}
        for r in recs if r.canonical_id
    ]

    merge_rows = []
    for cid, member_ids in members_by_canonical.items():
        if len(member_ids) < 2:
            continue
        member_recs = [r for r in recs if r.source.source_id in member_ids]
        member_sids: list[StrongId] = []
        for mr in member_recs:
            member_sids.extend(_strong_ids_of(mr))
        conflict = _conflicting_strong_ids(member_sids)
        merge_rows.append({
            "canonical_id": cid,
            "members": sorted(member_ids),
            "merge_key": "fuzzy_name",
            "conflicting_strong_ids": conflict,
        })
    merge_rows.sort(key=lambda x: x["canonical_id"])

    # filter log: one decision per canonical that reached the filtering stage.
    filter_rows = []
    seen_canonicals = sorted({r.canonical_id for r in recs if r.canonical_id})
    for cid in seen_canonicals:
        if cid in final_by_canonical:
            filter_rows.append({"canonical_id": cid, "kept": True, "rule": None})
        else:
            frec = next((r for r in recs if r.canonical_id == cid and r.fate is Fate.FILTERED_OUT), None)
            rule = frec.filter_rule if frec else "unknown"
            filter_rows.append({"canonical_id": cid, "kept": False, "rule": rule})

    master_artifacts = {
        "ingestion_manifest": Artifact(
            name="ingestion_manifest", state=ArtifactState.COMPLETE,
            complete_scopes=list(scenario.ingestion_batches), rows=ingestion_rows,
        ),
        "extraction_log": Artifact(
            name="extraction_log", state=ArtifactState.COMPLETE,
            complete_scopes=["extraction"], rows=extraction_rows,
        ),
        "normalized_records": Artifact(
            name="normalized_records", state=ArtifactState.COMPLETE,
            complete_scopes=["normalization"], rows=normalized_rows,
        ),
        "resolution_map": Artifact(
            name="resolution_map", state=ArtifactState.COMPLETE,
            complete_scopes=["resolution"], rows=resolution_rows,
        ),
        "merge_decision_log": Artifact(
            name="merge_decision_log", state=ArtifactState.COMPLETE,
            complete_scopes=["merge"], rows=merge_rows,
        ),
        "filter_log": Artifact(
            name="filter_log", state=ArtifactState.COMPLETE,
            complete_scopes=["filter"], rows=filter_rows,
        ),
    }

    return SimulationResult(
        scenario=scenario,
        ground_truth=ground_truth,
        sources=sources,
        final_dataset=final_dataset,
        master_artifacts=master_artifacts,
    )
