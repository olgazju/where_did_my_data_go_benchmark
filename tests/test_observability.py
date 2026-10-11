"""Observability: derive Full/Partial/Minimal; redaction only ever removes information."""
import json

from poc.models import ArtifactState, Observability
from poc.observability import build_bundle
from poc.scenarios import by_id
from poc.simulator import simulate

LEVELS = [Observability.FULL, Observability.PARTIAL, Observability.MINIMAL]


def test_minimal_drops_intermediate_artifacts():
    sim = simulate(by_id()["S07"].scenario)
    minimal = build_bundle(sim, Observability.MINIMAL)
    assert minimal.artifact("merge_decision_log").state is ArtifactState.MISSING
    assert minimal.artifact("resolution_map").state is ArtifactState.MISSING
    assert minimal.artifact("ingestion_manifest").state is ArtifactState.PARTIAL


def test_partial_drops_merge_log_and_named_resolution_rows():
    b = by_id()["S07"]
    sim = simulate(b.scenario)
    partial = build_bundle(sim, Observability.PARTIAL, set(b.spec.partial_resolution_drop))
    assert partial.artifact("merge_decision_log").state is ArtifactState.MISSING
    res_ids = {r["source_id"] for r in partial.artifact("resolution_map").rows}
    assert "S07-P" not in res_ids  # the dropped probe row


def test_redaction_never_adds_rows_or_guarantees():
    sim = simulate(by_id()["S03"].scenario)
    master = sim.master_artifacts
    for name, m in master.items():
        master_rows = {json.dumps(r, sort_keys=True) for r in m.rows}
        prev = master_rows
        for lvl in LEVELS:
            art = build_bundle(sim, lvl).artifacts[name]
            rows = {json.dumps(r, sort_keys=True) for r in art.rows}
            assert rows <= master_rows and rows <= prev
            assert set(art.complete_scopes) <= set(m.complete_scopes)
            prev = rows


def test_sources_and_final_constant_across_levels():
    sim = simulate(by_id()["S01"].scenario)
    full = build_bundle(sim, Observability.FULL)
    base_s = json.dumps(full.sources, sort_keys=True)
    base_f = json.dumps(full.final_dataset, sort_keys=True)
    for lvl in LEVELS:
        b = build_bundle(sim, lvl)
        assert json.dumps(b.sources, sort_keys=True) == base_s
        assert json.dumps(b.final_dataset, sort_keys=True) == base_f
