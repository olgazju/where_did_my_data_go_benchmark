"""Lineage-free baselines: solve the strong-id controls, fail the calibration traps."""
from poc.baselines import BASELINES, run_baselines
from poc.models import Label, Observability
from poc.observability import build_bundle
from poc.scenarios import by_id
from poc.simulator import simulate

MIN = Observability.MINIMAL


def _bundle(sid, level):
    b = by_id()[sid]
    return build_bundle(simulate(b.scenario), level, set(b.spec.partial_resolution_drop))


def test_strong_id_solves_positive_controls_at_minimal():
    for sid, final in (("S01", "ENT-01"), ("S02", "ENT-02"), ("S14", "ENT-14")):
        probe = by_id()[sid].spec.probe_source_id
        pred = BASELINES["strong_id"](_bundle(sid, MIN), probe)
        assert pred.status is Label.PRESENT_UNDER_OTHER_ID and pred.final_id == final


def test_name_similarity_fires_a_wrong_positive_on_calibration_probes():
    for sid in ("S09", "S10", "S12"):
        b = by_id()[sid]
        bundle = _bundle(sid, MIN)
        from poc.solver import solve
        assert solve(bundle, b.spec.probe_source_id).status is Label.UNDETERMINABLE
        pred = BASELINES["name_similarity"](bundle, b.spec.probe_source_id)
        assert pred.status is Label.PRESENT_UNDER_OTHER_ID  # baseline error, by design


def test_run_baselines_reports_unsolved_lineage_tasks():
    b = by_id()["S07"]
    rep = run_baselines(simulate(b.scenario),
                        [Observability.FULL, Observability.PARTIAL, Observability.MINIMAL],
                        ["S07-P"])
    # the wrong-merge probe at full is determinable but NOT solvable by lineage-free baselines
    assert "S07-P:full" in rep.unsolved_by_baselines_task_ids
