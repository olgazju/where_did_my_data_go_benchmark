"""Lineage-free baselines: solve the strong-id controls, fail the calibration traps."""
from poc.baselines import BASELINES
from poc.benchmark import scenario_bundle
from poc.evaluator import evaluate_one
from poc.models import Label
from poc.scenarios import by_id
from poc.solver import solve


def test_strong_id_solves_positive_controls_at_minimal():
    for sid, final in (("S01", "ENT-01"), ("S02", "ENT-02"), ("S14", "ENT-14")):
        b = by_id()[sid]
        pred = BASELINES["strong_id"](scenario_bundle(b, "minimal"), b.spec.probe_source_id)
        assert pred.status is Label.PRESENT_UNDER_OTHER_ID and pred.final_id == final


def test_name_similarity_fires_a_wrong_positive_on_calibration_probes():
    for sid in ("S09", "S10", "S12"):
        b = by_id()[sid]
        bundle = scenario_bundle(b, "minimal")
        assert solve(bundle, b.spec.probe_source_id).status is Label.UNDETERMINABLE
        pred = BASELINES["name_similarity"](bundle, b.spec.probe_source_id)
        assert pred.status is Label.PRESENT_UNDER_OTHER_ID  # baseline error, by design


def test_no_baseline_solves_the_wrong_merge_probe():
    b = by_id()["S07"]
    bundle = scenario_bundle(b, "full")
    ref = solve(bundle, b.spec.probe_source_id)
    assert ref.status is Label.WRONG_MERGE
    for fn in BASELINES.values():
        answer = fn(bundle, b.spec.probe_source_id).model_dump_json()
        assert not evaluate_one("S07:full", ref, answer).full_correct
