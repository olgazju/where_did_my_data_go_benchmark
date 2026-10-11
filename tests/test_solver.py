"""Reference solver: observable-only, monotone, and never sees ground truth."""
import copy
import inspect
import random

from poc import solver
from poc.models import ArtifactState, Label, Observability
from poc.observability import build_bundle
from poc.scenarios import build_all, by_id
from poc.simulator import simulate
from poc.solver import solve

ORDER = [Observability.FULL, Observability.PARTIAL, Observability.MINIMAL]


def _bundle(b, level):
    return build_bundle(simulate(b.scenario), level, set(b.spec.partial_resolution_drop))


def test_solver_never_accesses_ground_truth():
    src = inspect.getsource(solver)
    assert not hasattr(solver, "GroundTruthRecord")
    assert "true_entity_id" not in src and "actual_outcome" not in src


def test_wrong_merge_reports_public_final_id():
    full = solve(_bundle(by_id()["S07"], Observability.FULL), "S07-P")
    assert full.status is Label.WRONG_MERGE
    assert full.merge_target_id == "ENT-07" and full.merge_target_observable is True


def test_wrong_merge_target_null_when_unobservable():
    full = solve(_bundle(by_id()["S08"], Observability.FULL), "S08-P")
    assert full.status is Label.WRONG_MERGE
    assert full.merge_target_id is None and full.merge_target_observable is False


def test_removing_evidence_flips_to_undeterminable():
    b = by_id()["S07"]
    assert solve(_bundle(b, Observability.FULL), "S07-P").status is Label.WRONG_MERGE
    assert solve(_bundle(b, Observability.PARTIAL), "S07-P").status is Label.UNDETERMINABLE


def test_determinability_monotone_and_consistent_across_levels():
    for b in build_all():
        probe = b.spec.probe_source_id
        det_prev, sig_prev = True, None
        for lvl in [l for l in ORDER if l in b.spec.levels]:
            ans = solve(_bundle(b, lvl), probe)
            det = ans.status is not Label.UNDETERMINABLE
            assert not (det and not det_prev), f"{b.spec.scenario_id} re-determinable at {lvl}"
            if det:
                sig = (ans.status, ans.final_id, ans.merge_target_id)
                assert sig_prev is None or sig == sig_prev, f"{b.spec.scenario_id} contradictory across levels"
                sig_prev = sig
            det_prev = det


def test_property_redaction_never_invents_contradictory_positive():
    rng = random.Random(7)
    for sid, expect in (("S07", Label.WRONG_MERGE), ("S03", Label.PRESENT_UNDER_OTHER_ID)):
        b = by_id()[sid]
        full = build_bundle(simulate(b.scenario), Observability.FULL)
        base = solve(full, b.spec.probe_source_id)
        for _ in range(60):
            red = copy.deepcopy(full)
            for art in red.artifacts.values():
                if rng.random() < 0.5:
                    art.state, art.complete_scopes, art.rows = ArtifactState.MISSING, [], []
                elif art.rows and rng.random() < 0.5:
                    # valid redaction: dropping rows also drops the completeness guarantee
                    art.rows = [r for r in art.rows if rng.random() < 0.5]
                    art.state, art.complete_scopes = ArtifactState.PARTIAL, []
            ans = solve(red, b.spec.probe_source_id)
            # result may degrade to UNDETERMINABLE, but never a DIFFERENT positive outcome
            assert ans.status in (base.status, Label.UNDETERMINABLE), \
                f"{sid}: redaction produced {ans.status} (base {base.status})"
