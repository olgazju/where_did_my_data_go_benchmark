"""Offline tests for the Kaggle integration layer (zero API calls).

Covers the pure grid/scoring/aggregation in ``kagglebench`` and statically checks the Kaggle
entrypoints (importing one would start a live run).
"""
from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

import pytest

# Make the repo-root `kagglebench` package importable (poc is pip-installed).
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from kagglebench.scoring import (  # noqa: E402
    PRIMARY_METRIC,
    aggregate,
    build_grid,
    case_prompt,
    run_grid,
    score_response,
)
from poc.benchmark import official_scenarios  # noqa: E402

_PROBE_RE = re.compile(r"Investigate source record: (\S+)")


def _source_id_from_prompt(prompt: str) -> str:
    m = _PROBE_RE.search(prompt)
    assert m, "probe source id not found in prompt"
    return m.group(1)


def _answer(prompt: str, status: str, final_id=None, merge_target_id=None, refs=None) -> str:
    return json.dumps({
        "source_id": _source_id_from_prompt(prompt),
        "status": status,
        "final_id": final_id,
        "merge_target_id": merge_target_id,
        "evidence_refs": refs or [],
    })


# --------------------------------------------------------------------------------------
# Grid
# --------------------------------------------------------------------------------------
def test_grid_has_107_rows_over_33_conditions():
    grid = build_grid()
    conditions = {(r["scenario_id"], r["level"]) for r in grid}
    scenarios = {r["scenario_id"] for r in grid}
    assert len(conditions) == 33
    assert len(scenarios) == 15
    assert len(grid) == 107  # k=3 general + k=5 calibration

    # rep counts match each scenario's declared k for every one of its levels.
    from collections import Counter
    per_cond = Counter((r["scenario_id"], r["level"]) for r in grid)
    for b in official_scenarios():
        for lvl in b.spec.levels:
            assert per_cond[(b.spec.scenario_id, lvl.value)] == b.spec.k


def test_case_prompt_matches_official_render():
    from poc.models import Observability
    from poc.observability import build_bundle
    from poc.official_prompt import render_prompt
    from poc.simulator import simulate

    b = official_scenarios()[0]
    lvl = b.spec.levels[0].value
    bundle = build_bundle(simulate(b.scenario),
                          Observability(lvl), set(b.spec.partial_resolution_drop))
    assert case_prompt(b.spec.scenario_id, lvl) == render_prompt(bundle, b.spec.probe_source_id)


# --------------------------------------------------------------------------------------
# Per-case scoring
# --------------------------------------------------------------------------------------
def _name_trap_minimal():
    """A calibration scenario whose reference at MINIMAL is UNDETERMINABLE."""
    from poc.models import Observability
    for b in official_scenarios():
        s = b.spec
        if s.name_trap_at_minimal and Observability.MINIMAL in s.levels:
            if s.expected["minimal"].status == "UNDETERMINABLE":
                return s.scenario_id
    pytest.skip("no name-trap-at-minimal scenario found")


def test_correct_abstention_is_evidence_supported():
    sid = _name_trap_minimal()
    prompt = case_prompt(sid, "minimal")
    rec = score_response(sid, "minimal", 1, _answer(prompt, "UNDETERMINABLE"))
    assert rec["reference_status"] == "UNDETERMINABLE"
    assert rec["predicted_status"] == "UNDETERMINABLE"
    assert rec["status_correct"] is True
    assert rec["evidence_supported"] is True
    assert rec["unsupported_positive"] is False
    assert rec["malformed"] is False


def test_unsupported_positive_fires_on_name_trap():
    sid = _name_trap_minimal()
    prompt = case_prompt(sid, "minimal")
    # A committed positive with no citations: the classic unsupported guess.
    rec = score_response(sid, "minimal", 1, _answer(prompt, "PRESENT_UNDER_OTHER_ID", final_id="ENT-XX"))
    assert rec["reference_status"] == "UNDETERMINABLE"
    assert rec["predicted_status"] == "PRESENT_UNDER_OTHER_ID"
    assert rec["status_correct"] is False
    assert rec["evidence_supported"] is False
    assert rec["unsupported_positive"] is True
    assert rec["citation_integrity"] is False


def test_malformed_answer_is_tracked_and_not_a_positive():
    b = official_scenarios()[0]
    sid, lvl = b.spec.scenario_id, b.spec.levels[0].value
    rec = score_response(sid, lvl, 1, "this is not json")
    assert rec["malformed"] is True
    assert rec["predicted_status"] is None
    assert rec["evidence_supported"] is False
    assert rec["unsupported_positive"] is False


def test_abstaining_on_a_determinable_loss_is_not_evidence_supported():
    # Regression: the validator accepts any UNDETERMINABLE without citations, so the
    # per-case flag must also require the answer to be correct.
    from poc.models import Observability
    for b in official_scenarios():
        if b.spec.expected.get("full") and b.spec.expected["full"].status in {
            "INGESTION_LOSS", "EXTRACTION_LOSS", "FILTERED_OUT"
        } and Observability.FULL in b.spec.levels:
            sid = b.spec.scenario_id
            break
    else:
        pytest.skip("no determinable loss scenario at full")
    rec = score_response(sid, "full", 1, _answer(case_prompt(sid, "full"), "UNDETERMINABLE"))
    assert rec["status_correct"] is False
    assert rec["evidence_supported"] is False


# --------------------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------------------
def _undet_model(prompt: str) -> str:
    return _answer(prompt, "UNDETERMINABLE")


def test_always_abstain_full_grid_metrics():
    cases, metrics = run_grid(_undet_model)
    assert len(cases) == 107
    assert metrics[PRIMARY_METRIC] == metrics["primary_score"]
    # Abstaining everywhere: no unsupported positives, but the over-abstention guard fires.
    assert metrics["unsupported_positive_rate_scenario_macro"] == 0.0
    assert metrics["correct_abstention_rate"] == 1.0
    assert metrics["correct_positive_rate"] == 0.0
    assert metrics["guess_rate_on_undeterminable"] == 0.0
    # Some scenarios are fully UNDETERMINABLE (calibration), none are impossible.
    assert 0.0 < metrics["primary_score"] < 1.0
    # Abstention earns credit only where abstaining is the right answer.
    assert metrics["primary_score"] == metrics["status_accuracy_scenario_macro"]
    assert metrics["primary_score_ci95"] > 0.0
    assert metrics["n_cases"] == 107 and metrics["n_scenarios"] == 15
    assert metrics["n_malformed"] == 0


def test_aggregate_formulas_on_synthetic_cases():
    # Two scenarios, one transition each, hand-built to check every reduction.
    def case(sc, lvl, rep, ref, pred, supported, unsupported, status_ok):
        return {
            "scenario_id": sc, "group": "B", "level": lvl, "rep": rep,
            "reference_status": ref, "predicted_status": pred,
            "status_correct": status_ok, "full_correct": supported, "malformed": False,
            "evidence_supported": supported, "unsupported_positive": unsupported,
            "citation_integrity": supported, "evidence_sufficiency": supported,
            "identifier_evidence_validity": supported,
        }
    cases = [
        # SC1: full=PRESENT (got it, grounded), minimal=UNDET (abstained correctly) -> transition
        case("SC1", "full", 1, "PRESENT_UNDER_OTHER_ID", "PRESENT_UNDER_OTHER_ID", True, False, True),
        case("SC1", "minimal", 1, "UNDETERMINABLE", "UNDETERMINABLE", True, False, True),
        # SC2: full=WRONG_MERGE (wrong: abstained), minimal=UNDET (correct) -> transition, half correct
        case("SC2", "full", 1, "WRONG_MERGE", "UNDETERMINABLE", False, False, False),
        case("SC2", "minimal", 1, "UNDETERMINABLE", "UNDETERMINABLE", True, False, True),
    ]
    m = aggregate(cases)
    # evidence_supported: SC1 -> (1+1)/2=1.0 ; SC2 -> (0+1)/2=0.5 ; macro=0.75
    assert m[PRIMARY_METRIC] == 0.75
    # paired transition: SC1 pair correct (both) =1.0 ; SC2 pair (full wrong)=0.0 ; macro=0.5
    assert m["paired_evidence_transition_accuracy_scenario_macro"] == 0.5
    # correct abstention over the 2 UNDET-ref cases -> both abstained -> 1.0
    assert m["correct_abstention_rate"] == 1.0
    # correct-positive over the 2 positive-ref cases (SC1 full ok, SC2 full wrong) -> 0.5
    assert m["correct_positive_rate"] == 0.5


# --------------------------------------------------------------------------------------
# Unguided track
# --------------------------------------------------------------------------------------
def test_unguided_case_prompt_has_no_citation_contract():
    b = official_scenarios()[0]
    sid, lvl = b.spec.scenario_id, b.spec.levels[0].value
    assert "evidence_refs" in case_prompt(sid, lvl)
    assert "evidence_refs" not in case_prompt(sid, lvl, "unguided")


def test_unguided_primary_is_full_accuracy_and_citation_metrics_are_blank():
    cases, metrics = run_grid(_undet_model, track="unguided")
    assert {c["track"] for c in cases} == {"unguided"}
    assert metrics["primary_metric"] == "full_accuracy_scenario_macro"
    assert metrics["primary_score"] == metrics["full_accuracy_scenario_macro"]
    for key in ("citation_integrity_rate", "unsupported_positive_rate_scenario_macro", PRIMARY_METRIC):
        assert metrics[key] is None


def test_unguided_rewards_correct_answers_without_citations():
    from kagglebench.scoring import _SCENARIOS
    from poc.benchmark import scenario_bundle
    from poc.solver import solve

    def oracle(prompt: str) -> str:
        sid = _source_id_from_prompt(prompt)
        b = next(b for b in _SCENARIOS.values() if b.spec.probe_source_id == sid)
        lvl = re.search(r"# Observability level\n(\w+)", prompt).group(1).lower()
        ref = solve(scenario_bundle(b, lvl), sid)
        return _answer(prompt, ref.status.value, ref.final_id, ref.merge_target_id)

    _, unguided = run_grid(oracle, track="unguided")
    _, official = run_grid(oracle)
    assert unguided["primary_score"] == 1.0
    assert official["primary_score"] < 1.0  # same answers, but uncited positives don't count


# --------------------------------------------------------------------------------------
# Kaggle entrypoints (static: importing one would start a live run)
# --------------------------------------------------------------------------------------
_ENTRYPOINTS = {
    "task.py": ("where-did-my-data-go", "where_did_my_data_go"),
    "task_unguided.py": ("where-did-my-data-go-unguided", "where_did_my_data_go_unguided"),
}


def _tree(name: str) -> ast.Module:
    return ast.parse((_REPO_ROOT / "kaggle" / name).read_text())


@pytest.mark.parametrize("filename", sorted(_ENTRYPOINTS))
def test_kaggle_entrypoint_declares_and_runs_its_task(filename):
    task_slug, func_name = _ENTRYPOINTS[filename]
    tree = _tree(filename)
    # Mirror `kaggle b t push` validation: only string-literal names count.
    literal_names = {
        kw.value.value
        for node in tree.body if isinstance(node, ast.FunctionDef)
        for dec in node.decorator_list if isinstance(dec, ast.Call)
        for kw in dec.keywords if kw.arg == "name" and isinstance(kw.value, ast.Constant)
    }
    assert task_slug in literal_names
    # Kaggle silently records nothing unless the module itself calls `<task>.run(...)`.
    module_calls = [
        n.value for n in tree.body
        if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
        and isinstance(n.value.func, ast.Attribute) and n.value.func.attr == "run"
    ]
    assert [c.func.value.id for c in module_calls] == [func_name]


@pytest.mark.parametrize("filename", sorted(_ENTRYPOINTS))
def test_kaggle_entrypoint_keeps_real_annotations(filename):
    # kbench infers each task's result type from its return annotation object;
    # postponed (string) annotations make it raise at decoration time.
    future = [n for n in _tree(filename).body
              if isinstance(n, ast.ImportFrom) and n.module == "__future__"]
    assert not future


@pytest.mark.parametrize("filename", sorted(_ENTRYPOINTS))
def test_kaggle_task_descriptions_fit_the_255_char_limit(filename):
    # Kaggle rejects tasks whose description (explicit, or the docstring fallback) exceeds 255.
    tree = _tree(filename)
    consts = {n.targets[0].id: n.value.value for n in tree.body
              if isinstance(n, ast.Assign) and isinstance(n.value, ast.Constant)}
    for fn in (n for n in tree.body if isinstance(n, ast.FunctionDef) and n.decorator_list):
        kw = {k.arg: k.value for d in fn.decorator_list if isinstance(d, ast.Call) for k in d.keywords}
        desc = kw.get("description")
        if desc is None:
            text = ast.get_docstring(fn) or ""
        elif isinstance(desc, ast.Name):
            text = consts[desc.id]
        else:
            text = ast.literal_eval(desc)
        assert len(text) <= 255, f"{fn.name}: {len(text)} chars"


def test_kaggle_entrypoints_share_identical_code():
    def bodies(name: str) -> dict[str, str]:
        out = {}
        for n in _tree(name).body:
            if isinstance(n, ast.FunctionDef):
                body = n.body[1:] if ast.get_docstring(n) is not None else n.body
                key = "main" if n.name.startswith("where_did") else n.name
                out[key] = ast.dump(ast.Module(body=body, type_ignores=[]))
        return out

    assert bodies("task.py") == bodies("task_unguided.py")
