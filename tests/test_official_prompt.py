"""Regression: both prompt renders are byte-identical to the frozen golden prompts.

tests/fixtures/official_prompt/ holds the 33 prompts of the published benchmark run and
tests/fixtures/unguided_prompt/ the 33 unguided-control prompts.
"""
from pathlib import Path

from poc.benchmark import official_scenarios
from poc.models import Observability
from poc.observability import build_bundle
from poc.official_prompt import render_prompt, render_unguided_prompt
from poc.simulator import simulate

FIXTURES = Path(__file__).parent / "fixtures" / "official_prompt"
UNGUIDED_FIXTURES = Path(__file__).parent / "fixtures" / "unguided_prompt"


def _check_against_golden(render, fixtures: Path) -> None:
    checked = 0
    for b in official_scenarios():
        sim = simulate(b.scenario)
        for lvl in b.spec.levels:
            bundle = build_bundle(sim, lvl, set(b.spec.partial_resolution_drop))
            rendered = render(bundle, b.spec.probe_source_id)
            golden = fixtures / f"{b.spec.scenario_id}_{lvl.value}.txt"
            assert golden.exists(), f"missing golden fixture {golden.name}"
            assert rendered == golden.read_text(), f"prompt drift at {golden.name}"
            checked += 1
    assert checked == 33, f"expected 33 prompts, checked {checked}"


def test_all_33_official_prompts_match_golden_byte_for_byte():
    _check_against_golden(render_prompt, FIXTURES)


def test_all_33_unguided_prompts_match_golden_byte_for_byte():
    _check_against_golden(render_unguided_prompt, UNGUIDED_FIXTURES)


def test_official_prompt_contains_the_three_fixed_components():
    b = official_scenarios()[0]
    bundle = build_bundle(simulate(b.scenario), Observability.FULL, set(b.spec.partial_resolution_drop))
    p = render_prompt(bundle, b.spec.probe_source_id)
    assert "# Evidence-grounding rules" in p            # grounding block
    assert "# Identifier reporting" in p                # canonical-vs-final id clause
    assert "evidence_refs" in p and '"artifact"' in p   # structured citations


def test_unguided_prompt_has_no_grounding_rules_or_citations():
    b = official_scenarios()[0]
    bundle = build_bundle(simulate(b.scenario), Observability.FULL, set(b.spec.partial_resolution_drop))
    p = render_unguided_prompt(bundle, b.spec.probe_source_id)
    assert "# Identifier reporting" in p
    assert "# Evidence-grounding rules" not in p
    assert "evidence_refs" not in p
