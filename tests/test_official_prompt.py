"""Regression: the official P2 prompt render is byte-identical to the frozen golden prompts.

The 33 golden prompts in tests/fixtures/official_p2/ are the exact prompts used in the
published benchmark run. This guards the official prompt wording against any drift.
"""
from pathlib import Path

from poc.models import Observability
from poc.observability import build_bundle
from poc.official_prompt import render_official_prompt
from poc.scenarios import build_all
from poc.simulator import simulate

FIXTURES = Path(__file__).parent / "fixtures" / "official_p2"
LEVEL = {"full": Observability.FULL, "partial": Observability.PARTIAL, "minimal": Observability.MINIMAL}


def _primary():
    return [b for b in build_all() if b.spec.twin_of is None]


def test_all_33_official_prompts_match_golden_byte_for_byte():
    checked = 0
    for b in _primary():
        sim = simulate(b.scenario)
        for lvl in b.spec.levels:
            bundle = build_bundle(sim, LEVEL[lvl.value], set(b.spec.partial_resolution_drop))
            rendered = render_official_prompt(bundle, b.spec.probe_source_id)
            golden = FIXTURES / f"{b.spec.scenario_id}_{lvl.value}.txt"
            assert golden.exists(), f"missing golden fixture {golden.name}"
            assert rendered == golden.read_text(), f"prompt drift at {golden.name}"
            checked += 1
    assert checked == 33, f"expected 33 official prompts, checked {checked}"


def test_official_prompt_contains_the_three_fixed_components():
    b = _primary()[0]
    bundle = build_bundle(simulate(b.scenario), Observability.FULL, set(b.spec.partial_resolution_drop))
    p = render_official_prompt(bundle, b.spec.probe_source_id)
    assert "# Evidence-grounding rules" in p            # grounding block
    assert "# Identifier reporting" in p                # canonical-vs-final id clause
    assert "evidence_refs" in p and '"artifact"' in p   # structured citations
