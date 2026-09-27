"""Official 'Where Did My Data Go?' benchmark — build / run / replay.

Single official prompt (P2, see :mod:`poc.official_prompt`) and single official validator
(:mod:`poc.validators`). No prompt-condition selection.

* ``build``  — offline: run the deterministic validation gates, render the official prompts,
               write the task matrix + cost estimate. ZERO API calls.
* ``run``    — live: execute the matrix against the model (resumable, spending guardrail).
               Explicitly opt-in — requires ``ANTHROPIC_API_KEY``.
* ``replay`` — offline: re-score preserved responses with zero API calls.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from collections import defaultdict
from pathlib import Path

from .baselines import BASELINES
from .config import BenchmarkConfig
from .evaluator import evaluate_one, parse_prediction
from .models import ArtifactState, Label, Observability
from .observability import build_bundle
from .official_prompt import render_official_prompt
from .scenarios import build_all
from .simulator import simulate
from .solver import solve
from . import validators

LEVEL = {"full": Observability.FULL, "partial": Observability.PARTIAL, "minimal": Observability.MINIMAL}
ORDER = [Observability.FULL, Observability.PARTIAL, Observability.MINIMAL]
POSITIVE = {"PRESENT_UNDER_OTHER_ID", "WRONG_MERGE", "INGESTION_LOSS", "EXTRACTION_LOSS", "FILTERED_OUT"}
_P2_OUTPUT_TOKENS_EST = 425  # per-call output estimate for costing (from prior runs)


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def official_scenarios():
    """The primary scenarios (excludes the counterfactual twin, which shares a prompt)."""
    return [b for b in build_all() if b.spec.twin_of is None]


def _bundle(b, level: str):
    return build_bundle(simulate(b.scenario), LEVEL[level], set(b.spec.partial_resolution_drop))


# --------------------------------------------------------------------------------------
# Deterministic validation gates (offline)
# --------------------------------------------------------------------------------------
def _observable_blob(bundle) -> str:
    payload = {
        "sources": bundle.sources, "final_dataset": bundle.final_dataset,
        "artifacts": {n: {"state": a.state.value, "complete_scopes": sorted(a.complete_scopes), "rows": a.rows}
                      for n, a in sorted(bundle.artifacts.items()) if a.state is not ArtifactState.MISSING},
    }
    return json.dumps(payload, sort_keys=True)


def validate() -> tuple[bool, list[str]]:
    """Run all offline gates; return (ok, failures)."""
    failures: list[str] = []
    reg = {b.spec.scenario_id: b for b in build_all()}
    banned = ("true_entity_id", "actual_outcome", "disappearance_reason", "is_wrong_merge", "merged_with")

    for b in build_all():
        s = b.spec
        # reference-generation: solver == authored expected
        for lvl in s.levels:
            ans = solve(_bundle(b, lvl.value), s.probe_source_id)
            exp = s.expected[lvl.value]
            if not (ans.status.value == exp.status and ans.final_id == exp.final_id
                    and ans.merge_target_id == exp.merge_target_id
                    and ans.merge_target_observable == exp.merge_target_observable):
                failures.append(f"reference drift {s.scenario_id}:{lvl.value}")
            # no ground-truth leak
            blob = _bundle(b, lvl.value).model_dump_json()
            leak = next((t for t in banned if t in blob), None)
            if leak:
                failures.append(f"ground-truth leak {s.scenario_id}:{lvl.value} ({leak})")
        # monotone determinability
        det_prev = True
        for lvl in [l for l in ORDER if l in s.levels]:
            det = solve(_bundle(b, lvl.value), s.probe_source_id).status is not Label.UNDETERMINABLE
            if det and not det_prev:
                failures.append(f"non-monotone determinability {s.scenario_id}:{lvl.value}")
            det_prev = det
        # counterfactual twin: byte-identical observable bundle + same reference
        if s.twin_id:
            twin = reg[s.twin_id]
            lvl = s.levels[0].value
            if _observable_blob(_bundle(b, lvl)) != _observable_blob(_bundle(twin, lvl)):
                failures.append(f"counterfactual twin {s.scenario_id}/{s.twin_id} bundles differ")
            if solve(_bundle(b, lvl), s.probe_source_id).status != solve(_bundle(twin, lvl), twin.spec.probe_source_id).status:
                failures.append(f"counterfactual twin {s.scenario_id}/{s.twin_id} references differ")
        # baseline name-trap fires a wrong positive where designed
        if s.name_trap_at_minimal and Observability.MINIMAL in s.levels:
            bundle = _bundle(b, "minimal")
            ref = solve(bundle, s.probe_source_id)
            pred = BASELINES["name_similarity"](bundle, s.probe_source_id)
            if not (ref.status is Label.UNDETERMINABLE and pred.status is Label.PRESENT_UNDER_OTHER_ID):
                failures.append(f"expected name-trap at {s.scenario_id}:minimal")
    return (not failures), failures


# --------------------------------------------------------------------------------------
# Task matrix
# --------------------------------------------------------------------------------------
def build_matrix(cfg: BenchmarkConfig) -> list[dict]:
    matrix = []
    for b in official_scenarios():
        s = b.spec
        for lvl in s.levels:
            prompt = render_official_prompt(_bundle(b, lvl.value), s.probe_source_id)
            matrix.append({
                "scenario_id": s.scenario_id, "group": s.group, "level": lvl.value,
                "probe_source_id": s.probe_source_id, "k": s.k,
                "reference_status": s.expected[lvl.value].status,
                "reference_final_id": s.expected[lvl.value].final_id,
                "reference_merge_target_id": s.expected[lvl.value].merge_target_id,
                "merge_target_observable": s.expected[lvl.value].merge_target_observable,
                "prompt_sha256": _sha(prompt),
                "fingerprint": {"requested_model": cfg.model, "provider": "anthropic",
                                "max_tokens": cfg.max_tokens, "prompt_sha256": _sha(prompt)},
                "est_input_tokens": -(-len(prompt) // 4),
            })
    return matrix


# --------------------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------------------
def cmd_build(cfg: BenchmarkConfig) -> int:
    ok, failures = validate()
    print("== benchmark build (offline; zero API calls) ==")
    print(f"  official prompt: P2   validator: official (integrity + sufficiency + identifier)")
    print(f"  model: {cfg.model}  max_tokens: {cfg.max_tokens}  temperature: omitted")
    if not ok:
        print("VALIDATION FAILED:")
        for f in failures:
            print("   - " + f)
        return 1
    matrix = build_matrix(cfg)
    calls = sum(c["k"] for c in matrix)
    cost = sum(c["k"] * (c["est_input_tokens"] / 1e6 * 2 + _P2_OUTPUT_TOKENS_EST / 1e6 * 10) for c in matrix)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    (cfg.output_dir / "task_matrix.json").write_text(json.dumps(matrix, indent=2))
    (cfg.output_dir / "cost_estimate.json").write_text(json.dumps({
        "model": cfg.model, "scenarios": len({m["scenario_id"] for m in matrix}),
        "task_conditions": len(matrix), "total_calls": calls,
        "k_general": cfg.k_general, "k_calibration": cfg.k_calibration,
        "estimated_cost_usd": round(cost, 2),
    }, indent=2))
    print(f"  validation: PASSED  ({len(official_scenarios())} scenarios, {len(matrix)} task-conditions)")
    print(f"  task matrix + cost estimate -> {cfg.output_dir}")
    print(f"  total calls: {calls}   estimated cost: ${round(cost, 2)}")
    print("  live run:  ANTHROPIC_API_KEY=... uv run --locked python -m poc benchmark run")
    return 0


def cmd_run(cfg: BenchmarkConfig) -> int:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY not set — the live benchmark run is explicitly opt-in.")
    from .provider.anthropic import AnthropicProvider
    from .provider.base import LLMRequest

    matrix = build_matrix(cfg)
    scen = {b.spec.scenario_id: b for b in official_scenarios()}
    provider = AnthropicProvider(model=cfg.model)
    provider.preflight()
    samples_dir = cfg.output_dir / "samples"
    cost = 0.0
    consecutive_failures = 0

    for cell in matrix:
        b = scen[cell["scenario_id"]]
        bundle = _bundle(b, cell["level"])
        ref = solve(bundle, b.spec.probe_source_id)
        prompt = render_official_prompt(bundle, b.spec.probe_source_id)
        d = samples_dir / f"{cell['scenario_id']}_{cell['level']}"
        d.mkdir(parents=True, exist_ok=True)
        for i in range(1, cell["k"] + 1):
            path = d / f"{i:02d}.json"
            if path.exists():
                continue
            if cost >= cfg.budget_usd:
                print(f"!! budget guardrail hit (${cost:.3f} >= ${cfg.budget_usd}); stopping.")
                return 2
            try:
                resp = provider.complete(LLMRequest(model=cfg.model, max_tokens=cfg.max_tokens,
                                                    temperature=0.0, prompt=prompt))
            except Exception as e:  # noqa: BLE001
                consecutive_failures += 1
                print(f"  FAIL {cell['scenario_id']}:{cell['level']}#{i}: {e}")
                if consecutive_failures >= cfg.max_consecutive_failures:
                    print("!! too many consecutive failures; stopping.")
                    return 3
                time.sleep(2)
                continue
            consecutive_failures = 0
            cost += (resp.input_tokens or 0) / 1e6 * 2 + (resp.output_tokens or 0) / 1e6 * 10
            _score_and_persist(path, cell, ref, bundle, resp)
            print(f"  {cell['scenario_id']}:{cell['level']:<7}#{i} ref={ref.status.value:<22} "
                  f"pred={_pred_status(path)}  [${cost:.3f}]")
    print(f"DONE. real_cost=${cost:.3f}  samples -> {samples_dir}")
    return 0


def _score_and_persist(path: Path, cell: dict, ref, bundle, resp) -> None:
    rec = evaluate_one(f"{cell['scenario_id']}:{cell['level']}", ref, resp.text)
    refs, validity = None, None
    if not rec.malformed:
        pred = parse_prediction(resp.text).prediction
        if pred is not None:
            try:
                obj = json.loads(resp.text[resp.text.find("{"): resp.text.rfind("}") + 1])
                refs = obj.get("evidence_refs") if isinstance(obj.get("evidence_refs"), list) else []
            except Exception:  # noqa: BLE001
                refs = []
            validity = _validity(bundle, ref, pred, refs)
    path.write_text(json.dumps({
        "scenario_id": cell["scenario_id"], "level": cell["level"], "sample": int(path.stem),
        "probe_source_id": cell["probe_source_id"],
        "reference": {"status": ref.status.value, "final_id": ref.final_id,
                      "merge_target_id": ref.merge_target_id,
                      "merge_target_observable": ref.merge_target_observable},
        "fingerprint": cell["fingerprint"], "request_params": resp.request_params,
        "prediction": rec.model_dump(mode="json"), "evidence_refs": refs, "chain_validity": validity,
        "input_tokens": resp.input_tokens, "output_tokens": resp.output_tokens,
        "text": resp.text, "raw": resp.raw,
    }, indent=2, default=str))


def _validity(bundle, ref, pred, refs) -> dict:
    return {
        "citation_integrity": validators.citation_integrity(bundle, refs),
        "evidence_sufficiency": validators.evidence_sufficiency(bundle, ref, pred.status, refs),
        "status_evidence_validity": validators.status_evidence_validity(bundle, ref, pred.status, refs),
        "identifier_evidence_validity": validators.identifier_evidence_validity(
            bundle, ref, pred.status, pred.final_id, pred.merge_target_id, refs),
        "full_evidence_validity": validators.full_evidence_validity(
            bundle, ref, pred.status, pred.final_id, pred.merge_target_id, refs),
    }


def _pred_status(path: Path) -> str:
    return json.loads(path.read_text())["prediction"]["predicted_status"]


def cmd_replay(cfg: BenchmarkConfig) -> int:
    samples_dir = cfg.output_dir / "samples"
    files = sorted(samples_dir.rglob("*.json"))
    if not files:
        raise SystemExit(f"no saved samples under {samples_dir}; run `poc benchmark run` first")
    scen = {b.spec.scenario_id: b for b in official_scenarios()}
    per_sc = defaultdict(list)
    n = mal = 0
    for f in files:
        s = json.loads(f.read_text())
        b = scen[s["scenario_id"]]
        bundle = _bundle(b, s["level"])
        ref = solve(bundle, b.spec.probe_source_id)
        rec = evaluate_one(f"{s['scenario_id']}:{s['level']}", ref, s["text"])  # re-score, zero API
        per_sc[s["scenario_id"]].append(rec.model_dump(mode="json"))
        n += 1
        mal += rec.malformed
    # scenario-level macro accuracy
    macro = []
    for sc, recs in per_sc.items():
        macro.append(sum(r["status_correct"] for r in recs) / len(recs))
    summary = {"samples": n, "malformed": mal, "scenarios": len(per_sc),
               "status_accuracy_macro": round(sum(macro) / len(macro), 4) if macro else 0.0}
    (cfg.output_dir / "replay_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"replay (zero API): samples={n} malformed={mal} "
          f"scenario-macro status accuracy={summary['status_accuracy_macro']}")
    return 0
