"""Official benchmark commands: offline ``build``/``replay`` (zero API calls) and live, opt-in ``run``."""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
from collections import defaultdict
from pathlib import Path

from . import validators
from .baselines import BASELINES
from .config import BenchmarkConfig
from .evaluator import EvalRecord, evaluate_one, parse_prediction
from .models import ArtifactState, Label, Observability, ObservableBundle, ReferenceAnswer
from .observability import build_bundle
from .official_prompt import OFFICIAL_PROMPT_NAME, render_prompt
from .provider.base import LLMRequest, RawResponse
from .scenarios import BuiltScenario, build_all
from .simulator import simulate
from .solver import solve

ORDER = [Observability.FULL, Observability.PARTIAL, Observability.MINIMAL]
_INPUT_USD_PER_MTOK = 2.0  # claude-sonnet-5 pricing
_OUTPUT_USD_PER_MTOK = 10.0
_OUTPUT_TOKENS_EST = 425  # per-call output estimate for costing (from prior runs)


def _usd(input_tokens: float, output_tokens: float) -> float:
    """Return the USD cost of one call at claude-sonnet-5 prices."""
    return input_tokens / 1e6 * _INPUT_USD_PER_MTOK + output_tokens / 1e6 * _OUTPUT_USD_PER_MTOK


def official_scenarios() -> list[BuiltScenario]:
    """Return the primary scenarios, excluding the counterfactual twin (it shares S11's prompt)."""
    return [b for b in build_all() if b.spec.twin_of is None]


def scenario_bundle(b: BuiltScenario, level: str) -> ObservableBundle:
    """Return the scenario's observable bundle at ``level`` ("full", "partial" or "minimal")."""
    return build_bundle(simulate(b.scenario), Observability(level), set(b.spec.partial_resolution_drop))


# --------------------------------------------------------------------------------------
# Deterministic validation gates (offline)
# --------------------------------------------------------------------------------------
def _observable_blob(bundle: ObservableBundle) -> str:
    """Serialize the bundle's observable content canonically, skipping MISSING artifacts.

    Args:
        bundle — Observable evidence for one level.

    Returns:
        A sort-keyed JSON string; equal strings mean observationally identical bundles.
    """
    payload = {
        "sources": bundle.sources, "final_dataset": bundle.final_dataset,
        "artifacts": {n: {"state": a.state.value, "complete_scopes": sorted(a.complete_scopes), "rows": a.rows}
                      for n, a in sorted(bundle.artifacts.items()) if a.state is not ArtifactState.MISSING},
    }
    return json.dumps(payload, sort_keys=True)


def validate() -> tuple[bool, list[str]]:
    """Run the offline gates: reference drift, ground-truth leaks, monotonicity, twin, name trap.

    Returns:
        ``(ok, failures)`` where ``failures`` lists one message per failed gate.
    """
    failures: list[str] = []
    reg = {b.spec.scenario_id: b for b in build_all()}
    banned = ("true_entity_id", "actual_outcome", "disappearance_reason", "is_wrong_merge", "merged_with")

    for b in reg.values():
        s = b.spec
        statuses: dict[Observability, Label] = {}
        for lvl in s.levels:
            bundle = scenario_bundle(b, lvl.value)
            ans = solve(bundle, s.probe_source_id)
            statuses[lvl] = ans.status
            exp = s.expected[lvl.value]
            if not (ans.status.value == exp.status and ans.final_id == exp.final_id
                    and ans.merge_target_id == exp.merge_target_id
                    and ans.merge_target_observable == exp.merge_target_observable):
                failures.append(f"reference drift {s.scenario_id}:{lvl.value}")
            blob = bundle.model_dump_json()
            leak = next((t for t in banned if t in blob), None)
            if leak:
                failures.append(f"ground-truth leak {s.scenario_id}:{lvl.value} ({leak})")
        # Removing evidence may lose determinability but never gain it.
        det_prev = True
        for lvl in ORDER:
            if lvl not in statuses:
                continue
            det = statuses[lvl] is not Label.UNDETERMINABLE
            if det and not det_prev:
                failures.append(f"non-monotone determinability {s.scenario_id}:{lvl.value}")
            det_prev = det
        if s.twin_id:
            twin = reg[s.twin_id]
            lvl = s.levels[0]
            twin_bundle = scenario_bundle(twin, lvl.value)
            if _observable_blob(scenario_bundle(b, lvl.value)) != _observable_blob(twin_bundle):
                failures.append(f"counterfactual twin {s.scenario_id}/{s.twin_id} bundles differ")
            if statuses[lvl] != solve(twin_bundle, twin.spec.probe_source_id).status:
                failures.append(f"counterfactual twin {s.scenario_id}/{s.twin_id} references differ")
        if s.name_trap_at_minimal and Observability.MINIMAL in s.levels:
            pred = BASELINES["name_similarity"](scenario_bundle(b, "minimal"), s.probe_source_id)
            if not (statuses[Observability.MINIMAL] is Label.UNDETERMINABLE
                    and pred.status is Label.PRESENT_UNDER_OTHER_ID):
                failures.append(f"expected name-trap at {s.scenario_id}:minimal")
    return (not failures), failures


# --------------------------------------------------------------------------------------
# Task matrix
# --------------------------------------------------------------------------------------
def build_matrix(cfg: BenchmarkConfig) -> list[dict]:
    """Build one task cell per (official scenario, observability level).

    Args:
        cfg — Execution parameters; ``model`` and ``max_tokens`` go into each cell's fingerprint.

    Returns:
        Cells with the scenario, level, k, reference answer, prompt hash and input-token estimate.
    """
    matrix = []
    for b in official_scenarios():
        s = b.spec
        for lvl in s.levels:
            prompt = render_prompt(scenario_bundle(b, lvl.value), s.probe_source_id)
            prompt_sha256 = hashlib.sha256(prompt.encode()).hexdigest()
            exp = s.expected[lvl.value]
            matrix.append({
                "scenario_id": s.scenario_id, "group": s.group, "level": lvl.value,
                "probe_source_id": s.probe_source_id, "k": s.k,
                "reference_status": exp.status,
                "reference_final_id": exp.final_id,
                "reference_merge_target_id": exp.merge_target_id,
                "merge_target_observable": exp.merge_target_observable,
                "prompt_sha256": prompt_sha256,
                "fingerprint": {"requested_model": cfg.model, "provider": "anthropic",
                                "max_tokens": cfg.max_tokens, "prompt_sha256": prompt_sha256},
                "est_input_tokens": math.ceil(len(prompt) / 4),  # ~4 characters per token
            })
    return matrix


# --------------------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------------------
def cmd_build(cfg: BenchmarkConfig) -> int:
    """Run the validation gates, then write the task matrix and cost estimate. Makes no API calls.

    Args:
        cfg — Execution parameters; outputs go to ``cfg.output_dir``.

    Returns:
        Exit code: 0 on success, 1 if any validation gate fails.
    """
    ok, failures = validate()
    print("== benchmark build (offline; zero API calls) ==")
    print(f"  official prompt: {OFFICIAL_PROMPT_NAME}   "
          f"validator: official (integrity + sufficiency + identifier)")
    print(f"  model: {cfg.model}  max_tokens: {cfg.max_tokens}  temperature: omitted")
    if not ok:
        print("VALIDATION FAILED:")
        for f in failures:
            print("   - " + f)
        return 1
    matrix = build_matrix(cfg)
    n_scenarios = len({m["scenario_id"] for m in matrix})
    calls = sum(c["k"] for c in matrix)
    est_cost = round(sum(c["k"] * _usd(c["est_input_tokens"], _OUTPUT_TOKENS_EST) for c in matrix), 2)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    (cfg.output_dir / "task_matrix.json").write_text(json.dumps(matrix, indent=2))
    (cfg.output_dir / "cost_estimate.json").write_text(json.dumps({
        "model": cfg.model, "scenarios": n_scenarios,
        "task_conditions": len(matrix), "total_calls": calls,
        "estimated_cost_usd": est_cost,
    }, indent=2))
    print(f"  validation: PASSED  ({n_scenarios} scenarios, {len(matrix)} task-conditions)")
    print(f"  task matrix + cost estimate -> {cfg.output_dir}")
    print(f"  total calls: {calls}   estimated cost: ${est_cost}")
    print("  live run:  ANTHROPIC_API_KEY=... uv run --locked python -m poc benchmark run")
    return 0


def cmd_run(cfg: BenchmarkConfig) -> int:
    """Execute the task matrix against the live model, resuming past samples already saved.

    Args:
        cfg — Execution parameters, including the ``budget_usd`` guardrail and failure limit.

    Returns:
        Exit code: 0 when done, 2 if the budget guardrail stopped the run, 3 after too many
        consecutive API failures.

    Raises:
        SystemExit: If ``ANTHROPIC_API_KEY`` is not set.
    """
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY not set — the live benchmark run is explicitly opt-in.")
    # Imported here so build/replay (and Kaggle) never need the anthropic SDK.
    from .provider.anthropic import AnthropicProvider

    matrix = build_matrix(cfg)
    scen = {b.spec.scenario_id: b for b in official_scenarios()}
    provider = AnthropicProvider(model=cfg.model)
    provider.preflight()
    samples_dir = cfg.output_dir / "samples"
    cost = 0.0
    consecutive_failures = 0

    for cell in matrix:
        b = scen[cell["scenario_id"]]
        bundle = scenario_bundle(b, cell["level"])
        ref = solve(bundle, b.spec.probe_source_id)
        prompt = render_prompt(bundle, b.spec.probe_source_id)
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
            cost += _usd(resp.input_tokens or 0, resp.output_tokens or 0)
            rec = _score_and_persist(path, cell, ref, bundle, resp)
            pred_status = rec.predicted_status.value if rec.predicted_status else None
            print(f"  {cell['scenario_id']}:{cell['level']:<7}#{i} ref={ref.status.value:<22} "
                  f"pred={pred_status}  [${cost:.3f}]")
    print(f"DONE. real_cost=${cost:.3f}  samples -> {samples_dir}")
    return 0


def _score_and_persist(path: Path, cell: dict, ref: ReferenceAnswer, bundle: ObservableBundle,
                       resp: RawResponse) -> EvalRecord:
    """Score one response and save it with its reference, provenance and evidence-chain validity.

    Args:
        path — Sample file to write; its stem is the sample number.
        cell — Task-matrix cell the sample belongs to.
        ref — Reference answer for the cell.
        bundle — Observable evidence the model was shown.
        resp — The model response.

    Returns:
        The evaluation record for the response.
    """
    rec = evaluate_one(f"{cell['scenario_id']}:{cell['level']}", ref, resp.text)
    refs, validity = None, None
    if not rec.malformed:
        parsed = parse_prediction(resp.text)
        refs = parsed.evidence_refs
        validity = validators.chain_validity(bundle, ref, parsed.prediction, refs)
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
    return rec


def cmd_replay(cfg: BenchmarkConfig) -> int:
    """Re-score saved samples and write the scenario-macro status accuracy. Makes no API calls.

    Args:
        cfg — Execution parameters; samples are read from and the summary written to ``cfg.output_dir``.

    Returns:
        Exit code 0.

    Raises:
        SystemExit: If there are no saved samples.
    """
    samples_dir = cfg.output_dir / "samples"
    files = sorted(samples_dir.rglob("*.json"))
    if not files:
        raise SystemExit(f"no saved samples under {samples_dir}; run `poc benchmark run` first")
    scen = {b.spec.scenario_id: b for b in official_scenarios()}
    status_correct: dict[str, list[bool]] = defaultdict(list)
    malformed = 0
    for f in files:
        sample = json.loads(f.read_text())
        b = scen[sample["scenario_id"]]
        ref = solve(scenario_bundle(b, sample["level"]), b.spec.probe_source_id)
        rec = evaluate_one(f"{sample['scenario_id']}:{sample['level']}", ref, sample["text"])
        status_correct[sample["scenario_id"]].append(rec.status_correct)
        malformed += rec.malformed
    per_scenario = [sum(v) / len(v) for v in status_correct.values()]
    summary = {"samples": len(files), "malformed": malformed, "scenarios": len(status_correct),
               "status_accuracy_macro": round(sum(per_scenario) / len(per_scenario), 4)}
    (cfg.output_dir / "replay_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"replay (zero API): samples={len(files)} malformed={malformed} "
          f"scenario-macro status accuracy={summary['status_accuracy_macro']}")
    return 0
