"""Grid construction, per-case scoring and metric aggregation over the unchanged ``poc`` benchmark.

Prompts, the reference solver, the evaluator and the evidence-chain validator all come from
``poc``; this module only expands conditions by k, packages one model answer into a per-case
record, and aggregates records into the official metrics. It never calls a model.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from itertools import pairwise
from math import sqrt
from statistics import fmean, stdev
from typing import Any

from poc import validators
from poc.benchmark import official_scenarios, scenario_bundle
from poc.evaluator import evaluate_one, parse_prediction
from poc.models import Label
from poc.official_prompt import (
    OFFICIAL_PROMPT_NAME,
    UNGUIDED_PROMPT_NAME,
    render_prompt,
    render_unguided_prompt,
)
from poc.solver import solve

UNDETERMINABLE = Label.UNDETERMINABLE.value
POSITIVE = {label.value for label in Label} - {UNDETERMINABLE}
_LEVEL_ORDER = ("full", "partial", "minimal")

#: The primary leaderboard score of the official track (higher is better).
PRIMARY_METRIC = "evidence_supported_accuracy_scenario_macro"

#: Two tracks over the same grid. ``official`` gives the model the grounding rules and requires
#: citations, so it scores evidence-supported accuracy. ``unguided`` gives neither, so it scores
#: plain full-answer accuracy: does the model keep evidence discipline without being told?
TRACKS: dict[str, dict[str, Any]] = {
    "official": {"render": render_prompt, "prompt": OFFICIAL_PROMPT_NAME,
                 "primary": PRIMARY_METRIC, "primary_case_key": "evidence_supported"},
    "unguided": {"render": render_unguided_prompt, "prompt": UNGUIDED_PROMPT_NAME,
                 "primary": "full_accuracy_scenario_macro", "primary_case_key": "full_correct"},
}
# Citation-based metrics are meaningless when citations were never requested.
_CITATION_METRICS = (PRIMARY_METRIC, "unsupported_positive_rate_scenario_macro",
                     "citation_integrity_rate", "evidence_sufficiency_rate",
                     "identifier_evidence_validity_rate")

# The 15 official scenarios; the twin S11B shares S11's prompt and is excluded, as in poc.benchmark.
_SCENARIOS = {b.spec.scenario_id: b for b in official_scenarios()}


# --------------------------------------------------------------------------------------
# Grid + prompt
# --------------------------------------------------------------------------------------
def build_grid() -> list[dict[str, Any]]:
    """Expand the 33 observable conditions by their k into one row per sample.

    Returns:
        Rows of ``{"scenario_id", "level", "rep"}``; ``rep`` (1..k) keeps the repeated samples
        of a condition distinct so Kaggle runs each as its own sub-run.
    """
    rows: list[dict[str, Any]] = []
    for b in _SCENARIOS.values():
        s = b.spec
        for lvl in s.levels:
            for rep in range(1, s.k + 1):
                rows.append({"scenario_id": s.scenario_id, "level": lvl.value, "rep": rep})
    return rows


def case_prompt(scenario_id: str, level: str, track: str = "official") -> str:
    """Render the track's prompt for one (scenario, observability level).

    Args:
        scenario_id — Official scenario id, e.g. ``"S01"``.
        level — ``"full"``, ``"partial"`` or ``"minimal"``.
        track — Key of :data:`TRACKS`.

    Returns:
        The prompt text sent to the model.
    """
    b = _SCENARIOS[scenario_id]
    return TRACKS[track]["render"](scenario_bundle(b, level), b.spec.probe_source_id)


# --------------------------------------------------------------------------------------
# Per-case scoring
# --------------------------------------------------------------------------------------
_FALSE_VALIDITY = {
    "citation_integrity": False,
    "evidence_sufficiency": False,
    "status_evidence_validity": False,
    "identifier_evidence_validity": False,
    "full_evidence_validity": False,
}


def score_response(scenario_id: str, level: str, rep: int, response_text: str,
                   track: str = "official") -> dict[str, Any]:
    """Score one model answer deterministically against the solver's reference.

    Args:
        scenario_id — Official scenario id.
        level — Observability level of the prompt that was answered.
        rep — Sample index (1..k) within the condition.
        response_text — The model's raw answer; ``""`` scores as malformed.
        track — Key of :data:`TRACKS`, stamped on the record.

    Returns:
        The per-case record: reference, prediction, correctness, evidence-chain validity,
        derived ``evidence_supported`` / ``unsupported_positive`` flags and the raw answer.
    """
    b = _SCENARIOS[scenario_id]
    bundle = scenario_bundle(b, level)
    ref = solve(bundle, b.spec.probe_source_id)
    rec = evaluate_one(f"{scenario_id}:{level}", ref, response_text)

    refs: list = []
    validity = _FALSE_VALIDITY
    predicted_final_id = predicted_merge_target_id = None
    if not rec.malformed:
        parsed = parse_prediction(response_text)
        predicted_final_id = parsed.prediction.final_id
        predicted_merge_target_id = parsed.prediction.merge_target_id
        refs = parsed.evidence_refs
        validity = validators.chain_validity(bundle, ref, parsed.prediction, refs)

    predicted_status = rec.predicted_status.value if rec.predicted_status is not None else None
    full_evidence_validity = validity["full_evidence_validity"]
    unsupported_positive = predicted_status in POSITIVE and not full_evidence_validity
    # Correctness is required explicitly: the validator accepts any UNDETERMINABLE without
    # citations, even where the reference is a determinable loss.
    evidence_supported = rec.full_correct and full_evidence_validity

    return {
        "track": track,
        "scenario_id": scenario_id,
        "group": b.spec.group,
        "level": level,
        "rep": rep,
        "probe_source_id": b.spec.probe_source_id,
        # reference (evidence-supported ground truth)
        "reference_status": ref.status.value,
        "reference_final_id": ref.final_id,
        "reference_merge_target_id": ref.merge_target_id,
        "merge_target_observable": ref.merge_target_observable,
        # prediction
        "predicted_status": predicted_status,
        "predicted_final_id": predicted_final_id,
        "predicted_merge_target_id": predicted_merge_target_id,
        "malformed": rec.malformed,
        "status_correct": rec.status_correct,
        "full_correct": rec.full_correct,
        # evidence chain
        "evidence_refs": refs,
        **validity,
        # derived per-case signals
        "evidence_supported": evidence_supported,
        "unsupported_positive": unsupported_positive,
        # raw answer, preserved for audit / replay
        "response_text": response_text,
    }


# --------------------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------------------
def _rate(items: list[bool]) -> float | None:
    return round(fmean(1.0 if x else 0.0 for x in items), 4) if items else None


def _per_scenario(cases: list[dict], key: str) -> list[float]:
    by_scen: dict[str, list[dict]] = defaultdict(list)
    for c in cases:
        by_scen[c["scenario_id"]].append(c)
    return [fmean(1.0 if c[key] else 0.0 for c in cs) for cs in by_scen.values()]


def _scenario_macro(cases: list[dict], key: str) -> float:
    per = _per_scenario(cases, key)
    return round(fmean(per), 4) if per else 0.0


def _scenario_macro_ci95(cases: list[dict], key: str) -> float:
    """Return the half-width of a normal-approx 95% CI over scenarios (the independent unit)."""
    per = _per_scenario(cases, key)
    return round(1.96 * stdev(per) / sqrt(len(per)), 4) if len(per) > 1 else 0.0


def _paired_transition_macro(cases: list[dict]) -> float | None:
    """Compute paired evidence-transition accuracy, macro-averaged over transition scenarios.

    A transition is an adjacent level pair (full→partial→minimal) whose reference status
    changes; samples are paired by ``rep`` and a pair counts only if the status is correct at
    both endpoints.

    Args:
        cases — Per-case records from :func:`score_response`.

    Returns:
        The macro accuracy, or ``None`` if no scenario has a transition.
    """
    by_scen: dict[str, list[dict]] = defaultdict(list)
    for c in cases:
        by_scen[c["scenario_id"]].append(c)

    scen_scores: list[float] = []
    for cs in by_scen.values():
        by_level: dict[str, dict[int, dict]] = defaultdict(dict)
        ref_of: dict[str, str] = {}
        for c in cs:
            by_level[c["level"]][c["rep"]] = c
            ref_of[c["level"]] = c["reference_status"]
        present = [lvl for lvl in _LEVEL_ORDER if lvl in by_level]
        pair_scores: list[float] = []
        for a, b in pairwise(present):
            if ref_of[a] == ref_of[b]:
                continue  # not a transition
            reps = sorted(set(by_level[a]) & set(by_level[b]))
            if not reps:
                continue
            pair_scores.append(
                fmean(
                    1.0 if (by_level[a][r]["status_correct"] and by_level[b][r]["status_correct"]) else 0.0
                    for r in reps
                )
            )
        if pair_scores:
            scen_scores.append(fmean(pair_scores))
    return round(fmean(scen_scores), 4) if scen_scores else None


def aggregate(cases: list[dict], track: str = "official") -> dict[str, Any]:
    """Aggregate per-case records into the track's metric block.

    Headline views are macro-averaged per scenario because the observability variants of one
    probe are correlated; auxiliary rates are micro over their applicable cases.

    Args:
        cases — Per-case records from :func:`score_response`.
        track — Key of :data:`TRACKS`; ``unguided`` blanks the citation-based metrics.

    Returns:
        All metrics plus ``primary_score`` and its scenario-level ``primary_score_ci95``.
    """
    undet = [c for c in cases if c["reference_status"] == UNDETERMINABLE]
    positive_ref = [c for c in cases if c["reference_status"] in POSITIVE]
    committed = [c for c in cases if not c["malformed"] and c["predicted_status"] != UNDETERMINABLE]

    metrics: dict[str, Any] = {
        # PRIMARY (scenario-macro, higher is better)
        PRIMARY_METRIC: _scenario_macro(cases, "evidence_supported"),
        # accuracy (scenario-macro)
        "status_accuracy_scenario_macro": _scenario_macro(cases, "status_correct"),
        "full_accuracy_scenario_macro": _scenario_macro(cases, "full_correct"),
        # headline calibration metric (scenario-macro; lower is better)
        "unsupported_positive_rate_scenario_macro": _scenario_macro(cases, "unsupported_positive"),
        # calibration / over-abstention guards (micro over the applicable cases)
        "correct_abstention_rate": _rate([c["predicted_status"] == UNDETERMINABLE for c in undet]),
        # how often the model commits to ANY positive label when the evidence licenses none
        "guess_rate_on_undeterminable": _rate([c["predicted_status"] in POSITIVE for c in undet]),
        "correct_positive_rate": _rate([c["status_correct"] for c in positive_ref]),
        # paired transitions (macro over transition scenarios)
        "paired_evidence_transition_accuracy_scenario_macro": _paired_transition_macro(cases),
        # evidence-chain validity (micro over committed, non-UNDETERMINABLE answers)
        "citation_integrity_rate": _rate([c["citation_integrity"] for c in committed]),
        "evidence_sufficiency_rate": _rate([c["evidence_sufficiency"] for c in committed]),
        "identifier_evidence_validity_rate": _rate([c["identifier_evidence_validity"] for c in committed]),
        # denominators / provenance
        "n_cases": len(cases),
        "n_scenarios": len({c["scenario_id"] for c in cases}),
        "n_malformed": sum(1 for c in cases if c["malformed"]),
        "n_undeterminable_ref": len(undet),
        "n_positive_ref": len(positive_ref),
        "n_committed": len(committed),
        "track": track,
        "prompt": TRACKS[track]["prompt"],
        "primary_metric": TRACKS[track]["primary"],
    }
    if track == "unguided":
        metrics.update(dict.fromkeys(_CITATION_METRICS))
    metrics["primary_score"] = metrics[TRACKS[track]["primary"]]
    metrics["primary_score_ci95"] = _scenario_macro_ci95(cases, TRACKS[track]["primary_case_key"])
    return metrics


def run_grid(prompt_fn: Callable[[str], str],
             track: str = "official") -> tuple[list[dict], dict[str, Any]]:
    """Score the whole grid offline with a stand-in model (tests and local dry-runs).

    Args:
        prompt_fn — Maps a prompt to the model's raw answer.
        track — Key of :data:`TRACKS`.

    Returns:
        ``(cases, metrics)`` as produced by :func:`score_response` and :func:`aggregate`.
    """
    cases = [
        score_response(row["scenario_id"], row["level"], row["rep"],
                       prompt_fn(case_prompt(row["scenario_id"], row["level"], track)), track)
        for row in build_grid()
    ]
    return cases, aggregate(cases, track)
