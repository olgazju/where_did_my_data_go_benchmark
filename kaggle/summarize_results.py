"""Summarize downloaded Kaggle runs into markdown tables for the write-up (offline, no API).

Usage (after ``kaggle b t download where-did-my-data-go -o ./results``):

    PYTHONPATH=src:. python kaggle/summarize_results.py ./results

Per-case records are collected from every JSON file, grouped by track and model,
de-duplicated by (scenario, level, rep) and re-aggregated with ``kagglebench.scoring.aggregate``.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from kagglebench.scoring import TRACKS, aggregate, score_response

_CASE_KEYS = {"scenario_id", "level", "rep", "reference_status", "evidence_supported"}


def _iter_cases(obj: Any) -> Iterator[dict]:
    # Sub-run results are nested inside proto ``dictResult`` structs at varying depths.
    if isinstance(obj, dict):
        if _CASE_KEYS <= obj.keys():
            yield obj
            return
        for v in obj.values():
            yield from _iter_cases(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _iter_cases(v)


def _model_and_run(path: Path, root: Path) -> tuple[str, int]:
    """Return the model slug and numeric run id from ``<root>/<task>/<version>/<model>/<run_id>/…``."""
    parts = path.relative_to(root).parts
    if len(parts) > 4 and parts[3].isdigit():
        return parts[2], int(parts[3])
    return (parts[2] if len(parts) > 3 else parts[0]), 0


def _run_file_cost(data: dict) -> float:
    """Return the USD the Model Proxy reported for the calls recorded in one run file."""
    nanos = 0
    for conv in data.get("conversations", []):
        for req in conv.get("requests") or []:
            m = req.get("metrics") or {}
            nanos += int(m.get("inputTokensCostNanodollars", 0)) + int(m.get("outputTokensCostNanodollars", 0))
    return nanos / 1e9


def load(root: Path) -> dict[str, dict[str, list[dict]]]:
    """Load every per-case record under a downloaded results directory.

    Args:
        root — Output directory of ``kaggle b t download``.

    Returns:
        ``{track: {model: [case, ...]}}``, one record per (scenario, level, rep), taken from each
        model's latest run only so a re-run never mixes with an earlier partial one.
    """
    found: dict[tuple[str, str], dict[int, dict[tuple, dict]]] = defaultdict(lambda: defaultdict(dict))
    for f in root.rglob("*.json"):
        try:
            data = json.loads(f.read_text())
        except (OSError, ValueError):
            continue
        cases = list(_iter_cases(data))
        if len(cases) != 1:  # the parent run file repeats every case; sub-run files hold one each
            continue
        cost = _run_file_cost(data)
        for c in cases:
            # Re-score the saved raw answer so every run is judged by the current scorer.
            track = c.get("track", "official")
            c = score_response(c["scenario_id"], c["level"], int(c["rep"]), c["response_text"], track)
            c["cost_usd"] = cost  # one sub-run file holds one case
            model, run_id = _model_and_run(f, root)
            found[(track, model)][run_id][(c["scenario_id"], c["level"], c["rep"])] = c
    out: dict[str, dict[str, list[dict]]] = defaultdict(dict)
    for (track, model), runs in found.items():
        out[track][model] = list(runs[max(runs)].values())
    return out


def _per_1k(cases: list[dict]) -> str:
    """Return the mean Model Proxy cost of 1,000 investigations, formatted in USD."""
    return f"${1000 * sum(c.get('cost_usd', 0.0) for c in cases) / len(cases):.2f}" if cases else "—"


def _fmt(x: float | None) -> str:
    return "—" if x is None else f"{x:.3f}"


def _track_report(track: str, models: dict[str, list[dict]], metrics: dict[str, dict]) -> None:
    """Print one track's leaderboard, per-scenario matrix and fallen-for calibration traps.

    Args:
        track — Key of ``TRACKS``.
        models — Per-case records by model.
        metrics — :func:`aggregate` output by model.
    """
    key = TRACKS[track]["primary_case_key"]
    order = sorted(metrics, key=lambda m: -metrics[m]["primary_score"])

    print(f"\n## Track `{track}` — leaderboard ({TRACKS[track]['primary']})\n")
    print("| Model | Primary (±95% CI) | Status acc. | Guess rate on UND | Correct-positive "
          "| Transition acc. | Citation integrity | Cases | Malformed |")
    print("|---|---|---|---|---|---|---|---|---|")
    for m in order:
        x = metrics[m]
        print(f"| {m} | {x['primary_score']:.3f} ± {x['primary_score_ci95']:.3f} "
              f"| {_fmt(x['status_accuracy_scenario_macro'])} | {_fmt(x['guess_rate_on_undeterminable'])} "
              f"| {_fmt(x['correct_positive_rate'])} "
              f"| {_fmt(x['paired_evidence_transition_accuracy_scenario_macro'])} "
              f"| {_fmt(x['citation_integrity_rate'])} | {x['n_cases']} | {x['n_malformed']} |")

    scenarios = sorted({c["scenario_id"] for cs in models.values() for c in cs})
    print(f"\n### `{track}` — {key} by scenario\n")
    print("| Model | " + " | ".join(scenarios) + " |")
    print("|---|" + "---|" * len(scenarios))
    for m in order:
        cells = []
        for s in scenarios:
            cs = [c for c in models[m] if c["scenario_id"] == s]
            cells.append(f"{sum(bool(c[key]) for c in cs) / len(cs):.2f}" if cs else "—")
        print(f"| {m} | " + " | ".join(cells) + " |")

    print(f"\n### `{track}` — traps fallen for (positive claim where the reference is UNDETERMINABLE)\n")
    for m in order:
        hits = defaultdict(int)
        for c in models[m]:
            if c["reference_status"] == "UNDETERMINABLE" and c["predicted_status"] not in (None, "UNDETERMINABLE"):
                hits[f"{c['scenario_id']}:{c['level']} → {c['predicted_status']}"] += 1
        listed = ", ".join(f"{k} ×{v}" for k, v in sorted(hits.items())) or "none"
        print(f"- **{m}**: {listed}")


def main(root: Path) -> None:
    """Print the official-vs-unguided comparison, then each track's report.

    Args:
        root — Output directory of ``kaggle b t download``.
    """
    tracks = load(root)
    if not tracks:
        sys.exit(f"no per-case records found under {root}")
    metrics = {t: {m: aggregate(cs, t) for m, cs in models.items()} for t, models in tracks.items()}

    if {"official", "unguided"} <= metrics.keys():
        both = sorted(metrics["official"].keys() & metrics["unguided"].keys(),
                      key=lambda m: -metrics["unguided"][m]["primary_score"])
        # Compare like with like: full-answer accuracy exists on both tracks; the official
        # evidence-supported score also requires citations, so it gets its own column.
        print("## With rules vs. on instinct\n")
        print("| Model | Correct (instinct) | Correct (rules) | Δ | Guess rate on UND: instinct → rules "
              "| Evidence-supported (rules) | $ per 1k investigations (rules) |")
        print("|---|---|---|---|---|---|---|")
        for m in both:
            o, u = metrics["official"][m], metrics["unguided"][m]
            key = "full_accuracy_scenario_macro"
            print(f"| {m} | {u[key]:.3f} | {o[key]:.3f} | {o[key] - u[key]:+.3f} "
                  f"| {_fmt(u['guess_rate_on_undeterminable'])} → {_fmt(o['guess_rate_on_undeterminable'])} "
                  f"| {o['primary_score']:.3f} ± {o['primary_score_ci95']:.3f} "
                  f"| {_per_1k(tracks['official'][m])} |")

    for track in [t for t in TRACKS if t in tracks]:
        _track_report(track, tracks[track], metrics[track])


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "results"))
