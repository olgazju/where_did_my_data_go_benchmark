"""Minimal deterministic baselines.

These exist to demonstrate *experimental validity*: that some tasks genuinely require
lineage evidence and cannot be solved by naive strategies. Each baseline is intentionally
weak and uses only a subset of the observable evidence:

* ``name_similarity``   — matches the source name to a final entity name (names only).
* ``strong_id``         — matches a namespaced strong id shared with a final record.
* ``source_final_only`` — strong id, then name similarity; NO intermediate lineage.

None of the baselines read the intermediate artifacts, so their answer is identical at
every observability level — which is exactly why they cannot reproduce the level-dependent
reference answers.
"""

from __future__ import annotations

import difflib
import re
from collections.abc import Callable

from pydantic import BaseModel

from .evaluator import evaluate_one
from .models import Label, ObservableBundle, Observability, Prediction, StrongId
from .observability import build_bundle
from .simulator import SimulationResult
from .solver import solve

NAME_MATCH_THRESHOLD = 0.80


def _normalize(name: str) -> str:
    n = name.lower()
    n = re.sub(r"[.,]", " ", n)
    n = re.sub(r"\b(incorporated|inc|ltd|limited|llc|plc|partners|group|co|corp)\b", " ", n)
    return re.sub(r"\s+", " ", n).strip()


def _display_name(src: dict) -> str:
    for key in ("legal_name", "business_name", "org_mention"):
        if src.get(key):
            return src[key]
    return ""


def _source_strong_ids(src: dict) -> list[StrongId]:
    t = src.get("source_type")
    if t == "registry":
        return [StrongId(namespace=f"registry:{src['jurisdiction']}", value=src["registration_number"])]
    if t == "directory":
        return [StrongId(namespace="duns", value=src["duns"])] if src.get("duns") else []
    return []


def _present(source_id: str, final_id: str) -> Prediction:
    return Prediction(source_id=source_id, status=Label.PRESENT_UNDER_OTHER_ID, final_id=final_id)


def _undeterminable(source_id: str) -> Prediction:
    return Prediction(source_id=source_id, status=Label.UNDETERMINABLE)


# --------------------------------------------------------------------------------------
# The three baselines
# --------------------------------------------------------------------------------------
def name_similarity(bundle: ObservableBundle, source_id: str) -> Prediction:
    src = bundle.source(source_id)
    if not src:
        return _undeterminable(source_id)
    target = _normalize(_display_name(src))
    best_id, best_score = None, 0.0
    for row in bundle.final_dataset:
        score = difflib.SequenceMatcher(None, target, _normalize(row["canonical_name"])).ratio()
        if score > best_score:
            best_id, best_score = row["final_id"], score
    if best_id is not None and best_score >= NAME_MATCH_THRESHOLD:
        return _present(source_id, best_id)
    return _undeterminable(source_id)


def strong_id(bundle: ObservableBundle, source_id: str) -> Prediction:
    src = bundle.source(source_id)
    if not src:
        return _undeterminable(source_id)
    index: dict[str, str] = {}
    for row in bundle.final_dataset:
        for s in row.get("strong_ids", []):
            index[f"{s['namespace']}/{s['value']}"] = row["final_id"]
    for sid in _source_strong_ids(src):
        if sid.key() in index:
            return _present(source_id, index[sid.key()])
    return _undeterminable(source_id)


def source_final_only(bundle: ObservableBundle, source_id: str) -> Prediction:
    """Best effort using ONLY source records + final dataset (no lineage)."""
    by_id = strong_id(bundle, source_id)
    if by_id.status is Label.PRESENT_UNDER_OTHER_ID:
        return by_id
    return name_similarity(bundle, source_id)


BASELINES: dict[str, Callable[[ObservableBundle, str], Prediction]] = {
    "name_similarity": name_similarity,
    "strong_id": strong_id,
    "source_final_only": source_final_only,
}


# --------------------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------------------
class BaselineReport(BaseModel):
    # baseline name -> {"solved": int, "total": int, "solved_task_ids": [...]}
    per_baseline: dict[str, dict]
    # determinable tasks that NONE of the IMPLEMENTED lineage-free baselines solve.
    # NOTE: this does not prove no other lineage-free strategy could solve them — it is a
    # statement about the baselines actually implemented here.
    unsolved_by_baselines_task_ids: list[str]
    reference_determinable: int
    total_tasks: int

    def render(self) -> str:
        lines = ["Deterministic baseline report", "=" * 60]
        lines.append(f"tasks: {self.total_tasks}   reference-determinable: {self.reference_determinable}")
        lines.append("(a 'determinable' task has a reference answer other than UNDETERMINABLE)")
        lines.append("")
        for name, s in self.per_baseline.items():
            lines.append(
                f"  {name:<18} correct {s['solved']}/{s['total']} tasks "
                f"(of which {s['solved_determinable']}/{self.reference_determinable} determinable)"
            )
        lines.append("")
        lines.append("Determinable tasks NOT solved by the implemented lineage-free baselines")
        lines.append("(name-similarity, strong-id, source+final-only). This does NOT prove no")
        lines.append("other lineage-free strategy could solve them; it only reflects these baselines: "
                     f"{len(self.unsolved_by_baselines_task_ids)}")
        for tid in self.unsolved_by_baselines_task_ids:
            lines.append(f"    - {tid}")
        lines.append("=" * 60)
        return "\n".join(lines)


def run_baselines(
    sim: SimulationResult,
    levels: list[Observability],
    source_ids: list[str],
) -> BaselineReport:
    bundles = {lvl: build_bundle(sim, lvl) for lvl in levels}

    per_baseline: dict[str, dict] = {
        n: {"solved": 0, "solved_determinable": 0, "total": 0, "solved_task_ids": []} for n in BASELINES
    }
    unsolved_by_baselines: list[str] = []
    determinable = 0
    total = 0

    for lvl in levels:
        bundle = bundles[lvl]
        for sid in source_ids:
            total += 1
            task_id = f"{sid}:{lvl.value}"
            reference = solve(bundle, sid)
            ref_determinable = reference.status is not Label.UNDETERMINABLE
            if ref_determinable:
                determinable += 1

            any_solved = False
            for name, fn in BASELINES.items():
                pred = fn(bundle, sid)
                rec = evaluate_one(task_id, reference, _prediction_json(pred))
                per_baseline[name]["total"] += 1
                if rec.full_correct:
                    per_baseline[name]["solved"] += 1
                    per_baseline[name]["solved_task_ids"].append(task_id)
                    if ref_determinable:
                        per_baseline[name]["solved_determinable"] += 1
                        any_solved = True

            if ref_determinable and not any_solved:
                unsolved_by_baselines.append(task_id)

    return BaselineReport(
        per_baseline=per_baseline,
        unsolved_by_baselines_task_ids=unsolved_by_baselines,
        reference_determinable=determinable,
        total_tasks=total,
    )


def _prediction_json(pred: Prediction) -> str:
    import json

    return json.dumps({
        "source_id": pred.source_id,
        "status": pred.status.value,
        "final_id": pred.final_id,
        "merge_target_id": pred.merge_target_id,
    })
