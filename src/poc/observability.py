"""Derive the Full / Partial / Minimal views of one simulated execution.

Only evidence availability changes between levels: sources and final_dataset pass through
unchanged, and redaction only removes artifacts, rows, or completeness guarantees.
"""

from __future__ import annotations

import copy

from .labels import PIPELINE_DESCRIPTION
from .models import Artifact, ArtifactState, Observability, ObservableBundle
from .simulator import SimulationResult


def _missing(name: str) -> Artifact:
    return Artifact(name=name, state=ArtifactState.MISSING)


def _to_partial(art: Artifact, drop_source_ids: set[str]) -> Artifact:
    """Drop the completeness guarantee and the rows for the given source ids.

    Args:
        art — Artifact to downgrade.
        drop_source_ids — ``source_id`` values whose rows are removed.

    Returns:
        A ``PARTIAL`` copy with no completeness scopes.
    """
    rows = [r for r in art.rows if r.get("source_id") not in drop_source_ids]
    return Artifact(name=art.name, state=ArtifactState.PARTIAL, rows=rows)


def build_bundle(
    sim: SimulationResult,
    level: Observability,
    partial_resolution_drop: set[str] | None = None,
) -> ObservableBundle:
    """Derive the observable bundle for one observability level.

    Args:
        sim — Simulation whose master artifacts are redacted.
        level — Observability level to derive.
        partial_resolution_drop — ``source_id`` values whose resolution-map rows are removed
            at ``PARTIAL``; None removes none.

    Returns:
        The bundle for ``level``.

    Raises:
        ValueError: If ``level`` is not an :class:`Observability` member.
    """
    master = sim.master_artifacts
    drop = partial_resolution_drop or set()

    if level is Observability.FULL:
        artifacts = {name: copy.deepcopy(a) for name, a in master.items()}

    elif level is Observability.PARTIAL:
        artifacts = {
            "ingestion_manifest": copy.deepcopy(master["ingestion_manifest"]),  # still COMPLETE
            "extraction_log": copy.deepcopy(master["extraction_log"]),          # still COMPLETE
            "normalized_records": copy.deepcopy(master["normalized_records"]),  # still COMPLETE
            "resolution_map": _to_partial(master["resolution_map"], drop),
            "merge_decision_log": _missing("merge_decision_log"),
            "filter_log": copy.deepcopy(master["filter_log"]),                  # still COMPLETE
        }

    elif level is Observability.MINIMAL:
        artifacts = {
            "ingestion_manifest": _to_partial(master["ingestion_manifest"], drop_source_ids=set()),
            "extraction_log": _missing("extraction_log"),
            "normalized_records": _missing("normalized_records"),
            "resolution_map": _missing("resolution_map"),
            "merge_decision_log": _missing("merge_decision_log"),
            "filter_log": _missing("filter_log"),
        }
    else:  # pragma: no cover - e.g. a plain-string level, which fails the `is` checks
        raise ValueError(level)

    return ObservableBundle(
        level=level,
        pipeline_description=PIPELINE_DESCRIPTION,
        sources=copy.deepcopy(sim.sources),
        final_dataset=copy.deepcopy(sim.final_dataset),
        artifacts=artifacts,
    )
