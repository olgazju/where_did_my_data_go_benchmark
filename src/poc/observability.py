"""STEP 4 — derive the Full / Partial / Minimal views of ONE execution.

Only *evidence availability* changes between levels. The entity population, the actual
processing history, and the distractor entities are identical everywhere (sources and
final_dataset are passed through unchanged). Redaction never adds information: it only
removes artifacts or downgrades a completeness guarantee (COMPLETE -> PARTIAL/MISSING)
and only removes rows.
"""

from __future__ import annotations

import copy

from .labels import PIPELINE_DESCRIPTION
from .models import Artifact, ArtifactState, ObservableBundle, Observability
from .simulator import SimulationResult

# At Partial observability, these resolution-map rows are removed (the lineage links
# that a wrong-merge investigation and a mapping-only PRESENT case each depend on).
PARTIAL_RESOLUTION_DROP: set[str] = {"SRC-104", "SRC-106"}


def _downgrade_missing(name: str) -> Artifact:
    return Artifact(name=name, state=ArtifactState.MISSING, complete_scopes=[], rows=[])


def _to_partial(art: Artifact, drop_source_ids: set[str]) -> Artifact:
    rows = [r for r in art.rows if r.get("source_id") not in drop_source_ids]
    return Artifact(name=art.name, state=ArtifactState.PARTIAL, complete_scopes=[], rows=rows)


def _relax_to_partial(art: Artifact) -> Artifact:
    """Keep all rows but drop the completeness guarantee (no new info introduced)."""
    return Artifact(name=art.name, state=ArtifactState.PARTIAL, complete_scopes=[], rows=list(art.rows))


def build_bundle(
    sim: SimulationResult,
    level: Observability,
    partial_resolution_drop: set[str] | None = None,
) -> ObservableBundle:
    """Derive one observability view.

    ``partial_resolution_drop`` names the ``source_id`` resolution-map rows removed at the
    PARTIAL level. It defaults to the POC scenario's :data:`PARTIAL_RESOLUTION_DROP`; v0.2
    scenarios pass their own per-scenario policy (see the design spec). This only ever
    *removes* rows, so it cannot introduce information at a lower level.
    """
    master = sim.master_artifacts
    drop = PARTIAL_RESOLUTION_DROP if partial_resolution_drop is None else set(partial_resolution_drop)

    if level is Observability.FULL:
        artifacts = {name: copy.deepcopy(a) for name, a in master.items()}

    elif level is Observability.PARTIAL:
        artifacts = {
            "ingestion_manifest": copy.deepcopy(master["ingestion_manifest"]),  # still COMPLETE
            "extraction_log": copy.deepcopy(master["extraction_log"]),          # still COMPLETE
            "normalized_records": copy.deepcopy(master["normalized_records"]),  # still COMPLETE
            "resolution_map": _to_partial(master["resolution_map"], drop),
            "merge_decision_log": _downgrade_missing("merge_decision_log"),
            "filter_log": copy.deepcopy(master["filter_log"]),                  # still COMPLETE
        }

    elif level is Observability.MINIMAL:
        artifacts = {
            "ingestion_manifest": _relax_to_partial(master["ingestion_manifest"]),
            "extraction_log": _downgrade_missing("extraction_log"),
            "normalized_records": _downgrade_missing("normalized_records"),
            "resolution_map": _downgrade_missing("resolution_map"),
            "merge_decision_log": _downgrade_missing("merge_decision_log"),
            "filter_log": _downgrade_missing("filter_log"),
        }
    else:  # pragma: no cover - exhaustive
        raise ValueError(level)

    return ObservableBundle(
        level=level,
        pipeline_description=PIPELINE_DESCRIPTION,
        sources=copy.deepcopy(sim.sources),
        final_dataset=copy.deepcopy(sim.final_dataset),
        artifacts=artifacts,
    )


def build_all_bundles(
    sim: SimulationResult,
    levels: list[Observability],
    partial_resolution_drop: set[str] | None = None,
) -> dict[Observability, ObservableBundle]:
    return {lvl: build_bundle(sim, lvl, partial_resolution_drop) for lvl in levels}
