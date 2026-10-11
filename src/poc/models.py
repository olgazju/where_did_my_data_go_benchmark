"""Core data model.

Two worlds live here, and keeping them apart is the whole point of the benchmark:

* **Hidden truth** — :class:`GroundTruthRecord` and everything the simulator knows,
  including ``true_entity_id`` and the *actual* processing outcome. This is NEVER
  exposed to the solver or the LLM.
* **Observable evidence** — :class:`ObservableBundle`: the raw source records, the
  final dataset, and the intermediate :class:`Artifact` objects that survive at a
  given observability level. The solver and the LLM see exactly this and nothing more.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------------------
# Enums
# --------------------------------------------------------------------------------------
class Label(str, Enum):
    """The six classification labels a model may return for an investigated record."""

    INGESTION_LOSS = "INGESTION_LOSS"
    EXTRACTION_LOSS = "EXTRACTION_LOSS"
    WRONG_MERGE = "WRONG_MERGE"
    FILTERED_OUT = "FILTERED_OUT"
    PRESENT_UNDER_OTHER_ID = "PRESENT_UNDER_OTHER_ID"
    UNDETERMINABLE = "UNDETERMINABLE"


class Observability(str, Enum):
    """The three evidence-availability variants of the same execution."""

    FULL = "full"
    PARTIAL = "partial"
    MINIMAL = "minimal"


class ArtifactState(str, Enum):
    """Three-valued availability of an intermediate artifact.

    ``COMPLETE`` carries a completeness guarantee (see :attr:`Artifact.complete_scopes`)
    that licenses reasoning from *absence*. ``PARTIAL`` means some rows may be missing
    and absence proves nothing. ``MISSING`` means the artifact is not available at all.
    """

    COMPLETE = "COMPLETE"
    PARTIAL = "PARTIAL"
    MISSING = "MISSING"


# --------------------------------------------------------------------------------------
# Strong identifiers
# --------------------------------------------------------------------------------------
class StrongId(BaseModel):
    """A namespaced identifier with a uniqueness guarantee.

    Within a single ``namespace`` a ``value`` denotes *at most one* real-world entity.
    Two records carrying different values in the *same* namespace therefore cannot be
    the same entity — this is what makes an incorrect merge provable from observable
    evidence alone, without any access to ground truth. Name similarity is never a
    strong id.
    """

    namespace: str  # e.g. "registry:US-DE", "duns"
    value: str

    def key(self) -> str:
        """Return the ``namespace/value`` string form, e.g. ``registry:US-DE/4830012``."""
        return f"{self.namespace}/{self.value}"


def source_strong_ids(src: dict) -> list[StrongId]:
    """Return the strong ids carried by a serialized source record.

    Args:
        src — A source record as it appears in an observable bundle.

    Returns:
        The registry number for registry sources, the DUNS for directory sources that have one,
        and nothing for news sources.
    """
    if src.get("source_type") == "registry":
        return [StrongId(namespace=f"registry:{src['jurisdiction']}", value=src["registration_number"])]
    if src.get("source_type") == "directory" and src.get("duns"):
        return [StrongId(namespace="duns", value=src["duns"])]
    return []


def final_strong_keys(row: dict | None) -> set[str]:
    """Return the strong-id keys of a serialized final-dataset row.

    Args:
        row — A ``final_dataset`` row, or None.

    Returns:
        The row's ``namespace/value`` keys; empty for None.
    """
    if not row:
        return set()
    return {f"{s['namespace']}/{s['value']}" for s in row.get("strong_ids", [])}


# --------------------------------------------------------------------------------------
# Source records (three heterogeneous schemas)
# --------------------------------------------------------------------------------------
class RegistrySource(BaseModel):
    """Business registry record — legal name + registration number + jurisdiction."""

    source_id: str
    source_type: Literal["registry"] = "registry"
    batch: str
    legal_name: str
    registration_number: str
    jurisdiction: str  # e.g. "US-DE"

    def strong_ids(self) -> list[StrongId]:
        return [StrongId(namespace=f"registry:{self.jurisdiction}", value=self.registration_number)]

    def display_name(self) -> str:
        return self.legal_name


class DirectorySource(BaseModel):
    """Business directory record — trade name + optional DUNS + coarse attributes."""

    source_id: str
    source_type: Literal["directory"] = "directory"
    batch: str
    business_name: str
    duns: str | None = None
    category: str
    city: str

    def strong_ids(self) -> list[StrongId]:
        return [StrongId(namespace="duns", value=self.duns)] if self.duns else []

    def display_name(self) -> str:
        return self.business_name


class NewsSource(BaseModel):
    """News article — an organization *mention* with no structured identifier."""

    source_id: str
    source_type: Literal["news"] = "news"
    batch: str
    headline: str
    body: str
    org_mention: str
    published: str

    def strong_ids(self) -> list[StrongId]:
        return []

    def display_name(self) -> str:
        return self.org_mention


SourceRecord = RegistrySource | DirectorySource | NewsSource


# --------------------------------------------------------------------------------------
# Hidden ground truth
# --------------------------------------------------------------------------------------
class GroundTruthRecord(BaseModel):
    """Complete internal history of one source record. NEVER exposed to solver/LLM."""

    source_id: str
    true_entity_id: str

    ingested: bool
    extracted: bool
    normalized: bool
    canonical_id: str | None = None  # canonical assigned at resolution
    merged_with: list[str] = Field(default_factory=list)  # other source_ids sharing canonical
    is_wrong_merge: bool = False  # canonical mixes >1 distinct true entity
    filtered_out: bool = False
    filter_rule: str | None = None
    final_id: str | None = None  # ENT-... if present in final dataset

    actual_outcome: Label
    disappearance_reason: str | None = None  # human-readable explanation of the outcome


# --------------------------------------------------------------------------------------
# Observable artifacts
# --------------------------------------------------------------------------------------
class Artifact(BaseModel):
    """One intermediate artifact as seen at a particular observability level.

    ``rows`` are plain dicts (already redacted for the level). ``complete_scopes``
    lists the scopes for which the artifact is *guaranteed complete*; absence of a
    row is evidence only for a scope named here. It is empty unless ``state`` is
    ``COMPLETE``.
    """

    name: str
    state: ArtifactState
    complete_scopes: list[str] = Field(default_factory=list)
    rows: list[dict] = Field(default_factory=list)

    def is_complete_for(self, scope: str) -> bool:
        """Return whether absence of a row in this artifact is evidence for ``scope``.

        Args:
            scope — Completeness scope, e.g. a batch id or ``"extraction"``.

        Returns:
            True if the artifact is ``COMPLETE`` and guaranteed complete for ``scope``.
        """
        return self.state is ArtifactState.COMPLETE and scope in self.complete_scopes


class ObservableBundle(BaseModel):
    """Everything a solver/LLM may look at for one observability level."""

    level: Observability
    pipeline_description: str
    sources: list[dict]  # heterogeneous source payloads (constant across levels)
    final_dataset: list[dict]  # final canonical records (constant across levels)
    artifacts: dict[str, Artifact]  # keyed by artifact name

    def artifact(self, name: str) -> Artifact:
        """Look up an artifact by name.

        Args:
            name — Artifact name, e.g. ``"resolution_map"``.

        Returns:
            The artifact, or a ``MISSING`` placeholder if the bundle does not contain it.
        """
        return self.artifacts.get(name, Artifact(name=name, state=ArtifactState.MISSING))

    def source(self, source_id: str) -> dict | None:
        """Look up a source payload by id.

        Args:
            source_id — Id of the source record.

        Returns:
            The source payload, or None if no source has that id.
        """
        for s in self.sources:
            if s.get("source_id") == source_id:
                return s
        return None


# --------------------------------------------------------------------------------------
# Reference answer (produced by the deterministic solver) and predictions
# --------------------------------------------------------------------------------------
class ReferenceAnswer(BaseModel):
    """The evidence-supported answer the solver derives for one (record, level)."""

    source_id: str
    level: Observability
    status: Label
    final_id: str | None = None
    merge_target_id: str | None = None
    merge_target_observable: bool = False  # is merge_target_id establishable from evidence?
    evidence_refs: list[str] = Field(default_factory=list)


class Prediction(BaseModel):
    """A parsed model answer (the scored subset of fields)."""

    source_id: str
    status: Label
    final_id: str | None = None
    merge_target_id: str | None = None
