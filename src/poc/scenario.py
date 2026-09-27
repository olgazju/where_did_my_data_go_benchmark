"""Scenario data model.

Declarative building blocks for a benchmark scenario: the hidden fate of each source
record, the records themselves, and the final canonical entities. The simulator
(:mod:`poc.simulator`) turns a :class:`Scenario` into ground truth + observable artifacts,
so there is exactly one source of truth. The concrete benchmark scenarios live in
:mod:`poc.scenarios`.
"""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel

from .models import SourceRecord, StrongId


class Fate(str, Enum):
    """The hidden fate of a source record in an execution."""

    INGESTION_LOSS = "ingestion_loss"
    EXTRACTION_LOSS = "extraction_loss"
    RESOLVED_PRESENT = "resolved_present"  # correctly present in final under a canonical
    WRONG_MERGE_ABSORBED = "wrong_merge_absorbed"  # identity lost into another entity's canonical
    FILTERED_OUT = "filtered_out"  # resolved, then removed by a business rule


class ScenarioRecord(BaseModel):
    source: SourceRecord
    true_entity_id: str
    fate: Fate
    canonical_id: str | None = None  # assigned at resolution (all fates except the two losses)
    filter_rule: str | None = None  # only for FILTERED_OUT


class FinalEntity(BaseModel):
    """A canonical record emitted in the final dataset."""

    final_id: str
    canonical_id: str
    canonical_name: str
    strong_ids: list[StrongId]
    member_source_ids: list[str]  # sources that legitimately constitute this entity


class Scenario(BaseModel):
    records: list[ScenarioRecord]
    final_entities: list[FinalEntity]
    # Ingestion is organised into batches by source type; the ingestion manifest is
    # guaranteed complete for these batches at full/partial observability.
    ingestion_batches: list[str] = ["B1", "B2", "B3"]
