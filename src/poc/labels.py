"""Label definitions and pipeline description, rendered verbatim into the official prompt."""

from __future__ import annotations

from .models import Label

LABEL_DEFINITIONS: dict[Label, str] = {
    Label.INGESTION_LOSS: "The source record was not successfully ingested into the pipeline.",
    Label.EXTRACTION_LOSS: "Ingestion succeeded, but structured entity extraction failed for the record.",
    Label.WRONG_MERGE: "The record was incorrectly merged with a distinct entity (identity contradiction).",
    Label.FILTERED_OUT: "The entity was excluded from the final dataset by a business rule.",
    Label.PRESENT_UNDER_OTHER_ID: "The entity exists in the final dataset under a different identifier.",
    Label.UNDETERMINABLE: "The available evidence is insufficient to establish a unique outcome.",
}

PIPELINE_DESCRIPTION = (
    "A multi-source pipeline ingests business records from three heterogeneous source "
    "types (business registry records, business directory records, and news articles) "
    "and processes them through six stages: (1) source ingestion, (2) structured entity "
    "extraction, (3) normalization, (4) entity resolution / deduplication, (5) business "
    "filtering, and (6) emission of a final canonical entity dataset. Records may be lost "
    "at any stage, deduplicated into a shared canonical id, incorrectly merged with a "
    "distinct entity, or excluded by a business rule."
)
