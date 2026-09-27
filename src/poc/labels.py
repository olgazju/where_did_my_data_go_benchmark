"""Classification-label semantics, precedence, and the evidence-rule catalogue.

This module is documentation-as-code: the strings here are the single wording used
in the official prompt (:mod:`poc.official_prompt`), and the precedence rules encoded
here are the ones the reference solver implements.
"""

from __future__ import annotations

from .models import Label

# Human-readable definitions shown to the model. Kept terse and unambiguous.
LABEL_DEFINITIONS: dict[Label, str] = {
    Label.INGESTION_LOSS: "The source record was not successfully ingested into the pipeline.",
    Label.EXTRACTION_LOSS: "Ingestion succeeded, but structured entity extraction failed for the record.",
    Label.WRONG_MERGE: "The record was incorrectly merged with a distinct entity (identity contradiction).",
    Label.FILTERED_OUT: "The entity was excluded from the final dataset by a business rule.",
    Label.PRESENT_UNDER_OTHER_ID: "The entity exists in the final dataset under a different identifier.",
    Label.UNDETERMINABLE: "The available evidence is insufficient to establish a unique outcome.",
}

# Fields required alongside each status, for a *complete* answer.
REQUIRED_FIELDS: dict[Label, tuple[str, ...]] = {
    Label.PRESENT_UNDER_OTHER_ID: ("final_id",),
    # merge_target_id is required for WRONG_MERGE ONLY when it is observable; the
    # evaluator consults the reference answer's `merge_target_observable` flag rather
    # than demanding it unconditionally.
    Label.WRONG_MERGE: (),
}

# --------------------------------------------------------------------------------------
# Precedence
# --------------------------------------------------------------------------------------
# The record's *actual* processing history has exactly one terminal outcome, taken as
# the EARLIEST stage at which the record left the happy path:
#
#     ingest -> extract -> resolve/merge -> filter -> final
#
# So a record that fails ingestion is INGESTION_LOSS even though "it is also not in the
# final dataset". This ordering is used by the simulator to compute the hidden actual
# outcome.
ACTUAL_OUTCOME_STAGE_ORDER: tuple[Label, ...] = (
    Label.INGESTION_LOSS,
    Label.EXTRACTION_LOSS,
    Label.WRONG_MERGE,
    Label.FILTERED_OUT,
    Label.PRESENT_UNDER_OTHER_ID,
)

# The *observable* outcome (what the solver and the model must produce) is distinct
# from the actual outcome: it is the unique label POSITIVELY ESTABLISHED by the
# artifacts available at the given observability level. If no label is positively
# established (i.e. more than one outcome remains consistent with the evidence), the
# answer is UNDETERMINABLE. UNDETERMINABLE is an epistemic label about the evidence,
# never an actual pipeline failure.

PIPELINE_DESCRIPTION = (
    "A multi-source pipeline ingests business records from three heterogeneous source "
    "types (business registry records, business directory records, and news articles) "
    "and processes them through six stages: (1) source ingestion, (2) structured entity "
    "extraction, (3) normalization, (4) entity resolution / deduplication, (5) business "
    "filtering, and (6) emission of a final canonical entity dataset. Records may be lost "
    "at any stage, deduplicated into a shared canonical id, incorrectly merged with a "
    "distinct entity, or excluded by a business rule."
)
