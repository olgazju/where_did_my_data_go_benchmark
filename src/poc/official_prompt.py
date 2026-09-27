"""The single official benchmark prompt (P2).

Self-contained: the base task render plus the three fixed additions —
  * evidence-grounding rules,
  * canonical-vs-final id clarification, and
  * mandatory structured `evidence_refs`.

Wording is frozen; the byte-for-byte content is regression-checked against the golden prompts
in `tests/fixtures/official_p2/`. There is no prompt-condition selection: the historical P0/P1
variants have been retired.
"""
from __future__ import annotations

import json

from .labels import LABEL_DEFINITIONS
from .models import ArtifactState, ObservableBundle


# --------------------------------------------------------------------------------------
# Base task render (evidence + labels + response schema)
# --------------------------------------------------------------------------------------
def _artifact_payload(bundle: ObservableBundle) -> dict:
    out: dict = {}
    for name, art in bundle.artifacts.items():
        if art.state is ArtifactState.MISSING:
            continue
        out[name] = {
            "state": art.state.value,
            "guaranteed_complete_for": art.complete_scopes,
            "rows": art.rows,
        }
    return out


def _render_base(bundle: ObservableBundle, source_id: str) -> str:
    labels_block = "\n".join(f"- {lbl.value}: {desc}" for lbl, desc in LABEL_DEFINITIONS.items())
    evidence = {
        "source_records": bundle.sources,
        "final_dataset": bundle.final_dataset,
        "intermediate_artifacts": _artifact_payload(bundle),
    }
    return f"""\
You are auditing a multi-stage data pipeline to determine what happened to one source record.

# Pipeline
{bundle.pipeline_description}

# Observability level
{bundle.level.value.upper()}. You may reason ONLY from the artifacts provided below.
Each intermediate artifact declares a `state` and, when COMPLETE, the scopes for which
it is guaranteed complete. Absence of a row is evidence ONLY within a
`guaranteed_complete_for` scope. If more than one outcome remains consistent with the
available evidence, the answer is UNDETERMINABLE.

# Identifier rules
Strong identifiers are namespaced (e.g. "registry:US-DE/4830012", "duns/11-2200-345").
Within one namespace a value denotes at most one real entity, so two records with
different values in the SAME namespace cannot be the same entity. Name similarity is
NOT a strong identifier.

# Classification labels
{labels_block}

# Evidence
{json.dumps(evidence, indent=2)}

# Task
Investigate source record: {source_id}

Return ONLY a JSON object with exactly these keys:
{{
  "source_id": "{source_id}",
  "status": "<one of the labels above>",
  "final_id": "<final entity id if status is PRESENT_UNDER_OTHER_ID, else null>",
  "merge_target_id": "<the id it was wrongly merged into if status is WRONG_MERGE and that id is observable, else null>"
}}
Use null for non-applicable ids. Do not include any text outside the JSON object."""


# --------------------------------------------------------------------------------------
# Fixed P2 additions (byte-frozen)
# --------------------------------------------------------------------------------------
_GROUNDING_RULES = """\
# Evidence-grounding rules
- Base your status ONLY on what the available artifacts POSITIVELY ESTABLISH. Do not guess.
- Name similarity ALONE never establishes identity. To answer PRESENT_UNDER_OTHER_ID you need
  either (a) a strong identifier shared between the source record and a final record, or (b) an
  explicit lineage link (resolution map and/or merge-decision log) tying the record to a final entity.
- If the available evidence is consistent with more than one outcome — e.g. the record could match
  two different final entities, or could equally have been lost, filtered, or merged — you MUST
  answer UNDETERMINABLE.
- Absence of a record is evidence ONLY inside an artifact that is COMPLETE for the relevant scope;
  in a PARTIAL or MISSING artifact, absence proves nothing.
- Prefer UNDETERMINABLE over an unsupported positive claim.

"""

_ID_CLAUSE = """\
# Identifier reporting
The pipeline uses two id spaces. `canonical_id` (e.g. "C10") is an INTERNAL id assigned at
resolution and appears in the intermediate artifacts. `final_id` (e.g. "ENT-501") is the PUBLIC
id in `final_dataset`. Whenever you report `final_id` or `merge_target_id`, report the PUBLIC
`final_id` from `final_dataset`, never the internal `canonical_id`. To convert, match an
artifact's `canonical_id` to `final_dataset.canonical_id` and read its `final_id`.

"""

_CITATIONS = """\
# Evidence citations (required, structured)
For any status other than UNDETERMINABLE you MUST populate "evidence_refs" with a JSON array of
citation objects, each of the form {"artifact": "<artifact name>", "match": {"<field>": "<value>"}},
drawn ONLY from the artifacts shown above. Cite the specific rows that establish your conclusion,
e.g. {"artifact": "resolution_map", "match": {"source_id": "SRC-1", "canonical_id": "C10"}} or
{"artifact": "final_dataset", "match": {"final_id": "ENT-501", "strong_id": "duns/11-2200-345"}}.
Do NOT cite name similarity. If you cannot cite establishing artifacts, answer UNDETERMINABLE with
"evidence_refs": [].

"""

_MERGE_LINE = ('  "merge_target_id": "<the id it was wrongly merged into if status is WRONG_MERGE '
               'and that id is observable, else null>"\n}')
_MERGE_LINE_STRUCTURED = (
    '  "merge_target_id": "<the id it was wrongly merged into if status is WRONG_MERGE '
    'and that id is observable, else null>",\n'
    '  "evidence_refs": [{"artifact": "<artifact name>", "match": {"<field>": "<value>"}}]'
    '  // use [] for UNDETERMINABLE\n}'
)


def render_official_prompt(bundle: ObservableBundle, source_id: str) -> str:
    """The single official benchmark prompt for one (record, observability level)."""
    p = _render_base(bundle, source_id)
    p = p.replace("# Task", _ID_CLAUSE + "# Task", 1)
    p = p.replace("# Task", _GROUNDING_RULES + "# Task", 1)
    p = p.replace("# Task", _CITATIONS + "# Task", 1)
    p = p.replace(_MERGE_LINE, _MERGE_LINE_STRUCTURED, 1)
    assert "evidence_refs" in p, "structured schema patch failed"
    return p
