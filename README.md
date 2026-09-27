# Where Did My Data Go?

A benchmark that measures whether an LLM can **distinguish evidence-supported conclusions from
plausible guesses** when investigating a missing entity in a multi-stage data pipeline under
varying observability.

Each task shows the model the observable artifacts of a pipeline execution and asks: *what
happened to this source record?* The trap is calibration — when the surviving evidence is
insufficient, the correct answer is `UNDETERMINABLE`, even when a name-similar entity is sitting
right there in the final dataset. A deterministic reference solver and a deterministic evaluator
score the answers; there is no LLM judge.

## Research question

How does incomplete observability affect an LLM's ability to trace missing entities through a
data pipeline — and can explicit evidence-grounding instructions make it commit only when the
evidence licenses a conclusion, without making it over-abstain?

## What it measures

- **15 independent, hand-authored scenarios**, each a self-contained mini-pipeline built around
  one *probe record*, covering: presence via strong id / lineage, ingestion & extraction loss,
  filtering, wrong merges (target observable and not), misleading name matches, ambiguous
  multi-matches, and an observationally-identical **counterfactual** pair.
- **Three observability levels** (`full` / `partial` / `minimal`) derived from the *same* hidden
  execution — only which lineage artifacts survive changes. The same record can therefore have a
  *different evidence-supported answer* at each level.
- **Hidden truth vs. evidence-supported reference.** The simulator knows the real fate; the
  reference solver never sees it — it derives the label that the *observable* evidence positively
  establishes, or `UNDETERMINABLE`.

### The six labels

`INGESTION_LOSS`, `EXTRACTION_LOSS`, `WRONG_MERGE`, `FILTERED_OUT`, `PRESENT_UNDER_OTHER_ID`,
`UNDETERMINABLE`. Strong identifiers are namespaced with a uniqueness guarantee (e.g.
`registry:US-DE/4830012`, `duns/11-2200-345`), so a wrong merge is provable from observable
evidence alone. Absence is evidence only inside an artifact that is *complete* for the relevant
scope.

### The official prompt (P2)

A single official prompt (`poc.official_prompt`), no condition selection. It combines the base
task render with three fixed additions: **evidence-grounding rules** (name similarity never
establishes identity; prefer `UNDETERMINABLE` over an unsupported positive), a **canonical-vs-final
id clarification**, and **mandatory structured `evidence_refs`** for every positive answer.

### The official evidence validator

`poc.validators` scores each positive answer on three axes, keeping citation *integrity* separate
from evidence *sufficiency*:

- `citation_integrity` — every cited artifact row is real and observable (never hallucinated or
  from a MISSING artifact);
- `evidence_sufficiency` — the cited rows actually establish the claimed status;
- `identifier_evidence_validity` — the reported id is a **public** `ENT-…` id (never an internal
  canonical id), is cited, and matches the reference.

### Lineage-free baselines

`poc.baselines` (name-similarity, strong-id, source+final-only) exist to demonstrate that some
tasks genuinely require lineage evidence: the calibration probes are exactly where name-similarity
fires a *wrong* positive.

## Project structure

```
src/poc/
  models.py          data model: labels, namespaced strong ids, three-valued artifact state
  scenario.py        Scenario / ScenarioRecord / FinalEntity / Fate (the schema)
  scenarios.py       the 15 official benchmark scenarios + specs
  simulator.py       Scenario -> hidden ground truth + master (COMPLETE) artifacts
  observability.py   derive Full / Partial / Minimal views (only ever removes information)
  solver.py          deterministic reference solver (observable evidence only)
  evaluator.py       parse + validate + score model answers (no LLM judge)
  official_prompt.py the single official prompt (P2), byte-frozen wording
  validators.py      official evidence-chain validator (integrity / sufficiency / identifier)
  baselines.py       deterministic lineage-free baselines
  benchmark.py       build / run / replay + offline validation gates
  config.py          BenchmarkConfig (execution params only)
  cli.py             `python -m poc benchmark <build|run|replay>`
  provider/          minimal LLM provider seam (Anthropic)
tests/               deterministic tests + tests/fixtures/official_p2/ (33 golden prompts)
configs/benchmark.yaml
archive/history.tgz  research history (experiments, analyses, raw responses) — provenance only
```

## Installation

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```
uv sync
```

## Running

**Build (offline — zero API calls):** runs the validation gates, renders the official prompts,
and writes the task matrix + cost estimate.

```
uv run --locked python -m poc benchmark build
```

**Run (live — explicitly opt-in):** executes the matrix against the model. Requires an API key
in the environment; it is resumable and stops at a spending guardrail.

```
ANTHROPIC_API_KEY=... uv run --locked python -m poc benchmark run
```

**Replay (offline — zero API calls):** re-scores preserved responses.

```
uv run --locked python -m poc benchmark replay
```

Execution parameters (model, `max_tokens`, samples per task, budget) live in
`configs/benchmark.yaml`. The official prompt and validator are fixed in code and are not
configurable. `temperature` is intentionally omitted (the target model rejects it).

## Evaluation metrics

Reported per task and, as the **primary** generalization view, per scenario (macro-averaged —
the three observability variants of one probe are correlated and are not independent examples):

- **Status accuracy** and **full-answer accuracy** vs. the reference.
- **Unsupported-positive rate** (on `UNDETERMINABLE`-reference tasks) — the headline calibration
  metric.
- **Correct-abstention rate** and **correct-positive rate** (guards against over-abstention).
- **Paired evidence-transition accuracy** — per scenario, correct at *both* bracketing levels
  (commit when the evidence is present, abstain when it is removed).
- **Evidence-chain validity** — the three validator axes above, for the official prompt.

## Findings

On the 15 unseen scenarios, evidence-grounding instructions generalized: the macro
unsupported-positive rate fell from **0.247** (an ungrounded baseline prompt) to **0.000** under
the official P2 prompt, with **no** loss of correct-positive accuracy (no over-abstention) and
100% citation integrity. The full research history — the prompt-comparison experiments,
counterfactual studies, raw API responses, and analyses that led to this design — is preserved in
`archive/history.tgz`.

## Testing

```
uv run --locked pytest
```

Deterministic tests cover the data model, simulator, observability redaction, the reference solver
(including monotonicity invariants and no ground-truth access), the evaluator, the official
validator, the baselines, the benchmark build/replay path, and a byte-for-byte regression of the
official prompt against the 33 golden prompts in `tests/fixtures/official_p2/`.
