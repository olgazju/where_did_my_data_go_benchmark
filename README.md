# 🕵️ Where Did My Data Go?

**A benchmark for whether an LLM knows the difference between *"the evidence shows"* and *"it's probably that one."***

Every data engineer has had this ticket:

> *"Umbrella Corp is in the CRM export but missing from the dashboard. Where did it go?"*

You open the pipeline. Ingestion manifest, extraction log, entity-resolution map, merge log,
filter log. Half of them were rotated out last week. And in the final table there's an entity
called **"Umbrella Corporation Inc"**, which is *very* tempting.

A good investigator says *"it was filtered out by rule R3, here's the log row"* when the log row
exists. When the trail has gone cold, they say *"I can't tell"*, even with a name-alike staring
at them. A bad investigator picks the name-alike and moves on.

This benchmark measures which kind of investigator a model is. Every answer is scored by a
**deterministic reference solver and evidence validator. There is no LLM judge.**

> 📊 **Kaggle benchmark:** [www.kaggle.com/benchmarks/olgazju/where-did-my-data-go](https://www.kaggle.com/benchmarks/olgazju/where-did-my-data-go) · **Tasks:** `where-did-my-data-go` (with rules) and `where-did-my-data-go-unguided` (on instinct)

---

## What a task looks like

The model gets one pipeline run as JSON: the source records, the final dataset, and whichever
intermediate artifacts survived. Then it is asked what happened to one record.

```
Investigate source record: S09-P        (a news mention of "Vertex Materials")

final_dataset:      ENT-09D  "Vertex Materials Inc"  registry:US-DE/4830012
ingestion_manifest: PARTIAL  (S09-P was ingested)
resolution_map:     — not available at this observability level —
```

**The trap:** an entity with the same name sits right there in the final dataset.
**The right answer:** `UNDETERMINABLE`. A news mention has no strong id, the lineage that would
connect it to ENT-09D is gone, and a matching name is not identity. (In the hidden ground truth,
the record was filtered out and ENT-09D is a different company.)

The model must answer in strict JSON. Any positive claim must come with **structured citations**
to the exact artifact rows that prove it:

```json
{
  "source_id": "S04-P",
  "status": "INGESTION_LOSS",
  "final_id": null,
  "merge_target_id": null,
  "evidence_refs": [
    {"artifact": "ingestion_manifest", "match": {"guaranteed_complete_for": "B1"}}
  ]
}
```

### The six verdicts

| Label | Meaning |
|---|---|
| `PRESENT_UNDER_OTHER_ID` | It's in the final dataset, just under a different id |
| `INGESTION_LOSS` | It never made it into the pipeline |
| `EXTRACTION_LOSS` | Ingested, but entity extraction failed |
| `FILTERED_OUT` | A business rule removed it |
| `WRONG_MERGE` | It was merged into a *different* real-world entity (provable via conflicting strong ids) |
| `UNDETERMINABLE` | The surviving evidence doesn't single out one outcome |

### Three observability levels, one hidden truth

Each scenario is executed **once** by a simulator. The `full` / `partial` / `minimal` views are
derived from that one execution by *removing* artifacts. They never add or alter anything. So the
same record can have a different evidence-supported answer at each level. For example, S03 is
`PRESENT` at full and `UNDETERMINABLE` at minimal. A model has to commit when the evidence is
there **and** back off when it disappears.

Two rules make the puzzles fair:

- **Absence is evidence only inside a complete scope.** "Not in the manifest" proves
  `INGESTION_LOSS` only if the manifest is guaranteed `COMPLETE` for that batch.
- **Strong ids are namespaced and unique** (`registry:US-DE/4830012`, `duns/11-2200-345`).
  Two different values in one namespace means two different companies, which makes a wrong
  merge provable. A similar name never establishes identity.

---

## The 15 scenarios

Each one is a self-contained mini-pipeline built around a single probe record. Each targets one
failure mode.

| Group | Scenarios | What it tests |
|---|---|---|
| **A · Positive controls** | S01 registry id · S02 DUNS · S14 determinable even at minimal | Does the model commit when it *should*? (guards against "minimal ⇒ always abstain") |
| **B · Transitions** | S03 present-via-map · S04 ingestion loss · S05 extraction loss · S06 / S15 filtered out · S07 wrong merge | The answer changes as evidence is removed. Commit, then back off. |
| **C · Merge / id granularity** | S08 merge target not observable · S13 internal canonical id vs public final id | Report the right id, or `null` when it isn't observable |
| **D · Calibration traps** | S09 false name match · S10 identical-name decoy · S11 counterfactual twin · S12 ambiguous multi-match | Reference is `UNDETERMINABLE`. A name-alike is right there. |

**S11 / S11B** are a counterfactual pair. Their observable bundles are **byte-identical**, and
only the hidden truth differs (in one the name-match is real, in the other it isn't). Since no
model can tell them apart, the only defensible answer to both is `UNDETERMINABLE`. A test enforces
the byte-identity.

That gives 33 task conditions (scenario × level), sampled k=3 times (k=5 for the calibration
traps), for **107 model calls per run**.

---

## Two tracks: with rules, and on instinct

The same 107 investigations run as two Kaggle tasks:

| Task | The model gets | Primary score |
|---|---|---|
| `where-did-my-data-go` (**official**) | Task, labels, id rules **+ explicit evidence-grounding rules + a mandatory structured-citation contract** | Evidence-supported accuracy |
| `where-did-my-data-go-unguided` | Task, labels, id rules. That's it. | Full-answer accuracy |

Even the unguided prompt is not a trick. It still says that name similarity is not a strong
identifier, and that if more than one outcome fits the evidence the answer is `UNDETERMINABLE`.
The official track measures whether a model can *apply* evidence discipline once it is spelled
out. The unguided track measures whether the model has that discipline without being told. The
gap between the two tracks shows how much each model depends on the extra instructions.

## Scoring

**Official primary score: scenario-macro evidence-supported accuracy.** An answer counts only if
it is **correct** *and*, for a positive claim, **grounded**: every citation points to a real,
observable row, the cited rows actually establish the claimed status, and the reported id is
the public `ENT-…` id rather than an internal canonical id. Scores are averaged within each
scenario first and then across the 15 scenarios, because the three levels of one probe are
correlated. The task returns the score together with a 95% CI over scenarios.

**Unguided primary score: scenario-macro full-answer accuracy.** The status and ids must be
correct. No citations are requested, so citation metrics are reported as blank.

Secondary metrics, logged in every run:

| Metric | What it catches |
|---|---|
| `guess_rate_on_undeterminable` | **The headline calibration number:** how often the model commits to *any* conclusion when the evidence licenses none |
| `correct_positive_rate` | Over-abstention: refusing to commit when the evidence is clear |
| `paired_evidence_transition_accuracy` | Right at *both* ends of an evidence transition (commit with evidence, abstain without it) |
| `citation_integrity_rate` | Hallucinated citations: rows that don't exist or come from a missing artifact |
| `evidence_sufficiency_rate` | Real rows that don't actually prove the claim |
| `identifier_evidence_validity_rate` | Internal canonical id reported instead of the public final id |

### Floors: what does a dumb strategy score?

| Strategy | Primary | Status acc. | Guess rate on undeterminable |
|---|---|---|---|
| Always answer `UNDETERMINABLE` | 0.444 | 0.444 | 0.000 |
| Match by strong id only, never cite | 0.444 | 0.644 | 0.000 |
| Match by most similar name | 0.089 | 0.333 | **0.727** |

The name-matching heuristic is exactly the behavior the calibration traps are built to catch.

---

## Results

Kaggle Model Proxy, 107 calls per track per model, no failed calls. Sorted by price. Cost and
latency are for the official track. Live leaderboard: [www.kaggle.com/benchmarks/olgazju/where-did-my-data-go](https://www.kaggle.com/benchmarks/olgazju/where-did-my-data-go)

| Model | $ per 1k investigations | Median s / investigation | Guess rate on undeterminable, unguided | Full acc., unguided | Full acc., official | **Primary, official** (±95% CI) |
|---|---|---|---|---|---|---|
| GPT-6 Luna | $0.18 | 2.4 | 0.000 | 1.000 | 1.000 | 0.948 ± 0.051 |
| Claude Haiku 5.5 | $0.43 | 3.0 | 0.068 | 0.966 | 1.000 | 0.993 ± 0.015 |
| Qwen 3 Next 80B Instruct | $0.46 | 10.0 | **0.591** | 0.502 | 0.756 | 0.652 ± 0.223 |
| Gemini 3.5 Flash-Lite | $0.88 | 0.9 | **0.455** | 0.701 | 0.852 | 0.770 ± 0.178 |
| Gemma 4 31B | $1.18 | 37.7 | 0.000 | 0.993 | 0.933 | 0.815 ± 0.140 |
| GPT-6 Sol | $2.80 | 1.8 | 0.000 | 0.956 | 0.978 | 0.978 ± 0.044 |
| Claude Sonnet 5 | $6.07 | 2.6 | **0.295** | 0.874 | 1.000 | 0.993 ± 0.015 |
| Gemini 3.7 Flash | $7.10 | 6.9 | 0.091 | 0.898 | 0.933 | 0.830 ± 0.149 |

With the official prompt every model except Qwen (0.068) has a guess rate of 0.000. Tables per
scenario and the list of traps each model fell for come from `kaggle/summarize_results.py`.

### Development run

Development run before the Kaggle port: `claude-sonnet-5` via the Anthropic API, 107 calls per
track, re-scored with the current scorer. Both prompts are byte-identical to the ones the Kaggle
tasks send.

| Track | Primary (±95% CI) | Status acc. | Guess rate on undeterminable | Transition acc. |
|---|---|---|---|---|
| Unguided | 0.895 ± 0.114 | 0.895 | **0.250** | 0.611 |
| **Official** | **0.985 ± 0.020** | 1.000 | **0.000** | 1.000 |

The primary differs per track (full-answer accuracy vs. evidence-supported accuracy), so compare
the tracks on status accuracy and guess rate.

The unguided prompt already says that name similarity is not a strong identifier and that any
ambiguity means `UNDETERMINABLE`. Even so, the model claimed a match in 4 of 5 samples on the
ambiguous multi-match (S12), in 3 of 3 on the "lineage just disappeared" case (S03 minimal), and
in 3 of 3 on the "filter log is partial" case (S15 partial). Adding explicit grounding rules and
mandatory structured citations removed every unsupported guess without causing any
over-abstention. Citation integrity was 100%: no hallucinated rows.

The full research trail is archived in `archive/history.tgz`: four experiments, pre-registered
verdict rules, raw API responses, and a validator audit.

---

## Run it on Kaggle

Each Kaggle task is a thin wrapper. It calls the model through Kaggle's Model Proxy and hands
everything else to the unchanged benchmark code. See [`kaggle/README.md`](kaggle/README.md) for
the full workflow. In short:

```bash
bash kaggle/package_source.sh build/kaggle-dataset          # stage poc + kagglebench as one zip
kaggle datasets create -p build/kaggle-dataset -u           # public; set the owner in dataset-metadata.json first
kaggle b t push where-did-my-data-go          -f kaggle/task.py          -d <you>/where-did-my-data-go-src --wait
kaggle b t push where-did-my-data-go-unguided -f kaggle/task_unguided.py -d <you>/where-did-my-data-go-src --wait
kaggle b t run  where-did-my-data-go          -m <model-slug> -m <model-slug> --wait
kaggle b t run  where-did-my-data-go-unguided -m <model-slug> -m <model-slug> --wait
kaggle b t publish where-did-my-data-go && kaggle b t publish where-did-my-data-go-unguided
# then create the Benchmark in the Kaggle web UI and add both tasks to it
```

## Run it locally

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run --locked python -m poc benchmark build     # offline: validation gates + task matrix + cost. Zero API calls.
uv run --locked pytest                            # deterministic test suite
ANTHROPIC_API_KEY=... uv run --locked python -m poc benchmark run      # live, opt-in, resumable, $ guardrail
uv run --locked python -m poc benchmark replay    # offline: re-score preserved responses
```

A full local run is 107 calls, about $1 with Claude Sonnet 5.

---

## How it's built

```
src/poc/
  scenarios.py       the 15 hand-authored scenarios + their expected reference answers
  simulator.py       scenario -> hidden ground truth + complete pipeline artifacts
  observability.py   full / partial / minimal views (only ever removes information)
  solver.py          deterministic reference solver: sees observable evidence only, never the truth
  official_prompt.py the byte-frozen official prompt + its unguided control
  evaluator.py       parse + validate + score answers
  validators.py      citation integrity / evidence sufficiency / identifier validity
  baselines.py       lineage-free baselines (name similarity, strong id, source+final only)
  benchmark.py, cli.py   build / run / replay + offline validation gates
kagglebench/         Kaggle-free grid, per-case scoring, metric aggregation
kaggle/task.py       the official Kaggle task (task_unguided.py: same code, unguided track)
tests/               deterministic tests + 2×33 golden prompts (byte-for-byte regression)
```

Invariants enforced by tests:

1. **Two worlds stay separate.** The simulator knows the truth. The solver and the model see
   only the observable bundle. A test asserts the solver never touches ground-truth fields.
2. **Monotonicity.** Removing evidence can turn an answer into `UNDETERMINABLE` but never into a
   different positive answer (property test).
3. **References are reproduced, not hand-labelled.** An offline gate asserts the solver
   reproduces every authored expected answer at every level.
4. **The prompts are frozen.** All 2×33 rendered prompts are compared byte-for-byte with golden files.

## Limitations

- **15 scenarios is small.** They are hand-built to isolate one failure mode each, which keeps
  the results easy to interpret, but CIs are wide (±0.1–0.2 for mid-range models). Treat
  leaderboard gaps under ~0.1 as ties.
- **Synthetic pipelines.** The structure is realistic, but the companies and ids are invented.
  That keeps the benchmark free of training-data contamination, and also makes it less messy
  than production.
- **The official track states the rules**, so it measures whether a model can *apply* evidence
  discipline once told. That's why the unguided track exists. Read the two scores side by side.
- Kaggle calls use the SDK default (`temperature=0`), so repeated samples are close to
  deterministic and are not independent draws.
