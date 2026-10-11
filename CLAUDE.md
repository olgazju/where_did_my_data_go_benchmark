# CLAUDE.md

Guidance for coding agents working in this repository.

## What this is

**"Where Did My Data Go?"** — a benchmark measuring whether an LLM can distinguish
**evidence-supported conclusions from plausible guesses** when investigating a missing entity in
a multi-stage data pipeline under varying observability. A deterministic reference solver and a
deterministic evaluator score answers; there is **no LLM judge**.

The repo is a clean, standalone benchmark. The full research history that produced it (four
experiments, raw API responses, analyses, design spec) is archived in `archive/history.tgz` and
is **not** needed to run anything.

## Environment / tooling

- Python **3.12+**, managed by **uv**. `uv` is at `~/.local/bin` and is **not** on the default
  PATH. Before running uv commands: `export PATH="$HOME/.local/bin:$PATH"` and `unset VIRTUAL_ENV`.
- Always run through uv with the lockfile: `uv run --locked ...`.
- No git in this working copy — **deletions are permanent**. Confirm before destructive changes.

## Commands

```
uv sync
uv run --locked python -m poc benchmark build     # offline: validation gates + task matrix + cost. ZERO API calls.
ANTHROPIC_API_KEY=... uv run --locked python -m poc benchmark run     # live run. Explicitly opt-in; resumable; $ guardrail.
uv run --locked python -m poc benchmark replay    # offline: re-score preserved responses. ZERO API calls.
uv run --locked pytest                            # deterministic test suite
```

The local CLI runs only the official prompt (Evidence-Grounded Investigation Prompt). The Kaggle
layer adds a second, **unguided** track (`render_unguided_prompt`: the base render + id clause,
no grounding rules, no citations; byte-identical to the archived "P0").

## Architecture (`src/poc/`)

Single source of truth is the simulator: a `Scenario` → hidden ground truth + master (COMPLETE)
artifacts; observability views and reference labels all derive from that one execution.

| Module | Role |
|---|---|
| `models.py` | Labels, namespaced `StrongId` (+ the one shared `source_strong_ids` / `final_strong_keys` helper pair), three-valued `ArtifactState`, `ObservableBundle`, `ReferenceAnswer` |
| `scenario.py` | `Scenario` / `ScenarioRecord` / `FinalEntity` / `Fate` schema (models only) |
| `scenarios.py` | The **15 official scenarios** + `ScenarioSpec` (probe, levels, expected reference, partial-drop policy) |
| `simulator.py` | `simulate(scenario)` → ground truth + master artifacts (requires a scenario arg) |
| `observability.py` | `build_bundle(sim, level, partial_resolution_drop=None)` — Full/Partial/Minimal; only ever removes info |
| `solver.py` | Deterministic reference solver — observable evidence only, never ground truth |
| `evaluator.py` | Parse (prediction + `evidence_refs`) + score a model answer (status + full-answer); tracks malformed |
| `official_prompt.py` | The official prompt (**Evidence-Grounded Investigation Prompt**, `render_prompt()`) and its unguided control (`render_unguided_prompt()`), both byte-frozen |
| `validators.py` | Official evidence-chain validator: `citation_integrity`, `evidence_sufficiency`, `identifier_evidence_validity`; `chain_validity()` runs them all (used by both `benchmark.py` and `kagglebench`) |
| `baselines.py` | Lineage-free baselines (name-similarity, strong-id, source+final-only) |
| `benchmark.py` | `build` / `run` / `replay` + offline `validate()` gates |
| `config.py` | `BenchmarkConfig` (execution params only) |
| `cli.py` | `python -m poc benchmark <build|run|replay>` |
| `provider/` | Minimal LLM seam; only `anthropic.py` |

## Kaggle Benchmarks integration (`kagglebench/`, `kaggle/`)

A thin, additive layer for the official Kaggle Benchmarks platform. It reuses `poc` verbatim
and only handles execution/packaging/aggregation — it does **not** change the benchmark.

- `kagglebench/` — pure, Kaggle-free: `scoring.py` (grid, per-case scoring, metric aggregation
  over the unchanged `poc`, with `TRACKS` = `official` / `unguided`). It reuses `poc` helpers
  (`benchmark.scenario_bundle`, `validators.chain_validity`, `evaluator.parse_prediction`) rather
  than copying them. No `kaggle_benchmarks`/`pandas` import, so the core tests stay
  dependency-light. The task files bootstrap `sys.path` inline (they cannot import a helper from
  the dataset they are locating).
- `kaggle/task.py` — official task `where-did-my-data-go` (primary: scenario-macro
  evidence-supported accuracy). `kaggle/task_unguided.py` — `where-did-my-data-go-unguided`
  (primary: scenario-macro full accuracy). Identical code apart from track constants (a test
  enforces it). Both call `llm.prompt(...)` (Kaggle Model Proxy), never Anthropic, return
  `(score, ci95)`, and MUST end with `<task>.run(kbench.llm)` or Kaggle records nothing.
  `@kbench.task(name=...)` must be a string literal (push validates statically).
- `kaggle/package_source.sh` — stages `poc` + `kagglebench` as ONE `wdmdg_src.zip` (the CLI
  skips sub-directories by default); the task bootstrap imports from the zip or its extraction.
- `kaggle/summarize_results.py` — offline tables (with-rules vs on-instinct, per-track
  leaderboards, per-scenario matrix, traps) from `kaggle b t download` output.
- `kaggle/{README.md,requirements-kaggle.txt}` — workflow, deps.
- **Do not** add `kaggle-benchmarks` to `pyproject.toml`/`uv.lock`: it drags in
  playwright/jupyter/docker and forces `pydantic>=2.11.5`. It is pre-installed on Kaggle; keep it
  in `kaggle/requirements-kaggle.txt` only. Offline test: `tests/test_kaggle_task.py`.

## Core invariants — do not break these

1. **Two worlds stay separate.** The simulator knows hidden truth (`true_entity_id`,
   `actual_outcome`); the solver and the model see only the `ObservableBundle`. A test asserts the
   solver source never references ground-truth fields. No ground-truth token may leak into any bundle.
2. **Monotonicity.** Removing evidence can turn a determinable answer into `UNDETERMINABLE`, but
   never introduces a new contradictory positive. Enforced by property test.
3. **Absence is evidence only within a COMPLETE scope.** Dropping rows from an artifact must also
   drop its completeness guarantee (COMPLETE→PARTIAL).
4. **Strong ids are namespaced with uniqueness** (`registry:US-DE/4830012`, `duns/…`). Two
   different values in the same namespace ⇒ different entities ⇒ a provable wrong merge. Name
   similarity is never identity.
5. **`merge_target_id` only when observable**, else `null` (the evaluator consults
   `merge_target_observable`).
6. **Scenario independence.** Each scenario is its own mini-pipeline. The counterfactual twin
   (`S11`/`S11B`) must have **byte-identical** observable bundles at the tested level; only hidden
   truth differs. `S11B` is analysis-only (shares S11's prompt; not a separate API cell).
7. **`temperature` is omitted** for the target model (`claude-sonnet-5` rejects it); the provider
   drops it automatically. Do not add it to fingerprints.

## Constraints when changing the benchmark

- **Do not change** scenario definitions, reference labels, or prompt wording without
  intent. Both prompts are regression-locked: `tests/test_official_prompt.py` compares
  `render_prompt` / `render_unguided_prompt` byte-for-byte against the 33 golden prompts each in
  `tests/fixtures/official_prompt/` and `tests/fixtures/unguided_prompt/`. If you deliberately
  change wording, regenerate those fixtures in the same change and say so.
- **Two prompts, no more.** Official (archived "P2") and unguided (archived "P0"). The archived
  "P1" (rules without citations) is not a benchmark track.
- **Validator separation.** Keep citation *integrity* (are cited rows real/observable?) distinct
  from evidence *sufficiency* (do they establish the status?). Structured `evidence_refs` accept
  real field names (`registration_number`, `duns`) and list membership (`strong_ids`, `members`),
  and a completeness citation (`{"guaranteed_complete_for": "<scope>"}`) for absence-based labels.
- **API keys**: pass inline as an env var only (`ANTHROPIC_API_KEY=... uv run ...`). Never write a
  key to any file. `build` and `replay` make zero API calls; `run` is opt-in and key-gated.

## Scenario map (15, in `scenarios.py`)

- **A — positive controls** (should commit at every level): S01 registry strong id, S02 DUNS,
  S14 determinable-at-every-level (anti "minimal ⇒ abstain").
- **B — transitions** (same record, answer changes with evidence): S03 present-via-map, S04
  ingestion loss, S05 extraction loss, S06 filtered-out (link survives at partial), S07 wrong-merge
  (full-only), S15 filtered-out (link dropped at partial, full-only).
- **C — merge / id-granularity**: S08 wrong-merge target *not* observable (null), S13 wrong-merge
  where lineage exposes canonical id but the answer must be the public final id.
- **D — calibration probes** (reference `UNDETERMINABLE`; where grounding matters, k=5): S09 false
  name match / true entity absent, S10 near-duplicate name, S11 counterfactual A/B, S12 ambiguous
  multi-match.

33 observable task-conditions; k=3 general, k=5 calibration ⇒ 107 calls/run.

## Metrics

Reported per task and, as the **primary** view, per scenario (macro-averaged — the observability
variants of one probe are correlated). Status/full accuracy; **unsupported-positive rate** (the
headline calibration metric); correct-abstention; correct-positive (over-abstention guard); paired
**evidence-transition accuracy**; and the three evidence-chain validity axes.

## State & finding

The official Evidence-Grounded Investigation Prompt (evidence-grounding rules + canonical-vs-final
id clause + structured `evidence_refs`) generalized across the 15 unseen scenarios: macro
unsupported-positive fell to **0.000** with no over-abstention and 100% citation integrity. Details
are in `archive/history.tgz`.
