# Kaggle Benchmarks integration — "Where Did My Data Go?"

This directory is the **Kaggle-facing layer only**. It does not change the benchmark:
the prompts, deterministic solver, evaluator, and evidence validator are reused
verbatim from `poc`. The Kaggle layer only handles **execution, packaging, and
aggregation**.

| File | Role |
|---|---|
| `task.py` | Pushed to Kaggle as the official task `where-did-my-data-go`. Calls the model via `llm.prompt(...)` (Kaggle Model Proxy) — never Anthropic directly. |
| `task_unguided.py` | Pushed as `where-did-my-data-go-unguided`: identical code (a test enforces it), unguided prompt, full-answer accuracy as the score. |
| `package_source.sh` | Stages the benchmark source (`poc` + `kagglebench`) as a single `wdmdg_src.zip` for upload (the CLI skips sub-directories by default, so the source travels as one file; `task.py` imports from the zip or its extracted copy). |
| `requirements-kaggle.txt` | Runtime deps for a local dry-run (pre-installed on Kaggle). |
| `summarize_results.py` | Offline: turns `kaggle b t download` output into the leaderboard / per-scenario / traps tables. |
| `../kagglebench/` | Pure grid/scoring/aggregation over `poc`. No Kaggle or model dependency. Travels in the dataset. |

## What runs

- **Two tasks over the same grid**, each fanning out the **33 observable conditions × k**
  (k=3 general, k=5 calibration ⇒ **107 sub-runs**) and returning `(score, ci95)` — the
  primary leaderboard score and the half-width of its 95% CI over the 15 scenarios:
  - `where-did-my-data-go` (official): grounding rules + mandatory structured citations;
    score = *scenario-macro evidence-supported accuracy* (correct **and** grounded in cited
    observable evidence).
  - `where-did-my-data-go-unguided`: task, labels and id rules only; score =
    *scenario-macro full-answer accuracy*. Citation metrics are blank on this track.
- **Detailed results are preserved** as the per-case sub-runs (predictions, references,
  `evidence_refs`, citation integrity, evidence sufficiency, identifier validity,
  observability level). The full metric block is also logged/printed by the task:
  evidence-supported accuracy, guess rate on undeterminable cases, unsupported-positive
  rate, correct abstention, correct-positive rate, paired evidence-transition accuracy, and evidence-chain validity.

## Prerequisites

A Kaggle account with phone verification (Settings → Phone verification); without it the
benchmarks API answers 403. No other API key is needed — models are called through Kaggle's
Model Proxy within your free quota.

```bash
pip install -U kaggle          # >= 2.2 has the `benchmarks` command group (alias `kaggle b`)
kaggle auth login              # OAuth in the browser (or: export KAGGLE_API_TOKEN=... from kaggle.com/settings/api)
kaggle benchmarks init -y      # writes .env with MODEL_PROXY_* credentials
kaggle benchmarks quota        # check Model Proxy spend limits (daily / monthly $)
kaggle b t models              # the full model catalog available for `run`
KAGGLE_USER=<your-kaggle-username>
```

## 1. Package the source as a Kaggle Dataset

```bash
bash kaggle/package_source.sh build/kaggle-dataset
# set the owner in the staged metadata to your authenticated username:
python3 - "$KAGGLE_USER" <<'PY'
import json, sys
p = "build/kaggle-dataset/dataset-metadata.json"
m = json.load(open(p)); m["id"] = f"{sys.argv[1]}/where-did-my-data-go-src"
json.dump(m, open(p, "w"), indent=2)
PY

# first upload (-u = public, so others can re-run the task):
kaggle datasets create -p build/kaggle-dataset -u
# later updates:
kaggle datasets version -p build/kaggle-dataset -m "update benchmark source"
```

## 2. Push both tasks (attaching the source dataset)

```bash
kaggle b t push where-did-my-data-go          -f kaggle/task.py          -d "$KAGGLE_USER/where-did-my-data-go-src" --wait
kaggle b t push where-did-my-data-go-unguided -f kaggle/task_unguided.py -d "$KAGGLE_USER/where-did-my-data-go-src" --wait
```

Task names in `@kbench.task(name=...)` must be string literals — `push` validates the file
statically and ignores names held in variables.

## 3. First live run (this spends Model Proxy budget)

```bash
kaggle benchmarks tasks run where-did-my-data-go          -m anthropic/claude-sonnet-4 --wait
kaggle benchmarks tasks run where-did-my-data-go-unguided -m anthropic/claude-sonnet-4 --wait
# check progress / logs:
kaggle benchmarks tasks status where-did-my-data-go -m anthropic/claude-sonnet-4
kaggle benchmarks tasks log    where-did-my-data-go -m anthropic/claude-sonnet-4
```

Use the model id(s) your Model Proxy authorizes (`kaggle benchmarks tasks models` lists
the catalog). The task is model-agnostic; the model is selected here, not in code.

## 4. Download detailed results & publish

```bash
kaggle benchmarks tasks download where-did-my-data-go          -o ./results --force
kaggle benchmarks tasks download where-did-my-data-go-unguided -o ./results --force
PYTHONPATH=src:. python kaggle/summarize_results.py ./results     # markdown tables for the write-up
kaggle benchmarks tasks publish  where-did-my-data-go
kaggle benchmarks tasks publish  where-did-my-data-go-unguided
```

## 5. Create the Benchmark (web UI only)

The CLI manages **tasks**; a **benchmark** (the collection with the leaderboard you link to)
is created on kaggle.com: Benchmarks → *Create benchmark* → add both published tasks →
pick the models → publish. The resulting URL (`kaggle.com/benchmarks/<you>/<slug>`) is the
link the DEV submission requires. Then:

```bash
kaggle benchmarks leaderboard "$KAGGLE_USER/<benchmark-slug>" --show
```

## Local smoke test (optional, spends budget)

Each task file ends with `<task>.run(kbench.llm)` — without that call Kaggle records
nothing. Running the file locally therefore performs a real run against `LLM_DEFAULT`
from `.env`. `WDMDG_SMOKE=1` limits it to one sample per condition (33 calls):

```bash
python -m pip install -r kaggle/requirements-kaggle.txt
kaggle b init -y                                   # .env with proxy creds + LLM_DEFAULT
set -a; source .env; set +a
WDMDG_SMOKE=1 PYTHONPATH=src:. python kaggle/task.py
ls -1 *.run.json                                   # a run file means the task is push-ready
```

## Gotchas

- **Repeat `-d` on every push.** Re-pushing without it silently detaches the source dataset,
  and the task then fails at import with a "Benchmark source not found" error.
- **Repeat flags, don't space-separate them:** `-m a -m b`, not `-m a b`.
- **Proxy keys are short-lived.** On an auth error, `kaggle b auth -y`.
- **Errored/timed-out cases are scored as malformed**, not dropped, so the denominator is
  always 107 (or 33 in smoke mode).
