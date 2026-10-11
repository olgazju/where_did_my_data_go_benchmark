# %% [markdown]
# # Where Did My Data Go? — unguided track
# The same investigations as the official task, but the model gets only the task, the labels
# and the id rules: no evidence-grounding rules and no citation requirement. Does it keep
# evidence discipline without being told? Scoring is deterministic — no LLM judge.

# %%
"""Kaggle Benchmarks task "where-did-my-data-go-unguided" (unguided track).

Same wrapper as ``task.py`` with the unguided prompt (no grounding rules, no citations) and
full-answer accuracy as the score; keep the code identical apart from the track constants and
names (a test enforces it).

The model is called only through Kaggle's Model Proxy (``llm.prompt``); all scoring is
delegated to :mod:`kagglebench.scoring`. The benchmark source travels as an attached Kaggle
Dataset. Task names in ``@kbench.task(name=...)`` must be string literals: ``kaggle b t push``
validates the file statically.
"""
import os
import sys
import zipfile
from pathlib import Path

# Sub-run rendering is verbose and slow for 107 cases; the per-case dicts are still stored.
os.environ.setdefault("RENDER_SUBRUNS", "False")

_SOURCE_ZIP = "wdmdg_src.zip"  # produced by kaggle/package_source.sh


def _bootstrap_source(input_dir: str = "/kaggle/input") -> None:
    """Put the attached benchmark source on ``sys.path`` (no-op if ``poc`` already imports).

    Inlined here rather than in ``kagglebench`` because ``kagglebench`` ships in the same dataset.

    Args:
        input_dir — Kaggle input mount; the dataset sits at ``<slug>/`` or ``<owner>/<slug>/``
            and may hold an extracted ``poc/`` package or the zip itself.

    Raises:
        ModuleNotFoundError — Neither an extracted ``poc/`` nor ``wdmdg_src.zip`` was found.
    """
    try:
        import poc  # noqa: F401  (already importable, e.g. local dev)
        return
    except ModuleNotFoundError:
        pass

    base = Path(input_dir)
    for depth in range(1, 5):
        prefix = "*/" * depth
        for marker in sorted(base.glob(f"{prefix}poc/__init__.py")):
            sys.path.insert(0, str(marker.parent.parent.resolve()))
            return
        for archive in sorted(base.glob(f"{prefix}{_SOURCE_ZIP}")):
            if "poc/__init__.py" in zipfile.ZipFile(archive).namelist():
                sys.path.insert(0, str(archive.resolve()))
                return
    seen = sorted(str(p) for p in base.rglob("*"))[:20] if base.exists() else []
    raise ModuleNotFoundError(
        "Benchmark source not found: attach the dataset with "
        "`kaggle b t push ... -d <owner>/where-did-my-data-go-src`. "
        f"Saw under {input_dir}: {seen}"
    )


_bootstrap_source()

import kaggle_benchmarks as kbench  # noqa: E402
import pandas as pd  # noqa: E402  (Kaggle-provided; also a kaggle-benchmarks dep)

from kagglebench.scoring import (  # noqa: E402
    aggregate,
    build_grid,
    case_prompt,
    score_response,
)

TRACK = "unguided"
TASK_NAME = "where-did-my-data-go-unguided"
TASK_DESCRIPTION = (
    "Investigate a missing record in a multi-stage data pipeline under varying "
    "observability, with no grounding rules or citation requirement. Score: scenario-macro "
    "full-answer accuracy: does the model abstain on thin evidence without being told to?"
)


# %%
@kbench.task(name="wdmdg-investigate-record-unguided", description="One investigation, scored deterministically.",
             store_task=False)
def investigate_record(llm, scenario_id: str, level: str, rep: int) -> dict:
    """Ask the model one investigation and score it; stored as a sub-run, not a separate task.

    Args:
        llm — The Kaggle model under test.
        scenario_id — Official scenario id.
        level — Observability level.
        rep — Sample index within the condition.

    Returns:
        The per-case record from :func:`kagglebench.scoring.score_response`.
    """
    answer = llm.prompt(case_prompt(scenario_id, level, TRACK))
    return score_response(scenario_id, level, int(rep), answer, TRACK)


@kbench.task(name="where-did-my-data-go-unguided", description=TASK_DESCRIPTION)
def where_did_my_data_go_unguided(llm) -> tuple[float, float]:
    """Run every grid case through the model and aggregate the unguided-track metrics.

    Args:
        llm — The Kaggle model under test.

    Returns:
        ``(primary_score, ci95)``: scenario-macro full-answer accuracy and the half-width of
        its 95% CI over the 15 scenarios.
    """
    rows = build_grid()
    if os.environ.get("WDMDG_SMOKE"):  # local smoke test: one sample per condition (33 calls)
        rows = [r for r in rows if r["rep"] == 1]
    grid = pd.DataFrame(rows)

    with kbench.client.enable_cache():
        runs = investigate_record.evaluate(
            llm=[llm],
            evaluation_data=grid,
            n_jobs=4,
            timeout=300,
            on_failure="continue",
        )

    cases = [r.result for r in runs.completed_runs]

    # A case that errored or timed out is scored as an empty (malformed) answer, so an
    # unreliable model cannot improve its score by dropping out of the denominator.
    scored = {(c["scenario_id"], c["level"], c["rep"]) for c in cases}
    missing = [r for r in rows if (r["scenario_id"], r["level"], r["rep"]) not in scored]
    cases += [score_response(r["scenario_id"], r["level"], r["rep"], "", TRACK) for r in missing]

    metrics = aggregate(cases, TRACK)
    print(f"[{TASK_NAME}] cases={len(cases)} errored={len(missing)}")
    for key, value in metrics.items():
        print(f"  {key}: {value}")

    return float(metrics["primary_score"]), float(metrics["primary_score_ci95"])


# %%
# Without this call the pushed notebook defines the task but never runs it, and Kaggle
# records nothing. The model is whichever one `kaggle b t run -m <model>` selects.
where_did_my_data_go_unguided.run(kbench.llm)
