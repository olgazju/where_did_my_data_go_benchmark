# archive/

`history.tgz` is a compressed archive of the project's research history, retained for
provenance and reproducibility of the published results. It is **not** part of the runnable
benchmark and nothing in `src/poc/` depends on it.

Contents:

- `experiments/` — Experiments 01–04 (prompt-intervention, counterfactuals, the v0.2
  generalization run) with their scripts, rendered prompts, raw API responses, and reports.
- `analysis/` — Pilot-01 analysis.
- `design/` — the Benchmark v0.2 design specification.
- `runs/` — the sample POC run.
- `prompts/frozen_v1/` — the frozen P0/P1/P2 prompt-condition archive.
- `configs/poc.yaml` — the retired POC pilot configuration.

To inspect: `tar -xzf archive/history.tgz -C /tmp/history` (extracts into `/tmp/history`).
