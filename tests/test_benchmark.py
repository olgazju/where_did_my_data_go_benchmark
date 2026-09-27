"""Benchmark build/validate/replay — all offline (zero API calls)."""
import json

from poc import benchmark
from poc.config import BenchmarkConfig


def test_validation_gates_pass():
    ok, failures = benchmark.validate()
    assert ok, f"benchmark validation failed: {failures}"


def test_matrix_is_single_prompt_33_cells_107_calls():
    cfg = BenchmarkConfig()
    matrix = benchmark.build_matrix(cfg)
    assert len(matrix) == 33
    assert sum(c["k"] for c in matrix) == 107
    # single official prompt: every fingerprint pins one model + a prompt hash, no condition field
    assert all("condition" not in c for c in matrix)
    assert {b.spec.scenario_id for b in benchmark.official_scenarios()} and \
        len({b.spec.scenario_id for b in benchmark.official_scenarios()}) == 15


def test_build_writes_matrix_and_cost_zero_api(tmp_path):
    cfg = BenchmarkConfig(output_dir=tmp_path / "out")
    assert benchmark.cmd_build(cfg) == 0
    matrix = json.loads((cfg.output_dir / "task_matrix.json").read_text())
    cost = json.loads((cfg.output_dir / "cost_estimate.json").read_text())
    assert len(matrix) == 33 and cost["total_calls"] == 107


def test_replay_rescores_saved_samples_zero_api(tmp_path):
    cfg = BenchmarkConfig(output_dir=tmp_path / "out")
    # synthesize one preserved response for S01:minimal (reference PRESENT ENT-01)
    d = cfg.output_dir / "samples" / "S01_minimal"
    d.mkdir(parents=True)
    answer = json.dumps({"source_id": "S01-P", "status": "PRESENT_UNDER_OTHER_ID",
                         "final_id": "ENT-01", "merge_target_id": None, "evidence_refs": []})
    (d / "01.json").write_text(json.dumps({
        "scenario_id": "S01", "level": "minimal", "sample": 1, "text": answer,
    }))
    assert benchmark.cmd_replay(cfg) == 0
    summary = json.loads((cfg.output_dir / "replay_summary.json").read_text())
    assert summary["samples"] == 1 and summary["malformed"] == 0
