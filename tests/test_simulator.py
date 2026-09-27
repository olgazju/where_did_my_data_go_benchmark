"""Simulator: fates -> ground truth + consistent master artifacts (benchmark scenarios)."""
from poc.models import ArtifactState, Label
from poc.scenarios import by_id
from poc.simulator import simulate


def test_wrong_merge_absorption():
    sim = simulate(by_id()["S07"].scenario)
    absorbed = sim.gt("S07-P")
    assert absorbed.actual_outcome is Label.WRONG_MERGE
    assert absorbed.final_id is None and absorbed.is_wrong_merge is True
    survivor = sim.gt("S07-SURV")
    assert survivor.actual_outcome is Label.PRESENT_UNDER_OTHER_ID
    assert survivor.final_id == "ENT-07"


def test_ingestion_loss_omitted_from_manifest():
    sim = simulate(by_id()["S04"].scenario)
    assert sim.gt("S04-P").actual_outcome is Label.INGESTION_LOSS
    assert sim.gt("S04-P").ingested is False
    manifest_ids = {r["source_id"] for r in sim.master_artifacts["ingestion_manifest"].rows}
    assert "S04-P" not in manifest_ids


def test_filtered_out_recorded_in_filter_log():
    sim = simulate(by_id()["S06"].scenario)
    assert sim.gt("S06-P").actual_outcome is Label.FILTERED_OUT
    dropped = [r for r in sim.master_artifacts["filter_log"].rows if r["kept"] is False]
    assert any(r["canonical_id"] == "C6" for r in dropped)


def test_master_artifacts_all_complete():
    sim = simulate(by_id()["S01"].scenario)
    assert all(a.state is ArtifactState.COMPLETE for a in sim.master_artifacts.values())
