"""Official evidence validator: integrity, sufficiency, identifier — valid and adversarial."""
from poc.models import Label, Observability
from poc.observability import build_bundle
from poc.scenarios import by_id
from poc.simulator import simulate
from poc.solver import solve
from poc import validators as V

FULL, MIN = Observability.FULL, Observability.MINIMAL


def _b(sid, level):
    b = by_id()[sid]
    bundle = build_bundle(simulate(b.scenario), level, set(b.spec.partial_resolution_drop))
    return bundle, solve(bundle, b.spec.probe_source_id)


def test_present_strong_id_share_real_field_names_are_valid():
    bundle, ref = _b("S01", MIN)
    refs = [{"artifact": "source_records", "match": {"source_id": "S01-P", "registration_number": "07711234"}},
            {"artifact": "final_dataset", "match": {"final_id": "ENT-01", "strong_id": "registry:UK/07711234"}}]
    assert V.status_evidence_validity(bundle, ref, Label.PRESENT_UNDER_OTHER_ID, refs)
    assert V.identifier_evidence_validity(bundle, ref, Label.PRESENT_UNDER_OTHER_ID, "ENT-01", None, refs)
    assert V.full_evidence_validity(bundle, ref, Label.PRESENT_UNDER_OTHER_ID, "ENT-01", None, refs)


def test_present_duns_field_is_valid():
    bundle, ref = _b("S02", MIN)
    refs = [{"artifact": "source_records", "match": {"source_id": "S02-P", "duns": "11-2200-345"}},
            {"artifact": "final_dataset", "match": {"final_id": "ENT-02", "strong_id": "duns/11-2200-345"}}]
    assert V.status_evidence_validity(bundle, ref, Label.PRESENT_UNDER_OTHER_ID, refs)


def test_name_only_citation_is_insufficient():
    bundle, ref = _b("S01", MIN)
    refs = [{"artifact": "final_dataset", "match": {"canonical_name": "Halcyon Foods"}}]
    assert not V.status_evidence_validity(bundle, ref, Label.PRESENT_UNDER_OTHER_ID, refs)


def test_missing_artifact_citation_fails_integrity():
    bundle, ref = _b("S03", MIN)  # resolution_map MISSING at minimal
    refs = [{"artifact": "resolution_map", "match": {"source_id": "S03-P", "canonical_id": "C3"}}]
    assert not V.citation_integrity(bundle, refs)


def test_wrong_merge_canonical_id_fails_identifier_but_public_id_passes():
    bundle, ref = _b("S13", FULL)
    refs = [{"artifact": "merge_decision_log", "match": {"canonical_id": "C13"}},
            {"artifact": "final_dataset", "match": {"final_id": "ENT-13", "canonical_id": "C13"}}]
    assert V.status_evidence_validity(bundle, ref, Label.WRONG_MERGE, refs)
    assert not V.identifier_evidence_validity(bundle, ref, Label.WRONG_MERGE, None, "C13", refs)
    assert V.identifier_evidence_validity(bundle, ref, Label.WRONG_MERGE, None, "ENT-13", refs)


def test_wrong_merge_target_null_valid_bogus_target_invalid():
    bundle, ref = _b("S08", FULL)
    refs = [{"artifact": "merge_decision_log", "match": {"canonical_id": "C8"}},
            {"artifact": "source_records", "match": {"source_id": "S08-P", "strong_id": "registry:US-MA/7712300"}}]
    assert V.identifier_evidence_validity(bundle, ref, Label.WRONG_MERGE, None, None, refs)
    assert not V.identifier_evidence_validity(bundle, ref, Label.WRONG_MERGE, None, "ENT-08C", refs)


def test_ingestion_loss_completeness_citation_valid():
    bundle, ref = _b("S04", FULL)
    good = [{"artifact": "ingestion_manifest", "match": {"guaranteed_complete_for": "B1"}}]
    assert V.status_evidence_validity(bundle, ref, Label.INGESTION_LOSS, good)
    bogus = [{"artifact": "ingestion_manifest", "match": {"guaranteed_complete_for": "B9"}}]
    assert not V.citation_integrity(bundle, bogus)


def test_abstention_needs_no_citation():
    bundle, ref = _b("S09", MIN)
    assert V.status_evidence_validity(bundle, ref, Label.UNDETERMINABLE, [])
    assert V.full_evidence_validity(bundle, ref, Label.UNDETERMINABLE, None, None, [])
