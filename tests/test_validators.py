"""Official evidence validator: integrity, sufficiency, identifier — valid and adversarial."""
from poc import validators as V
from poc.models import Label, Observability, Prediction
from poc.observability import build_bundle
from poc.scenarios import by_id
from poc.simulator import simulate
from poc.solver import solve

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
    assert V.identifier_evidence_validity(bundle, ref, "ENT-01", None, refs)
    pred = Prediction(source_id="S01-P", status=Label.PRESENT_UNDER_OTHER_ID, final_id="ENT-01")
    assert V.chain_validity(bundle, ref, pred, refs)["full_evidence_validity"]


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
    assert not V.identifier_evidence_validity(bundle, ref, None, "C13", refs)
    assert V.identifier_evidence_validity(bundle, ref, None, "ENT-13", refs)


def test_wrong_merge_target_null_valid_bogus_target_invalid():
    bundle, ref = _b("S08", FULL)
    refs = [{"artifact": "merge_decision_log", "match": {"canonical_id": "C8"}},
            {"artifact": "source_records", "match": {"source_id": "S08-P", "strong_id": "registry:US-MA/7712300"}}]
    assert V.identifier_evidence_validity(bundle, ref, None, None, refs)
    assert not V.identifier_evidence_validity(bundle, ref, None, "ENT-08C", refs)


def test_ingestion_loss_completeness_citation_valid():
    bundle, ref = _b("S04", FULL)
    good = [{"artifact": "ingestion_manifest", "match": {"guaranteed_complete_for": "B1"}}]
    assert V.status_evidence_validity(bundle, ref, Label.INGESTION_LOSS, good)
    bogus = [{"artifact": "ingestion_manifest", "match": {"guaranteed_complete_for": "B9"}}]
    assert not V.citation_integrity(bundle, bogus)


def test_abstention_needs_no_citation():
    bundle, ref = _b("S09", MIN)
    assert V.status_evidence_validity(bundle, ref, Label.UNDETERMINABLE, [])
    pred = Prediction(source_id="S09-P", status=Label.UNDETERMINABLE)
    assert V.chain_validity(bundle, ref, pred, [])["full_evidence_validity"]


def test_boolean_cited_as_json_string_matches_but_wrong_value_does_not():
    bundle, _ = _b("S06", FULL)
    ok = [{"artifact": "filter_log", "match": {"canonical_id": "C6", "kept": "false"}}]
    wrong = [{"artifact": "filter_log", "match": {"canonical_id": "C6", "kept": "true"}}]
    assert V.citation_integrity(bundle, ok)
    assert not V.citation_integrity(bundle, wrong)


def test_final_strong_id_cited_by_its_text_or_parts_matches_the_real_row_only():
    bundle, ref = _b("S14", FULL)
    as_text = {"artifact": "final_dataset", "match": {"final_id": "ENT-14", "strong_ids": "registry:US-TX/3312345"}}
    as_parts = {"artifact": "final_dataset",
                "match": {"final_id": "ENT-14", "namespace": "registry:US-TX", "value": "3312345"}}
    bogus = {"artifact": "final_dataset",
             "match": {"final_id": "ENT-14", "namespace": "registry:US-TX", "value": "9999999"}}
    assert V.citation_integrity(bundle, [as_text]) and V.citation_integrity(bundle, [as_parts])
    assert not V.citation_integrity(bundle, [bogus])
    src = {"artifact": "source_records", "match": {"source_id": "S14-P", "registration_number": "3312345"}}
    assert V.status_evidence_validity(bundle, ref, Label.PRESENT_UNDER_OTHER_ID, [src, as_parts])
