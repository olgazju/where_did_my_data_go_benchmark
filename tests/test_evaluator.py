import json

from poc.evaluator import evaluate_one, parse_prediction
from poc.models import Label, Observability, ReferenceAnswer

SID = "SRC-1"


def _ref(status, **kw):
    return ReferenceAnswer(source_id=SID, level=Observability.FULL, status=status, **kw)


def _ans(status, source_id=SID, final_id=None, merge_target_id=None, extra=""):
    d = {"source_id": source_id, "status": status, "final_id": final_id, "merge_target_id": merge_target_id}
    body = json.dumps(d)
    return body if not extra else body[:-1] + f',{extra}}}'


# ------------------------------------------------------------------ parsing
def test_parse_tolerates_prose_and_fences():
    raw = 'Sure!\n```json\n{"source_id":"SRC-1","status":"INGESTION_LOSS","final_id":null,"merge_target_id":null}\n```'
    res = parse_prediction(raw)
    assert res.ok and res.prediction.status is Label.INGESTION_LOSS


def test_missing_source_id_is_malformed():
    res = parse_prediction('{"status":"INGESTION_LOSS"}')
    assert not res.ok and "source_id" in res.error


def test_invalid_status_is_malformed():
    res = parse_prediction('{"source_id":"SRC-1","status":"BANANA"}')
    assert not res.ok


def test_non_object_json_is_malformed():
    assert not parse_prediction('["SRC-1","INGESTION_LOSS"]').ok
    assert not parse_prediction("no json here").ok


# ------------------------------------------------------------------ source id validation
def test_source_id_mismatch_is_malformed():
    rec = evaluate_one("t", _ref(Label.INGESTION_LOSS), _ans("INGESTION_LOSS", source_id="SRC-999"))
    assert rec.malformed and "mismatch" in rec.error


# ------------------------------------------------------------------ status-specific ids
def test_present_requires_matching_final_id():
    ref = _ref(Label.PRESENT_UNDER_OTHER_ID, final_id="ENT-501")
    good = evaluate_one("t", ref, _ans("PRESENT_UNDER_OTHER_ID", final_id="ENT-501"))
    bad = evaluate_one("t", ref, _ans("PRESENT_UNDER_OTHER_ID", final_id="ENT-999"))
    missing = evaluate_one("t", ref, _ans("PRESENT_UNDER_OTHER_ID", final_id=None))
    assert good.full_correct
    assert bad.status_correct and not bad.full_correct
    assert missing.status_correct and not missing.full_correct  # required id absent


def test_present_with_stray_merge_target_is_not_full_correct():
    ref = _ref(Label.PRESENT_UNDER_OTHER_ID, final_id="ENT-501")
    rec = evaluate_one("t", ref, _ans("PRESENT_UNDER_OTHER_ID", final_id="ENT-501", merge_target_id="ENT-7"))
    assert rec.status_correct and not rec.full_correct  # non-applicable field not null


def test_wrong_merge_target_required_only_when_observable():
    observable = _ref(Label.WRONG_MERGE, merge_target_id="ENT-502", merge_target_observable=True)
    ok = evaluate_one("t", observable, _ans("WRONG_MERGE", merge_target_id="ENT-502"))
    missing = evaluate_one("t", observable, _ans("WRONG_MERGE", merge_target_id=None))
    assert ok.full_correct
    assert missing.status_correct and not missing.full_correct

    hidden = _ref(Label.WRONG_MERGE, merge_target_id=None, merge_target_observable=False)
    null_ok = evaluate_one("t", hidden, _ans("WRONG_MERGE", merge_target_id=None))
    guessed = evaluate_one("t", hidden, _ans("WRONG_MERGE", merge_target_id="ENT-9"))
    assert null_ok.full_correct  # not required when unobservable
    assert not guessed.full_correct  # asserting a target without evidence is a guess


def test_wrong_merge_with_stray_final_id_is_not_full_correct():
    ref = _ref(Label.WRONG_MERGE, merge_target_id="ENT-502", merge_target_observable=True)
    rec = evaluate_one("t", ref, _ans("WRONG_MERGE", final_id="ENT-3", merge_target_id="ENT-502"))
    assert rec.status_correct and not rec.full_correct


def test_non_applicable_ids_must_be_null():
    for status in ("INGESTION_LOSS", "EXTRACTION_LOSS", "FILTERED_OUT", "UNDETERMINABLE"):
        ref = _ref(Label(status))
        clean = evaluate_one("t", ref, _ans(status))
        dirty = evaluate_one("t", ref, _ans(status, final_id="ENT-1"))
        assert clean.full_correct
        assert dirty.status_correct and not dirty.full_correct


def test_extra_unknown_fields_ignored():
    rec = evaluate_one("t", _ref(Label.INGESTION_LOSS), _ans("INGESTION_LOSS", extra='"confidence":0.9'))
    assert rec.status_correct and rec.full_correct


# ------------------------------------------------------------------ malformed tracking
def test_malformed_tracked_separately():
    rec = evaluate_one("t", _ref(Label.INGESTION_LOSS), "not json at all")
    assert rec.malformed and not rec.status_correct and not rec.full_correct
