from poc.models import DirectorySource, NewsSource, RegistrySource, StrongId


def test_strong_id_key():
    assert StrongId(namespace="registry:US-DE", value="4830012").key() == "registry:US-DE/4830012"


def test_registry_strong_ids():
    r = RegistrySource(source_id="X", batch="B1", legal_name="Acme", registration_number="123", jurisdiction="US-DE")
    assert r.strong_ids() == [StrongId(namespace="registry:US-DE", value="123")]


def test_directory_strong_ids_optional():
    with_duns = DirectorySource(source_id="X", batch="B3", business_name="Acme", duns="9-9", category="c", city="x")
    without = DirectorySource(source_id="Y", batch="B3", business_name="Acme", duns=None, category="c", city="x")
    assert with_duns.strong_ids() == [StrongId(namespace="duns", value="9-9")]
    assert without.strong_ids() == []


def test_news_has_no_strong_id():
    n = NewsSource(source_id="X", batch="B2", headline="h", body="b", org_mention="Acme", published="2026-01-01")
    assert n.strong_ids() == []
