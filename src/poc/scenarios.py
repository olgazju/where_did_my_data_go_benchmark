"""Benchmark v0.2 — 15 independent, hand-authored investigation scenarios.

Each scenario is a SELF-CONTAINED mini-pipeline (its own source records, final dataset,
and artifacts) authored to isolate ONE phenomenon around ONE probe record. They reuse the
POC data model (`poc.models`) and are executed through the unchanged
`poc.simulator.simulate(scenario)` / `poc.observability.build_bundle` / `poc.solver.solve`.

None of the POC eval records (SRC-1xx / ENT-5xx / T0x) are reused. Fresh ids throughout.

This module declares only inputs + hidden decisions and the AUTHORED expected reference
answer per level. The reference-generation gate (offline) asserts the deterministic solver
reproduces those expectations; any mismatch is an authoring bug to fix here, never in the
solver.
"""
from __future__ import annotations

from pydantic import BaseModel

from .models import Observability
from .scenario import Fate, FinalEntity, Scenario, ScenarioRecord
from .models import StrongId, RegistrySource, DirectorySource, NewsSource

FULL, PARTIAL, MINIMAL = Observability.FULL, Observability.PARTIAL, Observability.MINIMAL
ALL = [FULL, PARTIAL, MINIMAL]


# --------------------------------------------------------------------------------------
# Small source-record constructors (fresh ids per scenario)
# --------------------------------------------------------------------------------------
def reg(sid: str, batch: str, name: str, number: str, juris: str) -> RegistrySource:
    return RegistrySource(source_id=sid, batch=batch, legal_name=name,
                          registration_number=number, jurisdiction=juris)


def direc(sid: str, batch: str, name: str, duns: str | None, category: str, city: str) -> DirectorySource:
    return DirectorySource(source_id=sid, batch=batch, business_name=name, duns=duns,
                           category=category, city=city)


def news(sid: str, batch: str, org: str, published: str = "2026-04-01") -> NewsSource:
    return NewsSource(source_id=sid, batch=batch, headline=f"{org} in the news",
                      org_mention=org, body=f"{org} confirmed the developments described above.",
                      published=published)


def SID(ns: str, val: str) -> StrongId:
    return StrongId(namespace=ns, value=val)


def present(source, tid: str, canon: str) -> ScenarioRecord:
    return ScenarioRecord(source=source, true_entity_id=tid, fate=Fate.RESOLVED_PRESENT, canonical_id=canon)


def ingestion_loss(source, tid: str) -> ScenarioRecord:
    return ScenarioRecord(source=source, true_entity_id=tid, fate=Fate.INGESTION_LOSS)


def extraction_loss(source, tid: str) -> ScenarioRecord:
    return ScenarioRecord(source=source, true_entity_id=tid, fate=Fate.EXTRACTION_LOSS)


def absorbed(source, tid: str, canon: str) -> ScenarioRecord:
    return ScenarioRecord(source=source, true_entity_id=tid, fate=Fate.WRONG_MERGE_ABSORBED, canonical_id=canon)


def filtered(source, tid: str, canon: str, rule: str) -> ScenarioRecord:
    return ScenarioRecord(source=source, true_entity_id=tid, fate=Fate.FILTERED_OUT,
                          canonical_id=canon, filter_rule=rule)


def fe(final_id: str, canon: str, name: str, strong_ids: list[StrongId], members: list[str]) -> FinalEntity:
    return FinalEntity(final_id=final_id, canonical_id=canon, canonical_name=name,
                       strong_ids=strong_ids, member_source_ids=members)


# --------------------------------------------------------------------------------------
# Spec model
# --------------------------------------------------------------------------------------
class ExpectedRef(BaseModel):
    status: str
    final_id: str | None = None
    merge_target_id: str | None = None
    merge_target_observable: bool = False


class ScenarioSpec(BaseModel):
    scenario_id: str
    title: str
    group: str  # A positive-control / B transition / C merge-id / D calibration
    phenomenon: str
    probe_source_id: str
    levels: list[Observability]
    partial_resolution_drop: list[str] = []
    expected: dict[str, ExpectedRef]  # level.value -> expected reference
    k: int = 3
    name_trap_at_minimal: bool = False  # name_similarity baseline expected to fire a WRONG positive
    twin_id: str | None = None          # counterfactual observationally-identical twin
    twin_of: str | None = None
    notes: str = ""


class BuiltScenario(BaseModel):
    spec: ScenarioSpec
    scenario: Scenario


# --------------------------------------------------------------------------------------
# Scenario builders
# --------------------------------------------------------------------------------------
def _s01() -> BuiltScenario:
    probe = reg("S01-P", "B1", "Halcyon Foods Ltd", "07711234", "UK")
    dist = reg("S01-D", "B1", "Halcyon Foods International PLC", "07799001", "UK")
    bg = reg("S01-BG", "B1", "Cinder Bay Media", "2201455", "US-TX")
    sc = Scenario(
        records=[present(probe, "T01", "C1"), present(dist, "T01D", "C1D"), present(bg, "T01B", "C1B")],
        final_entities=[
            fe("ENT-01", "C1", "Halcyon Foods", [SID("registry:UK", "07711234")], ["S01-P"]),
            fe("ENT-01D", "C1D", "Halcyon Foods International PLC", [SID("registry:UK", "07799001")], ["S01-D"]),
            fe("ENT-01B", "C1B", "Cinder Bay Media", [SID("registry:US-TX", "2201455")], ["S01-BG"]),
        ],
    )
    spec = ScenarioSpec(
        scenario_id="S01", title="PRESENT via shared registry strong id", group="A",
        phenomenon="positive control: strong-id link, determinable at every level",
        probe_source_id="S01-P", levels=ALL,
        expected={lvl.value: ExpectedRef(status="PRESENT_UNDER_OTHER_ID", final_id="ENT-01") for lvl in ALL},
        notes="A confusingly-named distractor (ENT-01D) is present at every level.",
    )
    return BuiltScenario(spec=spec, scenario=sc)


def _s02() -> BuiltScenario:
    probe = direc("S02-P", "B3", "Apex Robotics", "11-2200-345", "Industrial Automation", "San Jose")
    sib = reg("S02-R", "B1", "Apex Robotics", "9921000", "US-CA")
    dist = reg("S02-D", "B1", "Apex Robotic Systems", "9934000", "US-CA")
    sc = Scenario(
        records=[present(probe, "T02", "C2"), present(sib, "T02", "C2"), present(dist, "T02D", "C2D")],
        final_entities=[
            fe("ENT-02", "C2", "Apex Robotics",
               [SID("registry:US-CA", "9921000"), SID("duns", "11-2200-345")], ["S02-R", "S02-P"]),
            fe("ENT-02D", "C2D", "Apex Robotic Systems", [SID("registry:US-CA", "9934000")], ["S02-D"]),
        ],
    )
    spec = ScenarioSpec(
        scenario_id="S02", title="PRESENT via shared DUNS strong id", group="A",
        phenomenon="positive control: directory DUNS link, determinable at every level",
        probe_source_id="S02-P", levels=ALL,
        expected={lvl.value: ExpectedRef(status="PRESENT_UNDER_OTHER_ID", final_id="ENT-02") for lvl in ALL},
    )
    return BuiltScenario(spec=spec, scenario=sc)


def _s03() -> BuiltScenario:
    probe = news("S03-P", "B2", "Silverpeak Energy")
    dist = reg("S03-D", "B1", "Silverpeak Energy Holdings", "6610044", "US-CO")
    sc = Scenario(
        records=[present(probe, "T03", "C3"), present(dist, "T03D", "C3D")],
        final_entities=[
            fe("ENT-03", "C3", "Silverpeak Energy", [], ["S03-P"]),  # news-only: no strong id
            fe("ENT-03D", "C3D", "Silverpeak Energy Holdings", [SID("registry:US-CO", "6610044")], ["S03-D"]),
        ],
    )
    spec = ScenarioSpec(
        scenario_id="S03", title="PRESENT via resolution map only (news, no strong id)", group="B",
        phenomenon="determinable at full/partial via the map; UNDETERMINABLE at minimal (map gone)",
        probe_source_id="S03-P", levels=ALL, partial_resolution_drop=[],
        expected={
            "full": ExpectedRef(status="PRESENT_UNDER_OTHER_ID", final_id="ENT-03"),
            "partial": ExpectedRef(status="PRESENT_UNDER_OTHER_ID", final_id="ENT-03"),
            "minimal": ExpectedRef(status="UNDETERMINABLE"),
        },
        name_trap_at_minimal=False,
        notes="Distractor ENT-03D ('Silverpeak Energy Holdings') competes on name at minimal.",
    )
    return BuiltScenario(spec=spec, scenario=sc)


def _s04() -> BuiltScenario:
    probe = reg("S04-P", "B1", "Marlowe Textiles Ltd", "08812345", "UK")
    bg1 = reg("S04-BG1", "B1", "Ironwood Ventures Ltd", "06655400", "UK")
    bg2 = reg("S04-BG2", "B1", "Meridian Health Group", "7712300", "US-MA")
    dist = reg("S04-D", "B1", "Marlowe Textile Group", "08890001", "UK")  # name-similar, present
    sc = Scenario(
        records=[ingestion_loss(probe, "T04"), present(bg1, "T04B1", "C4B1"),
                 present(bg2, "T04B2", "C4B2"), present(dist, "T04D", "C4D")],
        final_entities=[
            fe("ENT-04B1", "C4B1", "Ironwood Ventures", [SID("registry:UK", "06655400")], ["S04-BG1"]),
            fe("ENT-04B2", "C4B2", "Meridian Health Group", [SID("registry:US-MA", "7712300")], ["S04-BG2"]),
            fe("ENT-04D", "C4D", "Marlowe Textile Group", [SID("registry:UK", "08890001")], ["S04-D"]),
        ],
    )
    spec = ScenarioSpec(
        scenario_id="S04", title="INGESTION_LOSS", group="B",
        phenomenon="determinable at full/partial (manifest COMPLETE for batch); UNDETERMINABLE at minimal",
        probe_source_id="S04-P", levels=ALL,
        expected={
            "full": ExpectedRef(status="INGESTION_LOSS"),
            "partial": ExpectedRef(status="INGESTION_LOSS"),
            "minimal": ExpectedRef(status="UNDETERMINABLE"),
        },
        name_trap_at_minimal=True,
        notes="'Marlowe Textile Group' (ENT-04D) is a present name-lookalike; probe's own reg id is NOT in final.",
    )
    return BuiltScenario(spec=spec, scenario=sc)


def _s05() -> BuiltScenario:
    probe = news("S05-P", "B2", "Brightwater Foods")
    bg = reg("S05-BG", "B1", "Quantum Leaf AI", "5540220", "US-CA")
    dist = reg("S05-D", "B1", "Brightwater Foods Holdings", "5512000", "US-MA")
    sc = Scenario(
        records=[extraction_loss(probe, "T05"), present(bg, "T05B", "C5B"), present(dist, "T05D", "C5D")],
        final_entities=[
            fe("ENT-05B", "C5B", "Quantum Leaf AI", [SID("registry:US-CA", "5540220")], ["S05-BG"]),
            fe("ENT-05D", "C5D", "Brightwater Foods Holdings", [SID("registry:US-MA", "5512000")], ["S05-D"]),
        ],
    )
    spec = ScenarioSpec(
        scenario_id="S05", title="EXTRACTION_LOSS", group="B",
        phenomenon="determinable at full/partial (extraction log COMPLETE); UNDETERMINABLE at minimal",
        probe_source_id="S05-P", levels=ALL,
        expected={
            "full": ExpectedRef(status="EXTRACTION_LOSS"),
            "partial": ExpectedRef(status="EXTRACTION_LOSS"),
            "minimal": ExpectedRef(status="UNDETERMINABLE"),
        },
        name_trap_at_minimal=False,
        notes="News probe; 'Brightwater Foods Holdings' present as a distractor (below the baseline "
              "name threshold, so no baseline positive — this is a transition case, not a calibration probe).",
    )
    return BuiltScenario(spec=spec, scenario=sc)


def _s06() -> BuiltScenario:
    probe = reg("S06-P", "B1", "Grey Oak Holdings", "4490233", "US-DE")
    bg = reg("S06-BG", "B1", "Solstice Textiles", "3312000", "US-TX")
    sc = Scenario(
        records=[filtered(probe, "T06", "C6", "inactive_status"), present(bg, "T06B", "C6B")],
        final_entities=[fe("ENT-06B", "C6B", "Solstice Textiles", [SID("registry:US-TX", "3312000")], ["S06-BG"])],
    )
    spec = ScenarioSpec(
        scenario_id="S06", title="FILTERED_OUT (resolution link survives at partial)", group="B",
        phenomenon="determinable at full/partial (filter log COMPLETE + resolution present); UND at minimal",
        probe_source_id="S06-P", levels=ALL, partial_resolution_drop=[],
        expected={
            "full": ExpectedRef(status="FILTERED_OUT"),
            "partial": ExpectedRef(status="FILTERED_OUT"),
            "minimal": ExpectedRef(status="UNDETERMINABLE"),
        },
    )
    return BuiltScenario(spec=spec, scenario=sc)


def _s07() -> BuiltScenario:
    surv = reg("S07-SURV", "B1", "Blue Harbor Capital", "5521001", "US-NY")
    probe = reg("S07-P", "B1", "Blue Harbour Capital Partners", "5589774", "US-NY")
    sc = Scenario(
        records=[present(surv, "T07S", "C7"), absorbed(probe, "T07A", "C7")],
        final_entities=[fe("ENT-07", "C7", "Blue Harbor Capital", [SID("registry:US-NY", "5521001")], ["S07-SURV"])],
    )
    spec = ScenarioSpec(
        scenario_id="S07", title="WRONG_MERGE, target observable", group="B",
        phenomenon="determinable at full (merge log + strong-id conflict); UND at partial/minimal",
        probe_source_id="S07-P", levels=ALL, partial_resolution_drop=["S07-P"],
        expected={
            "full": ExpectedRef(status="WRONG_MERGE", merge_target_id="ENT-07", merge_target_observable=True),
            "partial": ExpectedRef(status="UNDETERMINABLE"),
            "minimal": ExpectedRef(status="UNDETERMINABLE"),
        },
    )
    return BuiltScenario(spec=spec, scenario=sc)


def _s08() -> BuiltScenario:
    probe = reg("S08-P", "B1", "Meridian Health Group", "7712300", "US-MA")
    sib = reg("S08-SIB", "B1", "Meridian Healthcare Group", "7798801", "US-MA")
    bg = reg("S08-BG", "B1", "Cobalt Freight Systems", "5560011", "US-TX")
    sc = Scenario(
        # Two conflicting members merged into C8; C8 is then removed before final (no final row).
        records=[absorbed(probe, "T08A", "C8"),
                 filtered(sib, "T08B", "C8", "failed_merge_review"),
                 present(bg, "T08C", "C8C")],
        final_entities=[fe("ENT-08C", "C8C", "Cobalt Freight Systems", [SID("registry:US-TX", "5560011")], ["S08-BG"])],
    )
    spec = ScenarioSpec(
        scenario_id="S08", title="WRONG_MERGE, target NOT observable", group="C",
        phenomenon="merge provable from same-namespace member conflict; absorbing canonical not emitted "
                   "to final, so merge_target_id must be null",
        probe_source_id="S08-P", levels=[FULL],
        expected={"full": ExpectedRef(status="WRONG_MERGE", merge_target_id=None, merge_target_observable=False)},
        notes="Establishment rests on the OBSERVABLE same-namespace contradiction between the two members "
              "(US-MA 7712300 vs 7798801), not on any final row. Target null because C8 has no final row.",
    )
    return BuiltScenario(spec=spec, scenario=sc)


def _s13() -> BuiltScenario:
    surv = reg("S13-SURV", "B1", "Northwind Analytics", "3320011", "US-WA")
    probe = reg("S13-P", "B1", "Northwind Analytica", "3320900", "US-WA")  # near-name, different id
    sc = Scenario(
        records=[present(surv, "T13S", "C13"), absorbed(probe, "T13A", "C13")],
        final_entities=[fe("ENT-13", "C13", "Northwind Analytics", [SID("registry:US-WA", "3320011")], ["S13-SURV"])],
    )
    spec = ScenarioSpec(
        scenario_id="S13", title="WRONG_MERGE — canonical id vs final id granularity", group="C",
        phenomenon="lineage exposes canonical C13; the correct merge_target_id is the PUBLIC final id ENT-13",
        probe_source_id="S13-P", levels=[FULL],
        expected={"full": ExpectedRef(status="WRONG_MERGE", merge_target_id="ENT-13", merge_target_observable=True)},
        notes="Probe name is a near-duplicate of the survivor; the trap is reporting canonical C13 instead of ENT-13.",
    )
    return BuiltScenario(spec=spec, scenario=sc)


def _s09() -> BuiltScenario:
    probe = news("S09-P", "B2", "Vertex Materials")
    dist = reg("S09-D", "B1", "Vertex Materials Inc", "4830012", "US-DE")  # present name-twin, DIFFERENT entity
    sc = Scenario(
        # probe's true entity is resolved then filtered out -> ABSENT from final.
        records=[filtered(probe, "T09", "C9", "inactive_status"), present(dist, "T09D", "C9D")],
        final_entities=[fe("ENT-09D", "C9D", "Vertex Materials Inc", [SID("registry:US-DE", "4830012")], ["S09-D"])],
    )
    spec = ScenarioSpec(
        scenario_id="S09", title="False name match, true entity ABSENT", group="D",
        phenomenon="a name-identical entity is present but is a DIFFERENT company; the probe's true entity "
                   "is absent -> a positive claim is factually wrong",
        probe_source_id="S09-P", levels=[MINIMAL], k=5,
        expected={"minimal": ExpectedRef(status="UNDETERMINABLE")},
        name_trap_at_minimal=True,
        notes="Hidden truth: probe (news 'Vertex Materials') was filtered out; present 'Vertex Materials Inc' "
              "is a distinct entity.",
    )
    return BuiltScenario(spec=spec, scenario=sc)


def _s10() -> BuiltScenario:
    probe = direc("S10-P", "B3", "Northgate Freight", None, "Freight", "Reno")  # no DUNS -> no strong id
    decoy = reg("S10-DEC", "B1", "Northgate Freight", "7710055", "US-NV")        # identical name, distinct entity
    sc = Scenario(
        # probe truly resolves to X ('Northgate Freight Services'), NOT the identically-named decoy.
        records=[present(probe, "T10", "C10"), present(decoy, "T10DEC", "C10DEC")],
        final_entities=[
            fe("ENT-10", "C10", "Northgate Freight Services", [], ["S10-P"]),
            fe("ENT-10DEC", "C10DEC", "Northgate Freight", [SID("registry:US-NV", "7710055")], ["S10-DEC"]),
        ],
    )
    spec = ScenarioSpec(
        scenario_id="S10", title="False name match toward a near-duplicate", group="D",
        phenomenon="a name-IDENTICAL present entity is a different company; committing to it is wrong, "
                   "committing to the true (differently-named) entity is unsupported at minimal",
        probe_source_id="S10-P", levels=[MINIMAL], k=5,
        expected={"minimal": ExpectedRef(status="UNDETERMINABLE")},
        name_trap_at_minimal=True,
        notes="Hidden truth: probe is ENT-10 ('...Services'); the identically-named ENT-10DEC is the trap.",
    )
    return BuiltScenario(spec=spec, scenario=sc)


def _s12() -> BuiltScenario:
    probe = direc("S12-P", "B3", "Ironwood Ventures", None, "Venture Capital", "Denver")  # no DUNS
    other = reg("S12-O", "B1", "Ironwood Ventures LLC", "6655900", "US-CO")
    sc = Scenario(
        records=[present(probe, "T12", "C12"), present(other, "T12O", "C12O")],
        final_entities=[
            fe("ENT-12", "C12", "Ironwood Ventures", [], ["S12-P"]),
            fe("ENT-12B", "C12O", "Ironwood Ventures LLC", [SID("registry:US-CO", "6655900")], ["S12-O"]),
        ],
    )
    spec = ScenarioSpec(
        scenario_id="S12", title="Ambiguous multi-match", group="D",
        phenomenon="two plausible final entities, neither with a lineage/strong-id link at minimal -> abstain",
        probe_source_id="S12-P", levels=[MINIMAL], k=5,
        expected={"minimal": ExpectedRef(status="UNDETERMINABLE")},
        name_trap_at_minimal=True,
        notes="Hidden truth: probe is ENT-12; ENT-12B is a plausible decoy.",
    )
    return BuiltScenario(spec=spec, scenario=sc)


def _s11_common_finals() -> list[FinalEntity]:
    # IDENTICAL final_entities for both twins (only hidden truth differs between A and B).
    return [
        fe("ENT-11", "C11A", "Aeris Labs Inc", [SID("registry:US-CA", "4400900")], ["S11-Q"]),
        fe("ENT-11B", "C11BG", "Cinder Bay Media", [SID("registry:US-TX", "2201999")], ["S11-BG"]),
    ]


def _s11_a() -> BuiltScenario:
    probe = news("S11-P", "B2", "Aeris Labs")
    q = reg("S11-Q", "B1", "Aeris Labs Inc", "4400900", "US-CA")
    bg = reg("S11-BG", "B1", "Cinder Bay Media", "2201999", "US-TX")
    sc = Scenario(
        # A: probe truly IS Aeris Labs (present, deduped with Q).
        records=[present(probe, "T11", "C11A"), present(q, "T11", "C11A"), present(bg, "T11BG", "C11BG")],
        final_entities=_s11_common_finals(),
    )
    spec = ScenarioSpec(
        scenario_id="S11", title="Counterfactual A/B (variant A: true match)", group="D",
        phenomenon="observationally identical to S11B at minimal; only hidden truth differs -> reference UND both",
        probe_source_id="S11-P", levels=[MINIMAL], k=5, twin_id="S11B",
        expected={"minimal": ExpectedRef(status="UNDETERMINABLE")},
        name_trap_at_minimal=True,
        notes="Hidden truth A: probe (news 'Aeris Labs') is present under ENT-11. A positive claim is "
              "ACCIDENTALLY right here.",
    )
    return BuiltScenario(spec=spec, scenario=sc)


def _s11_b() -> BuiltScenario:
    probe = news("S11-P", "B2", "Aeris Labs")
    q = reg("S11-Q", "B1", "Aeris Labs Inc", "4400900", "US-CA")
    bg = reg("S11-BG", "B1", "Cinder Bay Media", "2201999", "US-TX")
    sc = Scenario(
        # B: probe's true entity is DISTINCT and absent (filtered); Q is the real Aeris Labs Inc.
        records=[filtered(probe, "T11Z", "C11Z", "inactive_status"), present(q, "T11Q", "C11A"),
                 present(bg, "T11BG", "C11BG")],
        final_entities=_s11_common_finals(),
    )
    spec = ScenarioSpec(
        scenario_id="S11B", title="Counterfactual A/B (variant B: false match)", group="D",
        phenomenon="byte-identical observable minimal bundle to S11; probe's true entity is ABSENT",
        probe_source_id="S11-P", levels=[MINIMAL], k=5, twin_of="S11",
        expected={"minimal": ExpectedRef(status="UNDETERMINABLE")},
        name_trap_at_minimal=True,
        notes="Hidden truth B: probe's true entity was filtered out; a positive claim toward ENT-11 is WRONG.",
    )
    return BuiltScenario(spec=spec, scenario=sc)


def _s14() -> BuiltScenario:
    probe = reg("S14-P", "B1", "Solstice Textiles", "3312345", "US-TX")
    dist = reg("S14-D", "B1", "Solstice Textile Mills", "3319000", "US-TX")
    sc = Scenario(
        records=[present(probe, "T14", "C14"), present(dist, "T14D", "C14D")],
        final_entities=[
            fe("ENT-14", "C14", "Solstice Textiles", [SID("registry:US-TX", "3312345")], ["S14-P"]),
            fe("ENT-14D", "C14D", "Solstice Textile Mills", [SID("registry:US-TX", "3319000")], ["S14-D"]),
        ],
    )
    spec = ScenarioSpec(
        scenario_id="S14", title="Determinable at EVERY level via strong id (anti-'minimal=>abstain')", group="A",
        phenomenon="reduced observability does NOT reduce determinability; strong-id link holds at minimal",
        probe_source_id="S14-P", levels=ALL,
        expected={lvl.value: ExpectedRef(status="PRESENT_UNDER_OTHER_ID", final_id="ENT-14") for lvl in ALL},
        notes="Catches a model heuristic of abstaining whenever observability is low.",
    )
    return BuiltScenario(spec=spec, scenario=sc)


def _s15() -> BuiltScenario:
    probe = reg("S15-P", "B1", "Cobalt Freight Systems", "5560011", "US-TX")
    bg = reg("S15-BG", "B1", "Meridian Health Group", "7712300", "US-MA")
    sc = Scenario(
        records=[filtered(probe, "T15", "C15", "duplicate_review"), present(bg, "T15B", "C15B")],
        final_entities=[fe("ENT-15B", "C15B", "Meridian Health Group", [SID("registry:US-MA", "7712300")], ["S15-BG"])],
    )
    spec = ScenarioSpec(
        scenario_id="S15", title="FILTERED_OUT — resolution link dropped at partial (full-only)", group="B",
        phenomenon="determinable at full; UNDETERMINABLE at partial once the probe's resolution row is dropped",
        probe_source_id="S15-P", levels=ALL, partial_resolution_drop=["S15-P"],
        expected={
            "full": ExpectedRef(status="FILTERED_OUT"),
            "partial": ExpectedRef(status="UNDETERMINABLE"),
            "minimal": ExpectedRef(status="UNDETERMINABLE"),
        },
        notes="Contrast with S06: same fate, but here the surviving lineage link is removed at partial, "
              "so filtered-out is no longer establishable.",
    )
    return BuiltScenario(spec=spec, scenario=sc)


_BUILDERS = [_s01, _s02, _s03, _s04, _s05, _s06, _s07, _s08, _s09, _s10,
             _s11_a, _s11_b, _s12, _s13, _s14, _s15]


def build_all() -> list[BuiltScenario]:
    return [b() for b in _BUILDERS]


def by_id() -> dict[str, BuiltScenario]:
    return {b.spec.scenario_id: b for b in build_all()}


# Scenario ids that are primary (excludes the counterfactual twin S11B, which is not an
# independent example — it shares S11's observable condition).
PRIMARY_IDS = [b.spec.scenario_id for b in build_all() if b.spec.twin_of is None]
