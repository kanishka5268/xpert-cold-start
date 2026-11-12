"""Unit tests for intermediate interpretation dataclasses."""

from dataclasses import fields, is_dataclass
from typing import get_args, get_origin, List, Optional, Union

import pytest

from src.interpretation import (
    InterpretedGoal,
    InterpretedOpenLoop,
    InterpretedOrganisation,
    InterpretedPerson,
)
from src.interpretation.records import (
    InterpretedGoal as RecordGoal,
    InterpretedOpenLoop as RecordOpenLoop,
    InterpretedOrganisation as RecordOrganisation,
    InterpretedPerson as RecordPerson,
)


def test_package_exports():
    """Verify that all four dataclasses are cleanly exported from src.interpretation."""
    assert InterpretedOpenLoop is RecordOpenLoop
    assert InterpretedGoal is RecordGoal
    assert InterpretedPerson is RecordPerson
    assert InterpretedOrganisation is RecordOrganisation


def test_successful_construction():
    """Verify successful instantiation of all four dataclasses with full valid parameters."""
    # 1. InterpretedOpenLoop
    loop = InterpretedOpenLoop(
        source_msg="M100",
        chat_id="C01",
        type="commitment",
        is_genuine=True,
        title="Send revised draft to Neha",
        owner="ME",
        target_person="P04",
        status="open",
        closed_by=None,
        due_date="2026-10-25",
        suggested_action="remind",
        confidence=0.92,
        validation_flags=["checked_context"],
    )
    assert loop.source_msg == "M100"
    assert loop.chat_id == "C01"
    assert loop.type == "commitment"
    assert loop.is_genuine is True
    assert loop.title == "Send revised draft to Neha"
    assert loop.owner == "ME"
    assert loop.target_person == "P04"
    assert loop.status == "open"
    assert loop.closed_by is None
    assert loop.due_date == "2026-10-25"
    assert loop.suggested_action == "remind"
    assert loop.confidence == 0.92
    assert loop.validation_flags == ["checked_context"]

    # 2. InterpretedGoal
    goal = InterpretedGoal(
        title="Run the half marathon in January",
        area="health",
        type="milestone",
        horizon="3mo",
        evidence_msgs=["M026", "M027"],
        is_genuine=True,
        confidence=0.88,
    )
    assert goal.title == "Run the half marathon in January"
    assert goal.area == "health"
    assert goal.type == "milestone"
    assert goal.horizon == "3mo"
    assert goal.evidence_msgs == ["M026", "M027"]
    assert goal.is_genuine is True
    assert goal.confidence == 0.88

    # 3. InterpretedPerson
    person = InterpretedPerson(
        person_id="P03",
        circle="staff",
        looks_after=["P04"],
        associated_org_names=["Brightside Design"],
        org_link_type="supports",
        evidence_msgs=["M010", "M012"],
        confidence=0.95,
    )
    assert person.person_id == "P03"
    assert person.circle == "staff"
    assert person.looks_after == ["P04"]
    assert person.associated_org_names == ["Brightside Design"]
    assert person.org_link_type == "supports"
    assert person.evidence_msgs == ["M010", "M012"]
    assert person.confidence == 0.95

    # 4. InterpretedOrganisation
    org = InterpretedOrganisation(
        name="Brightside Design",
        org_type="company",
        side="work",
        role="owner",
        evidence_msgs=["M001", "M005"],
        associated_chats=["C02", "C04"],
        confidence=0.90,
    )
    assert org.name == "Brightside Design"
    assert org.org_type == "company"
    assert org.side == "work"
    assert org.role == "owner"
    assert org.evidence_msgs == ["M001", "M005"]
    assert org.associated_chats == ["C02", "C04"]
    assert org.confidence == 0.90


def test_optional_fields_accept_none():
    """Verify that all optional fields in all dataclasses cleanly accept None values."""
    # InterpretedOpenLoop optional fields
    loop = InterpretedOpenLoop(
        source_msg="M101",
        chat_id="C02",
        type="question",
        is_genuine=False,
        title="Casual question",
        owner=None,
        target_person=None,
        status="dropped",
        closed_by=None,
        due_date=None,
        suggested_action=None,
        confidence=None,
        validation_flags=None,
    )
    assert loop.owner is None
    assert loop.target_person is None
    assert loop.closed_by is None
    assert loop.due_date is None
    assert loop.suggested_action is None
    assert loop.confidence is None
    assert loop.validation_flags is None

    # InterpretedGoal optional fields
    goal = InterpretedGoal(
        title="Tactical plan",
        area="work",
        type="habit",
        horizon="1yr",
        evidence_msgs=["M050"],
        is_genuine=False,
        confidence=None,
    )
    assert goal.confidence is None

    # InterpretedPerson optional fields
    person = InterpretedPerson(
        person_id="P99",
        circle=None,
        looks_after=[],
        associated_org_names=[],
        org_link_type=None,
        evidence_msgs=["M060"],
        confidence=None,
    )
    assert person.circle is None
    assert person.org_link_type is None
    assert person.confidence is None

    # InterpretedOrganisation optional fields
    org = InterpretedOrganisation(
        name="Mystery Org",
        org_type=None,
        side=None,
        role=None,
        evidence_msgs=[],
        associated_chats=[],
        confidence=None,
    )
    assert org.org_type is None
    assert org.side is None
    assert org.role is None
    assert org.confidence is None


def test_list_fields_accept_empty_lists():
    """Verify that all list fields accept empty lists without error."""
    loop = InterpretedOpenLoop(
        source_msg="M200",
        chat_id="C03",
        type="followup",
        is_genuine=True,
        title="Follow up on invoice",
        owner="ME",
        target_person="P05",
        status="open",
        closed_by=None,
        due_date=None,
        suggested_action="assist",
        confidence=0.75,
        validation_flags=[],
    )
    assert loop.validation_flags == []

    goal = InterpretedGoal(
        title="Learn Spanish",
        area="self",
        type="habit",
        horizon="1yr",
        evidence_msgs=[],
        is_genuine=True,
    )
    assert goal.evidence_msgs == []

    person = InterpretedPerson(
        person_id="P10",
        circle="friend",
        looks_after=[],
        associated_org_names=[],
        org_link_type=None,
        evidence_msgs=[],
    )
    assert person.looks_after == []
    assert person.associated_org_names == []
    assert person.evidence_msgs == []

    org = InterpretedOrganisation(
        name="Casual Club",
        org_type="society",
        side="home",
        role="member",
        evidence_msgs=[],
        associated_chats=[],
    )
    assert org.evidence_msgs == []
    assert org.associated_chats == []


def test_default_values_for_optional_metadata():
    """Verify that optional debugging metadata defaults to None when omitted."""
    # InterpretedOpenLoop defaults
    loop = InterpretedOpenLoop(
        source_msg="M300",
        chat_id="C01",
        type="commitment",
        is_genuine=True,
        title="Send summary",
        owner="ME",
        target_person="P01",
        status="open",
        closed_by=None,
        due_date="2026-10-30",
        suggested_action="remind",
    )
    assert loop.confidence is None
    assert loop.validation_flags is None

    # InterpretedGoal defaults
    goal = InterpretedGoal(
        title="Run 10k",
        area="health",
        type="milestone",
        horizon="3mo",
        evidence_msgs=["M301"],
        is_genuine=True,
    )
    assert goal.confidence is None

    # InterpretedPerson defaults
    person = InterpretedPerson(
        person_id="P02",
        circle="colleague",
        looks_after=["P03"],
        associated_org_names=["Das Printers"],
        org_link_type="runs",
        evidence_msgs=["M302"],
    )
    assert person.confidence is None

    # InterpretedOrganisation defaults
    org = InterpretedOrganisation(
        name="Das Printers",
        org_type="company",
        side="work",
        role="owner",
        evidence_msgs=["M303"],
        associated_chats=["C05"],
    )
    assert org.confidence is None


def test_independent_instances_do_not_share_mutable_lists():
    """Verify that separate instances do not share mutable list references."""
    # 1. InterpretedOpenLoop validation_flags
    flags_a = ["flag_1"]
    flags_b = ["flag_2"]
    loop1 = InterpretedOpenLoop(
        "M1", "C1", "commitment", True, "T1", "ME", None, "open", None, None, None, validation_flags=flags_a
    )
    loop2 = InterpretedOpenLoop(
        "M2", "C1", "commitment", True, "T2", "ME", None, "open", None, None, None, validation_flags=flags_b
    )
    loop1.validation_flags.append("extra_flag")
    assert loop1.validation_flags == ["flag_1", "extra_flag"]
    assert loop2.validation_flags == ["flag_2"]

    # 2. InterpretedGoal evidence_msgs
    list1 = ["M10"]
    list2 = ["M20"]
    goal1 = InterpretedGoal("G1", "self", "habit", "1yr", list1, True)
    goal2 = InterpretedGoal("G2", "self", "habit", "1yr", list2, True)
    goal1.evidence_msgs.append("M11")
    assert goal1.evidence_msgs == ["M10", "M11"]
    assert goal2.evidence_msgs == ["M20"]

    # 3. InterpretedPerson lists
    p1 = InterpretedPerson("P1", "friend", ["P2"], ["OrgA"], "runs", ["M1"])
    p2 = InterpretedPerson("P2", "friend", ["P3"], ["OrgB"], "supports", ["M2"])
    p1.looks_after.append("P4")
    p1.associated_org_names.append("OrgC")
    p1.evidence_msgs.append("M3")
    assert p1.looks_after == ["P2", "P4"]
    assert p2.looks_after == ["P3"]
    assert p1.associated_org_names == ["OrgA", "OrgC"]
    assert p2.associated_org_names == ["OrgB"]
    assert p1.evidence_msgs == ["M1", "M3"]
    assert p2.evidence_msgs == ["M2"]

    # 4. InterpretedOrganisation lists
    o1 = InterpretedOrganisation("O1", "company", "work", "owner", ["M1"], ["C1"])
    o2 = InterpretedOrganisation("O2", "company", "work", "owner", ["M2"], ["C2"])
    o1.evidence_msgs.append("M3")
    o1.associated_chats.append("C3")
    assert o1.evidence_msgs == ["M1", "M3"]
    assert o2.evidence_msgs == ["M2"]
    assert o1.associated_chats == ["C1", "C3"]
    assert o2.associated_chats == ["C2"]


def test_dataclass_field_names_and_types():
    """Verify that all four dataclasses declare the exact expected field names and types."""
    assert is_dataclass(InterpretedOpenLoop)
    assert is_dataclass(InterpretedGoal)
    assert is_dataclass(InterpretedPerson)
    assert is_dataclass(InterpretedOrganisation)

    # 1. InterpretedOpenLoop fields
    open_loop_fields = {f.name: f.type for f in fields(InterpretedOpenLoop)}
    expected_open_loop_names = [
        "source_msg",
        "chat_id",
        "type",
        "is_genuine",
        "title",
        "owner",
        "target_person",
        "status",
        "closed_by",
        "due_date",
        "suggested_action",
        "confidence",
        "validation_flags",
        "due_anchor_msg",
    ]
    assert list(open_loop_fields.keys()) == expected_open_loop_names
    assert open_loop_fields["source_msg"] is str
    assert open_loop_fields["chat_id"] is str
    assert open_loop_fields["type"] is str
    assert open_loop_fields["is_genuine"] is bool
    assert open_loop_fields["title"] is str
    assert open_loop_fields["owner"] == Optional[str]
    assert open_loop_fields["target_person"] == Optional[str]
    assert open_loop_fields["status"] is str
    assert open_loop_fields["closed_by"] == Optional[str]
    assert open_loop_fields["due_date"] == Optional[str]
    assert open_loop_fields["suggested_action"] == Optional[str]
    assert open_loop_fields["confidence"] == Optional[float]
    assert open_loop_fields["validation_flags"] == Optional[List[str]]
    assert open_loop_fields["due_anchor_msg"] == Optional[str]

    # 2. InterpretedGoal fields
    goal_fields = {f.name: f.type for f in fields(InterpretedGoal)}
    expected_goal_names = [
        "title",
        "area",
        "type",
        "horizon",
        "evidence_msgs",
        "is_genuine",
        "confidence",
    ]
    assert list(goal_fields.keys()) == expected_goal_names
    assert goal_fields["title"] is str
    assert goal_fields["area"] is str
    assert goal_fields["type"] is str
    assert goal_fields["horizon"] is str
    assert goal_fields["evidence_msgs"] == List[str]
    assert goal_fields["is_genuine"] is bool
    assert goal_fields["confidence"] == Optional[float]

    # 3. InterpretedPerson fields
    person_fields = {f.name: f.type for f in fields(InterpretedPerson)}
    expected_person_names = [
        "person_id",
        "circle",
        "looks_after",
        "associated_org_names",
        "org_link_type",
        "evidence_msgs",
        "confidence",
    ]
    assert list(person_fields.keys()) == expected_person_names
    assert person_fields["person_id"] is str
    assert person_fields["circle"] == Optional[str]
    assert person_fields["looks_after"] == List[str]
    assert person_fields["associated_org_names"] == List[str]
    assert person_fields["org_link_type"] == Optional[str]
    assert person_fields["evidence_msgs"] == List[str]
    assert person_fields["confidence"] == Optional[float]

    # 4. InterpretedOrganisation fields
    org_fields = {f.name: f.type for f in fields(InterpretedOrganisation)}
    expected_org_names = [
        "name",
        "org_type",
        "side",
        "role",
        "evidence_msgs",
        "associated_chats",
        "confidence",
    ]
    assert list(org_fields.keys()) == expected_org_names
    assert org_fields["name"] is str
    assert org_fields["org_type"] == Optional[str]
    assert org_fields["side"] == Optional[str]
    assert org_fields["role"] == Optional[str]
    assert org_fields["evidence_msgs"] == List[str]
    assert org_fields["associated_chats"] == List[str]
    assert org_fields["confidence"] == Optional[float]


def test_open_loop_due_anchor_msg_defaults_and_backwards_compatibility():
    """Verify InterpretedOpenLoop positional and keyword constructor backwards compatibility with due_anchor_msg."""
    # 1. 11 Positional args: defaults for confidence, validation_flags, due_anchor_msg
    loop11 = InterpretedOpenLoop(
        "M01", "C01", "commitment", True, "Title", "ME", None, "open", None, None, None
    )
    assert loop11.confidence is None
    assert loop11.validation_flags is None
    assert loop11.due_anchor_msg is None

    # 2. 13 Positional args: sets confidence and validation_flags, due_anchor_msg defaults to None
    loop13 = InterpretedOpenLoop(
        "M01", "C01", "commitment", True, "Title", "ME", None, "open", None, None, None, 0.85, ["flag_ok"]
    )
    assert loop13.confidence == 0.85
    assert loop13.validation_flags == ["flag_ok"]
    assert loop13.due_anchor_msg is None

    # 3. 14 Positional args: explicitly supplies due_anchor_msg
    loop14 = InterpretedOpenLoop(
        "M01", "C01", "commitment", True, "Title", "ME", None, "open", None, "2026-11-01", "remind", 0.9, ["flag_ok"], "M02"
    )
    assert loop14.confidence == 0.9
    assert loop14.validation_flags == ["flag_ok"]
    assert loop14.due_anchor_msg == "M02"

    # 4. Keyword-based construction
    loop_kw = InterpretedOpenLoop(
        source_msg="M01",
        chat_id="C01",
        type="deadline",
        is_genuine=True,
        title="Submit report",
        owner="ME",
        target_person=None,
        status="open",
        closed_by=None,
        due_date="2026-11-05",
        suggested_action="remind",
        confidence=0.95,
        validation_flags=None,
        due_anchor_msg="M03",
    )
    assert loop_kw.due_anchor_msg == "M03"
    assert loop_kw.due_date == "2026-11-05"

