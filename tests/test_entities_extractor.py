"""Unit and integration tests for People, Organisation, and Chat extraction."""

import json
from pathlib import Path
import pytest

from src.entities.extractor import (
    build_chats,
    build_organisations,
    build_people,
    extract_entities,
    is_substantive_message,
    select_primary_participant,
)
from src.entities.models import (
    EntitiesResult,
    EntityExtractionConfig,
    EntityExtractionDiagnostics,
    EntityExtractionError,
    UnresolvableChatMappingError,
    VALID_CIRCLES,
    VALID_ORG_LINKS,
    VALID_ORG_ROLES,
    VALID_ORG_SIDES,
    VALID_ORG_TYPES,
    is_valid_confidence,
    is_valid_llm_org_link,
    is_valid_llm_person_record,
    validate_entities_referential_integrity,
)
from src.entities.resolver import (
    CandidateOrganisation,
    GENUINE_LEGAL_SUFFIXES,
    MEANINGFUL_NAME_COMPONENTS,
    are_distinct_entities,
    find_candidate_org,
    generate_org_id,
    match_canonical_seed_org,
    normalize_org_name,
    strip_legal_suffixes,
)
from src.ingestion.loader import load_dataset
from src.interpretation.records import InterpretedOrganisation, InterpretedPerson
from src.preprocessing.context_builder import build_conversation_context


# ==============================================================================
# 1. Models & Substantive Message Tests
# ==============================================================================

def test_is_substantive_message_valid_kinds():
    assert is_substantive_message({"kind": "text", "body": "Please review the balance sheet."}) is True
    assert is_substantive_message({"kind": "voice_transcript", "body": "I will arrive by 5pm."}) is True
    assert is_substantive_message({"kind": "image_caption", "body": "Receipt for client travel"}) is True
    assert is_substantive_message({"kind": "document", "body": "Q3_financial_report.pdf"}) is True


def test_is_substantive_message_excludes_trivial_and_media():
    assert is_substantive_message({"kind": "text", "body": ""}) is False
    assert is_substantive_message({"kind": "text", "body": "  k  "}) is False
    assert is_substantive_message({"kind": "text", "body": "👍"}) is False
    assert is_substantive_message({"kind": "text", "body": "🙏"}) is False
    assert is_substantive_message({"kind": "text", "body": "<Media omitted>"}) is False
    assert is_substantive_message({"kind": "text", "body": "sticker omitted"}) is False
    assert is_substantive_message({"kind": "text", "body": "This message was deleted"}) is False
    assert is_substantive_message({"kind": "unknown_kind", "body": "Some text"}) is False


# ==============================================================================
# 2. Primary Participant Selection & Tie-Breaking Tests
# ==============================================================================

def test_select_primary_participant_case_1_substantive_counts_differ():
    chat = {"chat_id": "C10", "is_group": True, "participants": ["ME", "P01", "P02"]}
    messages = [
        {"id": "M1", "chat_id": "C10", "sender_id": "P01", "kind": "text", "body": "First substantive message", "ts": 100},
        {"id": "M2", "chat_id": "C10", "sender_id": "P01", "kind": "text", "body": "Second substantive message", "ts": 200},
        {"id": "M3", "chat_id": "C10", "sender_id": "P02", "kind": "text", "body": "Single substantive message", "ts": 300},
        {"id": "M4", "chat_id": "C10", "sender_id": "P02", "kind": "text", "body": "👍", "ts": 400},  # non-substantive
    ]
    target = select_primary_participant(chat, messages, all_direct_chats={}, global_substantive_counts={})
    assert target == "P01"


def test_select_primary_participant_case_2_counts_tie_direct_chat_tie_break():
    chat = {"chat_id": "C10", "is_group": True, "participants": ["ME", "P01", "P02"]}
    messages = [
        {"id": "M1", "chat_id": "C10", "sender_id": "P01", "kind": "text", "body": "Message from P1", "ts": 100},
        {"id": "M2", "chat_id": "C10", "sender_id": "P02", "kind": "text", "body": "Message from P2", "ts": 200},
    ]
    # ME has a direct chat with P02, but not P01
    all_direct_chats = {"P02": "C01"}
    target = select_primary_participant(chat, messages, all_direct_chats=all_direct_chats, global_substantive_counts={})
    assert target == "P02"


def test_select_primary_participant_case_2_counts_tie_earliest_timestamp_tie_break():
    chat = {"chat_id": "C10", "is_group": True, "participants": ["ME", "P01", "P02"]}
    messages = [
        {"id": "M2", "chat_id": "C10", "sender_id": "P02", "kind": "text", "body": "P2 earliest", "ts": 100},
        {"id": "M1", "chat_id": "C10", "sender_id": "P01", "kind": "text", "body": "P1 later", "ts": 200},
    ]
    target = select_primary_participant(chat, messages, all_direct_chats={}, global_substantive_counts={})
    assert target == "P02"


def test_select_primary_participant_case_2_equal_earliest_timestamp_lexicographical_msg_id_tie_break():
    chat = {"chat_id": "C10", "is_group": True, "participants": ["ME", "P01", "P02"]}
    messages = [
        {"id": "M015", "chat_id": "C10", "sender_id": "P02", "kind": "text", "body": "P2 same ts", "ts": 100},
        {"id": "M016", "chat_id": "C10", "sender_id": "P01", "kind": "text", "body": "P1 same ts", "ts": 100},
    ]
    # Timestamps are identical (100). M015 < M016, so P02 wins deterministically
    target = select_primary_participant(chat, messages, all_direct_chats={}, global_substantive_counts={})
    assert target == "P02"


def test_select_primary_participant_case_3_zero_messages_global_counts_tie_break():
    chat = {"chat_id": "C10", "is_group": True, "participants": ["ME", "P01", "P02"]}
    global_counts = {"P01": 5, "P02": 25}
    target = select_primary_participant(chat, [], all_direct_chats={}, global_substantive_counts=global_counts)
    assert target == "P02"


def test_select_primary_participant_case_4_completely_ungrounded_canonical_anchor():
    chat = {"chat_id": "C10", "is_group": True, "participants": ["ME", "P08", "P03"]}
    target = select_primary_participant(chat, [], all_direct_chats={}, global_substantive_counts={})
    assert target == "P03"  # sorted(["P08", "P03"])[0]


def test_select_primary_participant_solo_me_returns_none():
    chat = {"chat_id": "C99", "is_group": True, "participants": ["ME"]}
    target = select_primary_participant(chat, [], all_direct_chats={}, global_substantive_counts={})
    assert target is None


# ==============================================================================
# 3. Organisation Resolution & Deterministic ID Tests
# ==============================================================================

def test_generate_org_id_deterministic_and_collision_free():
    allocated = set()
    id1 = generate_org_id("Brightside Design", allocated)
    assert id1 == "org_brightside_design"
    assert id1 in allocated

    # Collision test: same name again
    id2 = generate_org_id("Brightside Design", allocated)
    assert id2.startswith("org_brightside_design_")
    assert id2 != id1
    assert id2 in allocated


def test_are_distinct_entities_shared_tokens_diverging():
    # "NEXUS" vs "Nexus DaaS"
    assert are_distinct_entities(
        "NEXUS", "Nexus DaaS",
        chats1={"C01", "C02"}, chats2={"C50"},
        people1={"P01", "P02"}, people2={"P73"}
    ) is True

    # Identical with legal suffix
    assert are_distinct_entities(
        "PostMyAd", "PostMyAd Pvt Ltd",
        chats1={"C10"}, chats2={"C10"},
        people1={"P10"}, people2={"P10"}
    ) is False


def test_match_canonical_seed_org():
    seed_orgs = ["Brightside Design", "NEXUS"]
    assert match_canonical_seed_org("brightside design", seed_orgs) == "Brightside Design"
    assert match_canonical_seed_org("Brightside Design team", seed_orgs) == "Brightside Design"
    assert match_canonical_seed_org("nexus", seed_orgs) == "NEXUS"
    assert match_canonical_seed_org("Unrelated Ltd", seed_orgs) is None


# ==============================================================================
# 4. Organisation Lifecycle & Precedence Tests
# ==============================================================================

@pytest.fixture
def synthetic_context(tmp_path):
    """Create a minimal synthetic ConversationContext for entity testing."""
    chats_file = tmp_path / "chats.jsonl"
    msgs_file = tmp_path / "messages.jsonl"
    seed_file = tmp_path / "seed.json"

    chats = [
        {"chat_id": "C01", "name": "Direct Neha", "is_group": False, "participants": ["ME", "P01"]},
        {"chat_id": "C02", "name": "Brightside team", "is_group": True, "participants": ["ME", "P01", "P02"]},
        {"chat_id": "C03", "name": "Ghar 🏠", "is_group": True, "participants": ["ME", "P03", "P04"]},
        {"chat_id": "C04", "name": "Casual sports", "is_group": True, "participants": ["ME", "P05", "P06"]},
    ]
    messages = [
        {"id": "M01", "chat_id": "C01", "sender_id": "ME", "sender_name": "ME", "kind": "text", "body": "Please review the pitch deck for clients", "ts": 100},
        {"id": "M02", "chat_id": "C01", "sender_id": "P01", "sender_name": "Neha", "kind": "text", "body": "Will do, Brightside report ready", "ts": 200},
        {"id": "M03", "chat_id": "C02", "sender_id": "P02", "sender_name": "Karan", "kind": "text", "body": "Brightside design invoice sent", "ts": 300},
        {"id": "M04", "chat_id": "C03", "sender_id": "P03", "sender_name": "Raju", "kind": "text", "body": "Raju driver pick up the kids from school", "ts": 400},
        {"id": "M05", "chat_id": "C04", "sender_id": "P05", "sender_name": "Player 5", "kind": "text", "body": "Pickleball match at 7am tomorrow", "ts": 500},
    ]
    seed = {
        "circles": {"P01": "colleague"},
        "organisations": ["Brightside Design", "Unused Seed Org"],
        "goals": [{"title": "Brightside Design to 10 retainer clients", "area": "work", "type": "milestone", "horizon": "1yr"}],
    }

    with open(chats_file, "w", encoding="utf-8") as f:
        for c in chats:
            f.write(json.dumps(c) + "\n")
    with open(msgs_file, "w", encoding="utf-8") as f:
        for m in messages:
            f.write(json.dumps(m) + "\n")
    with open(seed_file, "w", encoding="utf-8") as f:
        json.dump(seed, f)

    return build_conversation_context(load_dataset(tmp_path))


def test_build_organisations_preserves_seed_precedence_and_prunes_orphans(synthetic_context):
    interpreted_orgs = [
        # Discovered org with evidence and attached chat
        InterpretedOrganisation(
            name="Brightside Design",
            org_type="company",
            side="work",
            role="owner",
            evidence_msgs=["M03"],
            associated_chats=["C02"],
            confidence=0.95,
        ),
        # Discovered candidate that has 0 members and is NOT in seed -> should be pruned!
        InterpretedOrganisation(
            name="Phantom Corp",
            org_type="company",
            side="work",
            role="member",
            evidence_msgs=[],
            associated_chats=[],
            confidence=0.85,
        ),
    ]

    retained, pruned = build_organisations(synthetic_context, interpreted_orgs=interpreted_orgs)
    retained_names = [o.name for o in retained]

    # "Brightside Design" is seed & has members -> retained
    assert "Brightside Design" in retained_names

    # "Unused Seed Org" has 0 members but is a FOUNDATIONAL SEED ORG -> PRESERVED by Precedence Rule!
    assert "Unused Seed Org" in retained_names

    # "Phantom Corp" is non-seed and has 0 members -> PRUNED as an orphan!
    assert "Phantom Corp" not in retained_names
    assert "Phantom Corp" in pruned

    # "Home" created because "Ghar 🏠" chat is present with domestic context
    assert "Home" in retained_names
    home_org = next(o for o in retained if o.name == "Home")
    assert home_org.org_type == "residence"
    assert home_org.side == "home"


# ==============================================================================
# 5. People Extraction & Decoupled Circle Resolution Tests
# ==============================================================================

def test_build_people_decoupled_circle_and_uncertainty_omission(synthetic_context):
    retained_orgs, _ = build_organisations(synthetic_context)
    interpreted_people = [
        InterpretedPerson(
            person_id="P01",
            circle="colleague",
            looks_after=[],
            associated_org_names=["Brightside Design"],
            org_link_type="runs",
            evidence_msgs=["M02"],
            confidence=0.95,
        ),
        InterpretedPerson(
            person_id="P03",
            circle="staff",
            looks_after=["P04"],
            associated_org_names=["Home"],
            org_link_type="supports",
            evidence_msgs=["M04"],
            confidence=0.90,
        ),
    ]

    people, omitted = build_people(synthetic_context, retained_orgs, interpreted_people=interpreted_people)
    people_by_id = {p["id"]: p for p in people}

    # P01 is seed & colleague
    assert "P01" in people_by_id
    assert people_by_id["P01"]["circle"] == "colleague"
    assert people_by_id["P01"]["links"][0]["link"] == "runs"

    # P03 is staff with looks_after P04
    assert "P03" in people_by_id
    assert people_by_id["P03"]["circle"] == "staff"
    assert people_by_id["P03"]["looks_after"] == ["P04"]

    # P05 and P06 in "Casual sports" have no reliable relationship evidence:
    # They MUST NOT be blindly defaulted to "friend"! They are omitted from people.
    assert "P05" not in people_by_id
    assert "P06" not in people_by_id
    assert "P05" in omitted or "P06" in omitted

    # "ME" is NEVER included
    assert "ME" not in people_by_id


# ==============================================================================
# 6. Chat Mapping & Solo-ME Strict Policy Tests
# ==============================================================================

def test_build_chats_direct_and_group_mappings(synthetic_context):
    retained_orgs, _ = build_organisations(synthetic_context)
    chats, diagnostics = build_chats(synthetic_context, retained_orgs)

    chat_map = {c["chat_id"]: c for c in chats}

    # C01 is direct -> maps to P01
    assert "person" in chat_map["C01"]
    assert chat_map["C01"]["person"] == "P01"

    # C02 is "Brightside team" -> maps to organisation
    assert "organisation" in chat_map["C02"]
    assert chat_map["C02"]["organisation"] == "org_brightside_design"

    # C03 is "Ghar 🏠" -> maps to organisation Home
    assert "organisation" in chat_map["C03"]
    assert chat_map["C03"]["organisation"] == "org_home"

    # C04 is "Casual sports" (ambiguous social group) -> maps to primary participant P05
    assert "person" in chat_map["C04"]
    assert chat_map["C04"]["person"] == "P05"


def test_build_chats_solo_me_ungrounded_raises_extraction_error(tmp_path):
    """A solo-ME chat without independently grounded org evidence raises UnresolvableChatMappingError."""
    chats_file = tmp_path / "chats.jsonl"
    msgs_file = tmp_path / "messages.jsonl"

    chats = [
        {"chat_id": "C99", "name": "Random Personal Notes", "is_group": True, "participants": ["ME"]},
    ]
    with open(chats_file, "w", encoding="utf-8") as f:
        for c in chats:
            f.write(json.dumps(c) + "\n")
    with open(msgs_file, "w", encoding="utf-8") as f:
        pass

    context = build_conversation_context(load_dataset(tmp_path))
    retained_orgs = [
        CandidateOrganisation(name="PostMyAd", is_seed=True, id="org_postmyad"),
    ]

    with pytest.raises(UnresolvableChatMappingError, match="contains only 'ME' with zero organisational evidence"):
        build_chats(context, retained_orgs)


def test_build_chats_solo_me_grounded_in_title_maps_to_org(tmp_path):
    """A solo-ME chat whose title explicitly references a verified organisation maps to that org."""
    chats_file = tmp_path / "chats.jsonl"
    msgs_file = tmp_path / "messages.jsonl"

    chats = [
        {"chat_id": "C99", "name": "PostMyAd draft notes", "is_group": True, "participants": ["ME"]},
    ]
    with open(chats_file, "w", encoding="utf-8") as f:
        for c in chats:
            f.write(json.dumps(c) + "\n")
    with open(msgs_file, "w", encoding="utf-8") as f:
        pass

    context = build_conversation_context(load_dataset(tmp_path))
    retained_orgs = [
        CandidateOrganisation(name="PostMyAd", is_seed=True, id="org_postmyad"),
    ]

    chats_out, diags = build_chats(context, retained_orgs)
    assert len(chats_out) == 1
    assert chats_out[0]["chat_id"] == "C99"
    assert chats_out[0]["organisation"] == "org_postmyad"


# ==============================================================================
# 7. End-to-End Entity Extraction & Referential Integrity
# ==============================================================================

def test_extract_entities_end_to_end_synthetic(synthetic_context):
    result = extract_entities(synthetic_context)
    assert isinstance(result, EntitiesResult)

    out_dict = result.to_dict()
    assert "people" in out_dict
    assert "organisations" in out_dict
    assert "chats" in out_dict

    # Check 100% of input chats are tied
    input_cids = set(synthetic_context.chats.keys())
    tied_cids = {c["chat_id"] for c in out_dict["chats"]}
    assert input_cids == tied_cids

    # Validate referential integrity helper returns 0 errors
    input_pids = set(p for p in synthetic_context.participants.keys() if p != "ME")
    errors = validate_entities_referential_integrity(result, input_cids, input_pids)
    assert errors == []


def test_extract_entities_on_example_dataset():
    """Verify entity extraction on official example dataset passes referential validation."""
    example_path = Path("example")
    if not example_path.exists():
        pytest.skip("example folder not found")

    dataset = load_dataset(example_path)
    context = build_conversation_context(dataset)
    result = extract_entities(context)

    input_cids = set(context.chats.keys())
    input_pids = set(p for p in context.participants.keys() if p != "ME")

    errors = validate_entities_referential_integrity(result, input_cids, input_pids)
    assert errors == []

    # Verify every chat has exactly one target
    for c in result.chats:
        assert ("person" in c) ^ ("organisation" in c)


def test_extract_entities_on_dev_dataset():
    """Verify entity extraction on official data/dev dataset passes referential validation."""
    dev_path = Path("data/dev")
    if not dev_path.exists():
        pytest.skip("data/dev folder not found")

    dataset = load_dataset(dev_path)
    context = build_conversation_context(dataset)
    result = extract_entities(context)

    input_cids = set(context.chats.keys())
    input_pids = set(p for p in context.participants.keys() if p != "ME")

    errors = validate_entities_referential_integrity(result, input_cids, input_pids)
    assert errors == []

    # Verify 138/138 chats are tied
    assert len(result.chats) == 138


# ==============================================================================
# 8. Organisation Lifecycle & Person Linking Tests (Targeted Correction)
# ==============================================================================

def test_non_seed_org_with_validated_person_survives_pruning_zero_chats(tmp_path):
    """A non-seed organisation with a validated person association survives pruning even if it has no attached group chat."""
    chats_file = tmp_path / "chats.jsonl"
    msgs_file = tmp_path / "messages.jsonl"
    seed_file = tmp_path / "seed.json"

    # P01 is in a 1:1 direct chat with ME (no group chat for "Vertex Labs")
    chats = [
        {"chat_id": "C01", "name": "Direct Alice", "is_group": False, "participants": ["ME", "P01"]},
    ]
    messages = [
        {"id": "M01", "chat_id": "C01", "sender_id": "P01", "sender_name": "Alice", "kind": "text", "body": "Vertex Labs monthly retainer invoice attached", "ts": 100},
    ]
    seed = {"circles": {}, "organisations": [], "goals": []}

    with open(chats_file, "w", encoding="utf-8") as f:
        for c in chats:
            f.write(json.dumps(c) + "\n")
    with open(msgs_file, "w", encoding="utf-8") as f:
        for m in messages:
            f.write(json.dumps(m) + "\n")
    with open(seed_file, "w", encoding="utf-8") as f:
        json.dump(seed, f)

    context = build_conversation_context(load_dataset(tmp_path))

    # Interpreted non-seed org with 0 attached chats
    interpreted_orgs = [
        InterpretedOrganisation(
            name="Vertex Labs",
            org_type="company",
            side="work",
            role="member",
            evidence_msgs=["M01"],
            associated_chats=[],  # 0 attached chats!
            confidence=0.90,
        )
    ]
    # Interpreted validated person who works at Vertex Labs
    interpreted_people = [
        InterpretedPerson(
            person_id="P01",
            circle="vendor",
            looks_after=[],
            associated_org_names=["Vertex Labs"],
            org_link_type="runs",
            evidence_msgs=["M01"],
            confidence=0.92,
        )
    ]

    result = extract_entities(
        context=context,
        interpreted_people=interpreted_people,
        interpreted_orgs=interpreted_orgs,
    )

    org_names = [o["name"] for o in result.organisations]
    # Vertex Labs must survive pruning!
    assert "Vertex Labs" in org_names
    vertex_org = next(o for o in result.organisations if o["name"] == "Vertex Labs")
    assert vertex_org["id"] == "org_vertex_labs"

    # P01 must have link to org_vertex_labs
    p01_entry = next(p for p in result.people if p["id"] == "P01")
    assert p01_entry["circle"] == "vendor"
    assert "links" in p01_entry
    assert p01_entry["links"] == [{"organisation": "org_vertex_labs", "link": "runs"}]

    # Referential integrity check
    errors = validate_entities_referential_integrity(
        result=result,
        input_chat_ids={"C01"},
        input_person_ids={"P01"},
    )
    assert errors == []


def test_non_seed_org_with_no_person_and_no_chat_is_pruned(tmp_path):
    """A non-seed organisation with neither a valid person association nor an attached chat is still pruned."""
    chats_file = tmp_path / "chats.jsonl"
    msgs_file = tmp_path / "messages.jsonl"
    seed_file = tmp_path / "seed.json"

    chats = [
        {"chat_id": "C01", "name": "Direct Bob", "is_group": False, "participants": ["ME", "P01"]},
    ]
    messages = [
        {"id": "M01", "chat_id": "C01", "sender_id": "P01", "sender_name": "Bob", "kind": "text", "body": "Hello Bob here", "ts": 100},
    ]
    seed = {"circles": {}, "organisations": [], "goals": []}

    with open(chats_file, "w", encoding="utf-8") as f:
        for c in chats:
            f.write(json.dumps(c) + "\n")
    with open(msgs_file, "w", encoding="utf-8") as f:
        for m in messages:
            f.write(json.dumps(m) + "\n")
    with open(seed_file, "w", encoding="utf-8") as f:
        json.dump(seed, f)

    context = build_conversation_context(load_dataset(tmp_path))

    # Interpreted non-seed org with 0 attached chats and NO associated person
    interpreted_orgs = [
        InterpretedOrganisation(
            name="Phantom Corp",
            org_type="company",
            side="work",
            role="member",
            evidence_msgs=[],
            associated_chats=[],
            confidence=0.85,
        )
    ]
    interpreted_people = [
        InterpretedPerson(
            person_id="P01",
            circle="friend",
            looks_after=[],
            associated_org_names=[],  # No org association!
            org_link_type=None,
            evidence_msgs=["M01"],
            confidence=0.90,
        )
    ]

    result = extract_entities(
        context=context,
        interpreted_people=interpreted_people,
        interpreted_orgs=interpreted_orgs,
    )

    org_names = [o["name"] for o in result.organisations]
    # Phantom Corp must be pruned as an orphan!
    assert "Phantom Corp" not in org_names
    assert "Phantom Corp" in result.diagnostics.pruned_orphan_org_names


def test_low_confidence_or_invalid_person_cannot_keep_org_alive(tmp_path):
    """A low-confidence or invalid person association cannot keep an organisation alive."""
    chats_file = tmp_path / "chats.jsonl"
    msgs_file = tmp_path / "messages.jsonl"
    seed_file = tmp_path / "seed.json"

    chats = [
        {"chat_id": "C01", "name": "Direct Charlie", "is_group": False, "participants": ["ME", "P01"]},
        {"chat_id": "C02", "name": "Direct Dave", "is_group": False, "participants": ["ME", "P02"]},
    ]
    messages = [
        {"id": "M01", "chat_id": "C01", "sender_id": "P01", "sender_name": "Charlie", "kind": "text", "body": "Hey there", "ts": 100},
        {"id": "M02", "chat_id": "C02", "sender_id": "P02", "sender_name": "Dave", "kind": "text", "body": "Meeting notes", "ts": 200},
    ]
    seed = {"circles": {}, "organisations": [], "goals": []}

    with open(chats_file, "w", encoding="utf-8") as f:
        for c in chats:
            f.write(json.dumps(c) + "\n")
    with open(msgs_file, "w", encoding="utf-8") as f:
        for m in messages:
            f.write(json.dumps(m) + "\n")
    with open(seed_file, "w", encoding="utf-8") as f:
        json.dump(seed, f)

    context = build_conversation_context(load_dataset(tmp_path))

    interpreted_orgs = [
        InterpretedOrganisation(
            name="Shady LLC",
            org_type="company",
            side="work",
            role="member",
            evidence_msgs=[],
            associated_chats=[],
            confidence=0.85,
        ),
        InterpretedOrganisation(
            name="Ghost Ventures",
            org_type="company",
            side="work",
            role="member",
            evidence_msgs=[],
            associated_chats=[],
            confidence=0.85,
        ),
    ]
    interpreted_people = [
        # P01 mentions Shady LLC, but has confidence 0.40 (below min_llm_circle_confidence 0.65)
        InterpretedPerson(
            person_id="P01",
            circle="colleague",
            looks_after=[],
            associated_org_names=["Shady LLC"],
            org_link_type="runs",
            evidence_msgs=["M01"],
            confidence=0.40,  # LOW CONFIDENCE!
        ),
        # P02 mentions Ghost Ventures, but circle is unknown/invalid -> person is omitted!
        InterpretedPerson(
            person_id="P02",
            circle=None,  # INVALID/OMITTED CIRCLE!
            looks_after=[],
            associated_org_names=["Ghost Ventures"],
            org_link_type="runs",
            evidence_msgs=[],
            confidence=0.90,
        ),
    ]

    result = extract_entities(
        context=context,
        interpreted_people=interpreted_people,
        interpreted_orgs=interpreted_orgs,
    )

    org_names = [o["name"] for o in result.organisations]
    # Both Shady LLC and Ghost Ventures must be pruned!
    assert "Shady LLC" not in org_names
    assert "Ghost Ventures" not in org_names
    assert "Shady LLC" in result.diagnostics.pruned_orphan_org_names
    assert "Ghost Ventures" in result.diagnostics.pruned_orphan_org_names


def test_person_link_emitted_only_when_org_survives(tmp_path):
    """A person link is emitted only when its organisation survives final resolution."""
    chats_file = tmp_path / "chats.jsonl"
    msgs_file = tmp_path / "messages.jsonl"
    seed_file = tmp_path / "seed.json"

    chats = [
        {"chat_id": "C01", "name": "Direct Eve", "is_group": False, "participants": ["ME", "P01"]},
        {"chat_id": "C02", "name": "Global Corp All Hands", "is_group": True, "participants": ["ME", "P01", "P02"]},
    ]
    messages = [
        {"id": "M01", "chat_id": "C01", "sender_id": "P01", "sender_name": "Eve", "kind": "text", "body": "Reviewing Global Corp quarterly goals", "ts": 100},
        {"id": "M02", "chat_id": "C02", "sender_id": "P02", "sender_name": "Frank", "kind": "text", "body": "Global Corp all hands agenda", "ts": 200},
    ]
    seed = {
        "circles": {"P01": "colleague"},
        "organisations": [],
        "goals": [],
    }

    with open(chats_file, "w", encoding="utf-8") as f:
        for c in chats:
            f.write(json.dumps(c) + "\n")
    with open(msgs_file, "w", encoding="utf-8") as f:
        for m in messages:
            f.write(json.dumps(m) + "\n")
    with open(seed_file, "w", encoding="utf-8") as f:
        json.dump(seed, f)

    context = build_conversation_context(load_dataset(tmp_path))

    interpreted_orgs = [
        # Survives because C02 is attached
        InterpretedOrganisation(
            name="Global Corp",
            org_type="company",
            side="work",
            role="member",
            evidence_msgs=["M02"],
            associated_chats=["C02"],
            confidence=0.95,
        ),
        # Fails org confidence floor (0.50 < min_llm_org_confidence 0.70) -> does NOT survive
        InterpretedOrganisation(
            name="Unverified Agency",
            org_type="company",
            side="work",
            role="member",
            evidence_msgs=[],
            associated_chats=[],
            confidence=0.50,
        ),
    ]
    # P01 claims association with both Global Corp AND Unverified Agency
    interpreted_people = [
        InterpretedPerson(
            person_id="P01",
            circle="colleague",
            looks_after=[],
            associated_org_names=["Global Corp", "Unverified Agency"],
            org_link_type="runs",
            evidence_msgs=["M01"],
            confidence=0.90,
        )
    ]

    result = extract_entities(
        context=context,
        interpreted_people=interpreted_people,
        interpreted_orgs=interpreted_orgs,
    )

    org_names = [o["name"] for o in result.organisations]
    assert "Global Corp" in org_names
    assert "Unverified Agency" not in org_names

    p01 = next(p for p in result.people if p["id"] == "P01")
    # P01 must have link to Global Corp, but NEVER to Unverified Agency!
    assert "links" in p01
    linked_org_ids = [l["organisation"] for l in p01["links"]]
    assert "org_global_corp" in linked_org_ids
    assert len(linked_org_ids) == 1

    # Referential integrity must have 0 errors!
    errors = validate_entities_referential_integrity(
        result=result,
        input_chat_ids={"C01", "C02"},
        input_person_ids={"P01", "P02"},
    )
    assert errors == []


def test_seed_organisations_preserved_unconditionally(tmp_path):
    """Seed organisations remain preserved unconditionally even with 0 chats and 0 members."""
    chats_file = tmp_path / "chats.jsonl"
    msgs_file = tmp_path / "messages.jsonl"
    seed_file = tmp_path / "seed.json"

    chats = [
        {"chat_id": "C01", "name": "Direct Grace", "is_group": False, "participants": ["ME", "P01"]},
    ]
    messages = [
        {"id": "M01", "chat_id": "C01", "sender_id": "P01", "sender_name": "Grace", "kind": "text", "body": "Hello", "ts": 100},
    ]
    # Seed contains an organisation with ZERO chats and ZERO members in conversations
    seed = {
        "circles": {"P01": "friend"},
        "organisations": ["Foundational Seed Holding"],
        "goals": [],
    }

    with open(chats_file, "w", encoding="utf-8") as f:
        for c in chats:
            f.write(json.dumps(c) + "\n")
    with open(msgs_file, "w", encoding="utf-8") as f:
        for m in messages:
            f.write(json.dumps(m) + "\n")
    with open(seed_file, "w", encoding="utf-8") as f:
        json.dump(seed, f)

    context = build_conversation_context(load_dataset(tmp_path))

    result = extract_entities(context=context)

    org_names = [o["name"] for o in result.organisations]
    # Must be preserved by Seed Precedence Rule!
    assert "Foundational Seed Holding" in org_names
    holding = next(o for o in result.organisations if o["name"] == "Foundational Seed Holding")
    assert holding["id"] == "org_foundational_seed_holding"
    assert "Foundational Seed Holding" not in result.diagnostics.pruned_orphan_org_names


def test_example_and_dev_datasets_remain_structurally_valid():
    """Existing example and development dataset behavior remains structurally valid."""
    # Test example dataset
    example_path = Path("example")
    if example_path.exists():
        example_context = build_conversation_context(load_dataset(example_path))
        example_res = extract_entities(example_context)
        example_errors = validate_entities_referential_integrity(
            result=example_res,
            input_chat_ids=set(example_context.chats.keys()),
            input_person_ids=set(p for p in example_context.participants.keys() if p != "ME"),
        )
        assert example_errors == []

    # Test dev dataset
    dev_path = Path("data/dev")
    if dev_path.exists():
        dev_context = build_conversation_context(load_dataset(dev_path))
        dev_res = extract_entities(dev_context)
        dev_errors = validate_entities_referential_integrity(
            result=dev_res,
            input_chat_ids=set(dev_context.chats.keys()),
            input_person_ids=set(p for p in dev_context.participants.keys() if p != "ME"),
        )
        assert dev_errors == []
        assert len(dev_res.chats) == 138


# ==============================================================================
# 9. Targeted Correction 2: Alias Safety, Confidence & Evidence Attributes
# ==============================================================================

# --- Fix 1: Safe Organisation Alias Matching ---

def test_alias_matching_distinct_entities_with_meaningful_descriptors():
    """Ensure business ventures with meaningful descriptors are not falsely merged."""
    assert are_distinct_entities("Brightside Design", "Brightside") is True
    assert are_distinct_entities("Vertex Labs", "Vertex") is True
    assert are_distinct_entities("NEXUS", "Nexus DaaS") is True
    assert are_distinct_entities("Acme Media", "Acme") is True
    assert are_distinct_entities("Apex Technologies", "Apex") is True
    assert are_distinct_entities("Global Studio", "Global") is True


def test_alias_matching_legal_suffixes_strip_and_match():
    """Only genuine corporate legal suffixes are stripped during canonical matching."""
    assert are_distinct_entities("PostMyAd", "PostMyAd Pvt Ltd") is False
    assert are_distinct_entities("PostMyAd Ltd", "PostMyAd") is False
    assert are_distinct_entities("Alpha Corp", "Alpha") is False
    assert are_distinct_entities("Beta LLP", "Beta") is False
    assert are_distinct_entities("Gamma Inc", "Gamma") is False

    assert strip_legal_suffixes("PostMyAd Pvt Ltd") == "PostMyAd"
    assert strip_legal_suffixes("PostMyAd Ltd") == "PostMyAd"
    assert strip_legal_suffixes("PostMyAd Private Limited") == "PostMyAd"
    assert strip_legal_suffixes("Brightside Design") == "Brightside Design"
    assert strip_legal_suffixes("Vertex Labs") == "Vertex Labs"
    assert strip_legal_suffixes("Nexus DaaS") == "Nexus DaaS"


def test_canonical_seed_and_candidate_matching_safety():
    """Test seed matching and candidate lookup preserve distinct ventures while matching aliases."""
    seed_orgs = ["NEXUS", "PostMyAd"]

    # Exact case-insensitive match
    assert match_canonical_seed_org("nexus", seed_orgs) == "NEXUS"
    assert match_canonical_seed_org("postmyad", seed_orgs) == "PostMyAd"

    # Legal suffix match
    assert match_canonical_seed_org("PostMyAd Pvt Ltd", seed_orgs) == "PostMyAd"
    assert match_canonical_seed_org("PostMyAd Ltd", seed_orgs) == "PostMyAd"

    # Distinct venture must NOT match seed or collapse
    assert match_canonical_seed_org("Nexus DaaS", seed_orgs) is None

    # Candidate lookup
    cand_nexus = CandidateOrganisation(name="NEXUS", is_seed=True)
    cand_postmyad = CandidateOrganisation(name="PostMyAd", is_seed=True)
    cand_brightside_design = CandidateOrganisation(name="Brightside Design", is_seed=False)
    candidates = [cand_nexus, cand_postmyad, cand_brightside_design]

    # PostMyAd Pvt Ltd resolves to PostMyAd
    res_pma = find_candidate_org("PostMyAd Pvt Ltd", candidates, seed_org_names=seed_orgs)
    assert res_pma is cand_postmyad

    # Case-insensitive match
    res_ci = find_candidate_org("nexus", candidates, seed_org_names=seed_orgs)
    assert res_ci is cand_nexus

    # Brightside must NOT match Brightside Design
    assert find_candidate_org("Brightside", [cand_brightside_design]) is None
    # Vertex must NOT match Vertex Labs
    cand_vertex_labs = CandidateOrganisation(name="Vertex Labs", is_seed=False)
    assert find_candidate_org("Vertex", [cand_vertex_labs]) is None
    # NEXUS candidate does not match Nexus DaaS mention
    assert find_candidate_org("Nexus DaaS", [cand_nexus]) is None


# --- Fix 2: Confidence Validation for LLM-Derived Associations ---

def test_is_valid_confidence_exhaustive_boundaries():
    """Verify strict confidence validation rejects None, booleans, non-numerics, inf, nan, out-of-range."""
    threshold = 0.65

    # Rejection cases
    assert is_valid_confidence(None, threshold) is False
    assert is_valid_confidence(True, threshold) is False
    assert is_valid_confidence(False, threshold) is False
    assert is_valid_confidence("0.95", threshold) is False
    assert is_valid_confidence("high", threshold) is False
    assert is_valid_confidence([0.95], threshold) is False
    assert is_valid_confidence({"conf": 0.95}, threshold) is False
    assert is_valid_confidence(float("nan"), threshold) is False
    assert is_valid_confidence(float("inf"), threshold) is False
    assert is_valid_confidence(float("-inf"), threshold) is False
    assert is_valid_confidence(-0.1, threshold) is False
    assert is_valid_confidence(1.1, threshold) is False
    assert is_valid_confidence(0.40, threshold) is False
    assert is_valid_confidence(0.6499, threshold) is False

    # Acceptance cases
    assert is_valid_confidence(0.65, threshold) is True
    assert is_valid_confidence(0.70, threshold) is True
    assert is_valid_confidence(0.95, threshold) is True
    assert is_valid_confidence(1.0, threshold) is True


def test_is_valid_llm_records_validation():
    """Verify is_valid_llm_person_record and is_valid_llm_org_link enforce threshold and valid structure."""
    p_valid = InterpretedPerson(
        person_id="P01",
        circle="friend",
        looks_after=[],
        associated_org_names=["Acme Corp"],
        org_link_type="runs",
        evidence_msgs=["M01"],
        confidence=0.85,
    )
    assert is_valid_llm_person_record(p_valid, 0.65) is True
    assert is_valid_llm_org_link(p_valid, 0.65) is True

    # Invalid circle
    p_bad_circle = InterpretedPerson(
        person_id="P01",
        circle="bestie",
        looks_after=[],
        associated_org_names=[],
        org_link_type=None,
        evidence_msgs=["M01"],
        confidence=0.85,
    )
    assert is_valid_llm_person_record(p_bad_circle, 0.65) is False

    # Low confidence
    p_low_conf = InterpretedPerson(
        person_id="P01",
        circle="friend",
        looks_after=[],
        associated_org_names=["Acme Corp"],
        org_link_type="runs",
        evidence_msgs=["M01"],
        confidence=0.40,
    )
    assert is_valid_llm_person_record(p_low_conf, 0.65) is False
    assert is_valid_llm_org_link(p_low_conf, 0.65) is False

    # Boolean confidence
    p_bool_conf = InterpretedPerson(
        person_id="P01",
        circle="friend",
        looks_after=[],
        associated_org_names=[],
        org_link_type=None,
        evidence_msgs=["M01"],
        confidence=True,
    )
    assert is_valid_llm_person_record(p_bool_conf, 0.65) is False

    # Empty evidence msgs
    p_no_ev = InterpretedPerson(
        person_id="P01",
        circle="friend",
        looks_after=[],
        associated_org_names=[],
        org_link_type=None,
        evidence_msgs=[],
        confidence=0.90,
    )
    assert is_valid_llm_person_record(p_no_ev, 0.65) is False


def test_seed_and_lexical_circles_work_without_llm_confidence(tmp_path):
    """Seed and lexical circle resolutions succeed even when LLM confidence is missing or invalid."""
    chats_file = tmp_path / "chats.jsonl"
    msgs_file = tmp_path / "messages.jsonl"
    seed_file = tmp_path / "seed.json"

    chats = [
        {"chat_id": "C01", "name": "Direct Alice", "is_group": False, "participants": ["ME", "P01"]},
        {"chat_id": "C02", "name": "Direct Mummy", "is_group": False, "participants": ["ME", "P02"]},
    ]
    messages = [
        {"id": "M01", "chat_id": "C01", "sender_id": "P01", "sender_name": "Alice", "kind": "text", "body": "Hey there", "ts": 100},
        {"id": "M02", "chat_id": "C02", "sender_id": "P02", "sender_name": "Mummy", "kind": "text", "body": "Dinner ready", "ts": 200},
    ]
    # P01 is seeded as friend; P02 has lexical 'Mummy' (parent)
    seed = {
        "circles": {"P01": "friend"},
        "organisations": [],
        "goals": [],
    }

    with open(chats_file, "w", encoding="utf-8") as f:
        for c in chats:
            f.write(json.dumps(c) + "\n")
    with open(msgs_file, "w", encoding="utf-8") as f:
        for m in messages:
            f.write(json.dumps(m) + "\n")
    with open(seed_file, "w", encoding="utf-8") as f:
        json.dump(seed, f)

    context = build_conversation_context(load_dataset(tmp_path))

    # P01 has invalid LLM confidence (None or boolean), P02 has no LLM record
    interpreted_people = [
        InterpretedPerson(
            person_id="P01",
            circle="vendor",  # LLM says vendor, but seed is friend and conf is None!
            looks_after=[],
            associated_org_names=[],
            org_link_type=None,
            evidence_msgs=["M01"],
            confidence=None,
        )
    ]

    result = extract_entities(context=context, interpreted_people=interpreted_people)
    people_by_id = {p["id"]: p for p in result.people}

    # P01 resolved via seed
    assert "P01" in people_by_id
    assert people_by_id["P01"]["circle"] == "friend"

    # P02 resolved via lexical indicator (Mummy -> parent)
    assert "P02" in people_by_id
    assert people_by_id["P02"]["circle"] == "parent"


def test_invalid_llm_confidence_cannot_keep_organisation_alive(tmp_path):
    """An ungrounded organisation referenced only by low-confidence or non-numeric LLM link is pruned."""
    chats_file = tmp_path / "chats.jsonl"
    msgs_file = tmp_path / "messages.jsonl"
    seed_file = tmp_path / "seed.json"

    chats = [
        {"chat_id": "C01", "name": "Direct Alice", "is_group": False, "participants": ["ME", "P01"]},
    ]
    messages = [
        {"id": "M01", "chat_id": "C01", "sender_id": "P01", "sender_name": "Alice", "kind": "text", "body": "Hello", "ts": 100},
    ]
    seed = {
        "circles": {"P01": "friend"},
        "organisations": [],
        "goals": [],
    }

    with open(chats_file, "w", encoding="utf-8") as f:
        for c in chats:
            f.write(json.dumps(c) + "\n")
    with open(msgs_file, "w", encoding="utf-8") as f:
        for m in messages:
            f.write(json.dumps(m) + "\n")
    with open(seed_file, "w", encoding="utf-8") as f:
        json.dump(seed, f)

    context = build_conversation_context(load_dataset(tmp_path))

    # Candidate org discovered via low-confidence LLM org record (0.40 < 0.70)
    interpreted_orgs = [
        InterpretedOrganisation(
            name="Phantom Ventures",
            org_type="company",
            side="work",
            role="member",
            evidence_msgs=[],
            associated_chats=[],
            confidence=0.40,
        )
    ]
    # Person attempts to link to Phantom Ventures with invalid confidence (None)
    interpreted_people = [
        InterpretedPerson(
            person_id="P01",
            circle="friend",
            looks_after=[],
            associated_org_names=["Phantom Ventures"],
            org_link_type="runs",
            evidence_msgs=["M01"],
            confidence=None,
        )
    ]

    result = extract_entities(
        context=context,
        interpreted_people=interpreted_people,
        interpreted_orgs=interpreted_orgs,
    )

    org_names = [o["name"] for o in result.organisations]
    # Phantom Ventures must NOT survive
    assert "Phantom Ventures" not in org_names
    p01 = next(p for p in result.people if p["id"] == "P01")
    assert "links" not in p01


def test_org_link_from_invalid_llm_association_omitted_even_if_person_kept(tmp_path):
    """If a person is validly extracted, but their org link has invalid confidence, the link is omitted."""
    chats_file = tmp_path / "chats.jsonl"
    msgs_file = tmp_path / "messages.jsonl"
    seed_file = tmp_path / "seed.json"

    chats = [
        {"chat_id": "C01", "name": "Direct Bob", "is_group": False, "participants": ["ME", "P01"]},
        {"chat_id": "C02", "name": "Acme Team", "is_group": True, "participants": ["ME", "P02", "P03"]},
    ]
    messages = [
        {"id": "M01", "chat_id": "C01", "sender_id": "P01", "sender_name": "Bob", "kind": "text", "body": "Hey", "ts": 100},
        {"id": "M02", "chat_id": "C02", "sender_id": "P02", "sender_name": "Alice", "kind": "text", "body": "Acme project update", "ts": 200},
    ]
    seed = {
        "circles": {"P01": "friend"},
        "organisations": ["Acme Corp"],
        "goals": [],
    }

    with open(chats_file, "w", encoding="utf-8") as f:
        for c in chats:
            f.write(json.dumps(c) + "\n")
    with open(msgs_file, "w", encoding="utf-8") as f:
        for m in messages:
            f.write(json.dumps(m) + "\n")
    with open(seed_file, "w", encoding="utf-8") as f:
        json.dump(seed, f)

    context = build_conversation_context(load_dataset(tmp_path))

    # P01 is seeded as friend, but their LLM org link has confidence=0.45 (< 0.65 threshold)
    interpreted_people = [
        InterpretedPerson(
            person_id="P01",
            circle="friend",
            looks_after=[],
            associated_org_names=["Acme Corp"],
            org_link_type="runs",
            evidence_msgs=["M01"],
            confidence=0.45,
        )
    ]

    result = extract_entities(context=context, interpreted_people=interpreted_people)
    p01 = next(p for p in result.people if p["id"] == "P01")

    assert p01["circle"] == "friend"
    # Link must be omitted due to sub-threshold confidence
    assert "links" not in p01


# --- Fix 3: Evidence-Grounded Organisation Attributes ---

def test_unused_seed_org_attributes_default_safely(tmp_path):
    """Unused seed organisation with no goals/messages has type='other', side='home', role='other'."""
    chats_file = tmp_path / "chats.jsonl"
    msgs_file = tmp_path / "messages.jsonl"
    seed_file = tmp_path / "seed.json"

    chats = [
        {"chat_id": "C01", "name": "Direct Dave", "is_group": False, "participants": ["ME", "P01"]},
    ]
    messages = [
        {"id": "M01", "chat_id": "C01", "sender_id": "P01", "sender_name": "Dave", "kind": "text", "body": "Hi", "ts": 100},
    ]
    # Seed org with no goals, messages, or corporate suffix
    seed = {
        "circles": {"P01": "friend"},
        "organisations": ["Inactive Foundation"],
        "goals": [],
    }

    with open(chats_file, "w", encoding="utf-8") as f:
        for c in chats:
            f.write(json.dumps(c) + "\n")
    with open(msgs_file, "w", encoding="utf-8") as f:
        for m in messages:
            f.write(json.dumps(m) + "\n")
    with open(seed_file, "w", encoding="utf-8") as f:
        json.dump(seed, f)

    context = build_conversation_context(load_dataset(tmp_path))
    result = extract_entities(context=context)

    inactive_org = next(o for o in result.organisations if o["name"] == "Inactive Foundation")
    assert inactive_org["type"] == "other"
    assert inactive_org["side"] == "home"
    assert inactive_org["role"] == "other"


def test_org_with_no_ownership_evidence_is_not_owner(tmp_path):
    """An organisation with normal commercial work evidence receives role='member' or 'other', never 'owner'."""
    chats_file = tmp_path / "chats.jsonl"
    msgs_file = tmp_path / "messages.jsonl"
    seed_file = tmp_path / "seed.json"

    chats = [
        {"chat_id": "C01", "name": "Vendor Sync", "is_group": True, "participants": ["ME", "P01"]},
    ]
    messages = [
        {"id": "M01", "chat_id": "C01", "sender_id": "P01", "sender_name": "Supplier", "kind": "text", "body": "Vendor invoice sent for review", "ts": 100},
    ]
    # Seed org has work area, but NO founder/scaling/ownership goal
    seed = {
        "circles": {"P01": "vendor"},
        "organisations": ["Supplier Logistics"],
        "goals": [{"title": "Review Supplier Logistics contract", "area": "work"}],
    }

    with open(chats_file, "w", encoding="utf-8") as f:
        for c in chats:
            f.write(json.dumps(c) + "\n")
    with open(msgs_file, "w", encoding="utf-8") as f:
        for m in messages:
            f.write(json.dumps(m) + "\n")
    with open(seed_file, "w", encoding="utf-8") as f:
        json.dump(seed, f)

    context = build_conversation_context(load_dataset(tmp_path))
    result = extract_entities(context=context)

    supp_org = next(o for o in result.organisations if o["name"] == "Supplier Logistics")
    assert supp_org["side"] == "work"
    assert supp_org["type"] == "company"
    # Must NOT be owner!
    assert supp_org["role"] != "owner"
    assert supp_org["role"] == "member"


def test_explicit_ownership_evidence_produces_owner(tmp_path):
    """Seed goal scaling/clients or explicit founder message produces role='owner'."""
    chats_file = tmp_path / "chats.jsonl"
    msgs_file = tmp_path / "messages.jsonl"
    seed_file = tmp_path / "seed.json"

    chats = [
        {"chat_id": "C01", "name": "Brightside Sync", "is_group": True, "participants": ["ME", "P01"]},
        {"chat_id": "C02", "name": "Startup Chat", "is_group": True, "participants": ["ME", "P02"]},
    ]
    messages = [
        {"id": "M01", "chat_id": "C01", "sender_id": "P01", "sender_name": "Alice", "kind": "text", "body": "Design sprint ready", "ts": 100},
        {"id": "M02", "chat_id": "C02", "sender_id": "ME", "sender_name": "Arjun", "kind": "text", "body": "I founded Alpha Media last year", "ts": 200},
    ]
    # Brightside has explicit scaling goal; Alpha Media has explicit founder message
    seed = {
        "circles": {"P01": "colleague", "P02": "partner"},
        "organisations": ["Brightside Design", "Alpha Media"],
        "goals": [{"title": "Scale Brightside Design to 10 clients", "area": "work"}],
    }

    with open(chats_file, "w", encoding="utf-8") as f:
        for c in chats:
            f.write(json.dumps(c) + "\n")
    with open(msgs_file, "w", encoding="utf-8") as f:
        for m in messages:
            f.write(json.dumps(m) + "\n")
    with open(seed_file, "w", encoding="utf-8") as f:
        json.dump(seed, f)

    context = build_conversation_context(load_dataset(tmp_path))
    result = extract_entities(context=context)

    bs = next(o for o in result.organisations if o["name"] == "Brightside Design")
    assert bs["role"] == "owner"
    assert bs["side"] == "work"
    assert bs["type"] == "company"

    am = next(o for o in result.organisations if o["name"] == "Alpha Media")
    assert am["role"] == "owner"
    assert am["side"] == "work"


def test_home_residence_evidence_grounded(tmp_path):
    """'Home' organisation is created ONLY when domestic evidence exists; role='owner' ONLY with explicit ownership evidence."""
    chats_file = tmp_path / "chats.jsonl"
    msgs_file = tmp_path / "messages.jsonl"
    seed_file = tmp_path / "seed.json"

    # Dataset 1: No domestic chats or messages
    chats_no_home = [
        {"chat_id": "C01", "name": "Work Project", "is_group": True, "participants": ["ME", "P01"]},
    ]
    messages_no_home = [
        {"id": "M01", "chat_id": "C01", "sender_id": "P01", "sender_name": "Alice", "kind": "text", "body": "Project milestone met", "ts": 100},
    ]
    seed = {"circles": {"P01": "colleague"}, "organisations": [], "goals": []}

    with open(chats_file, "w", encoding="utf-8") as f:
        for c in chats_no_home:
            f.write(json.dumps(c) + "\n")
    with open(msgs_file, "w", encoding="utf-8") as f:
        for m in messages_no_home:
            f.write(json.dumps(m) + "\n")
    with open(seed_file, "w", encoding="utf-8") as f:
        json.dump(seed, f)

    context_no_home = build_conversation_context(load_dataset(tmp_path))
    result_no_home = extract_entities(context=context_no_home)
    assert not any(o["name"] == "Home" for o in result_no_home.organisations)

    # Dataset 2: Chat titled 'Flat maintenance' alone (MUST NOT produce role='owner')
    chats_flat_maint = [
        {"chat_id": "C01", "name": "Flat maintenance", "is_group": True, "participants": ["ME", "P01"]},
    ]
    messages_flat_maint = [
        {"id": "M01", "chat_id": "C01", "sender_id": "P01", "sender_name": "Alice", "kind": "text", "body": "Maintenance fee due", "ts": 100},
    ]
    with open(chats_file, "w", encoding="utf-8") as f:
        for c in chats_flat_maint:
            f.write(json.dumps(c) + "\n")
    with open(msgs_file, "w", encoding="utf-8") as f:
        for m in messages_flat_maint:
            f.write(json.dumps(m) + "\n")

    context_flat = build_conversation_context(load_dataset(tmp_path))
    result_flat = extract_entities(context=context_flat)
    home_flat = next((o for o in result_flat.organisations if o["name"] == "Home"), None)
    assert home_flat is not None
    assert home_flat["type"] == "residence"
    assert home_flat["side"] == "home"
    assert home_flat["role"] != "owner"
    assert home_flat["role"] == "other"
    # Existing chat mapping still maps to Home
    chat_flat = next(c for c in result_flat.chats if c["chat_id"] == "C01")
    assert chat_flat.get("organisation") == home_flat["id"]

    # Dataset 3: Chat titled 'Ghar' alone (MUST NOT produce role='owner')
    chats_ghar = [
        {"chat_id": "C01", "name": "Ghar", "is_group": True, "participants": ["ME", "P01"]},
    ]
    messages_ghar = [
        {"id": "M01", "chat_id": "C01", "sender_id": "P01", "sender_name": "Alice", "kind": "text", "body": "Dinner ready at 8", "ts": 100},
    ]
    with open(chats_file, "w", encoding="utf-8") as f:
        for c in chats_ghar:
            f.write(json.dumps(c) + "\n")
    with open(msgs_file, "w", encoding="utf-8") as f:
        for m in messages_ghar:
            f.write(json.dumps(m) + "\n")

    context_ghar = build_conversation_context(load_dataset(tmp_path))
    result_ghar = extract_entities(context=context_ghar)
    home_ghar = next((o for o in result_ghar.organisations if o["name"] == "Home"), None)
    assert home_ghar is not None
    assert home_ghar["type"] == "residence"
    assert home_ghar["side"] == "home"
    assert home_ghar["role"] != "owner"
    assert home_ghar["role"] == "other"

    # Dataset 4: Explicit ownership language ("my flat" / "I own this property" / "I own the flat") produces role='owner'
    chats_owner = [
        {"chat_id": "C01", "name": "Flat maintenance", "is_group": True, "participants": ["ME", "P01"]},
    ]
    messages_owner = [
        {"id": "M01", "chat_id": "C01", "sender_id": "ME", "sender_name": "Arjun", "kind": "text", "body": "I own this property and maintenance fee is cleared", "ts": 100},
    ]
    with open(chats_file, "w", encoding="utf-8") as f:
        for c in chats_owner:
            f.write(json.dumps(c) + "\n")
    with open(msgs_file, "w", encoding="utf-8") as f:
        for m in messages_owner:
            f.write(json.dumps(m) + "\n")

    context_owner = build_conversation_context(load_dataset(tmp_path))
    result_owner = extract_entities(context=context_owner)
    home_owner = next((o for o in result_owner.organisations if o["name"] == "Home"), None)
    assert home_owner is not None
    assert home_owner["type"] == "residence"
    assert home_owner["side"] == "home"
    assert home_owner["role"] == "owner"

    # Dataset 5: Home chat mentioning business ownership ("I founded Alpha Media") MUST NOT change Home to company or owner
    chats_home_biz = [
        {"chat_id": "C01", "name": "Flat maintenance", "is_group": True, "participants": ["ME", "P01"]},
    ]
    messages_home_biz = [
        {"id": "M01", "chat_id": "C01", "sender_id": "ME", "sender_name": "Arjun", "kind": "text", "body": "I founded Alpha Media last year, please coordinate with society", "ts": 100},
    ]
    with open(chats_file, "w", encoding="utf-8") as f:
        for c in chats_home_biz:
            f.write(json.dumps(c) + "\n")
    with open(msgs_file, "w", encoding="utf-8") as f:
        for m in messages_home_biz:
            f.write(json.dumps(m) + "\n")

    context_home_biz = build_conversation_context(load_dataset(tmp_path))
    result_home_biz = extract_entities(context=context_home_biz)
    home_biz = next((o for o in result_home_biz.organisations if o["name"] == "Home"), None)
    assert home_biz is not None
    assert home_biz["type"] == "residence"
    assert home_biz["side"] == "home"
    assert home_biz["role"] == "other"

    # Dataset 6: Home chat mentioning invoice/client MUST NOT change Home to company or member
    chats_home_inv = [
        {"chat_id": "C01", "name": "Ghar", "is_group": True, "participants": ["ME", "P01"]},
    ]
    messages_home_inv = [
        {"id": "M01", "chat_id": "C01", "sender_id": "P01", "sender_name": "Alice", "kind": "text", "body": "Please send the invoice and receipt for the plumber", "ts": 100},
    ]
    with open(chats_file, "w", encoding="utf-8") as f:
        for c in chats_home_inv:
            f.write(json.dumps(c) + "\n")
    with open(msgs_file, "w", encoding="utf-8") as f:
        for m in messages_home_inv:
            f.write(json.dumps(m) + "\n")

    context_home_inv = build_conversation_context(load_dataset(tmp_path))
    result_home_inv = extract_entities(context=context_home_inv)
    home_inv = next((o for o in result_home_inv.organisations if o["name"] == "Home"), None)
    assert home_inv is not None
    assert home_inv["type"] == "residence"
    assert home_inv["side"] == "home"
    assert home_inv["role"] == "other"

    # Dataset 7: Explicit residential ownership with "I own the flat"
    chats_own_flat = [
        {"chat_id": "C01", "name": "Ghar", "is_group": True, "participants": ["ME", "P01"]},
    ]
    messages_own_flat = [
        {"id": "M01", "chat_id": "C01", "sender_id": "ME", "sender_name": "Arjun", "kind": "text", "body": "I own the flat so I will handle the repairs", "ts": 100},
    ]
    with open(chats_file, "w", encoding="utf-8") as f:
        for c in chats_own_flat:
            f.write(json.dumps(c) + "\n")
    with open(msgs_file, "w", encoding="utf-8") as f:
        for m in messages_own_flat:
            f.write(json.dumps(m) + "\n")

    context_own_flat = build_conversation_context(load_dataset(tmp_path))
    result_own_flat = extract_entities(context=context_own_flat)
    home_own_flat = next((o for o in result_own_flat.organisations if o["name"] == "Home"), None)
    assert home_own_flat is not None
    assert home_own_flat["type"] == "residence"
    assert home_own_flat["side"] == "home"
    assert home_own_flat["role"] == "owner"

