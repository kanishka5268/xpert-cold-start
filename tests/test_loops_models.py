"""Tests for open loop models, HelpAction contracts, and schema serialization.

Verifies Milestone 1 data models:
- HelpAction strict validation and action-specific requirements.
- Strict UTC instant enforcement ('Z' or '+00:00' strictly, rejecting '-00:00', naive, and non-zero offsets).
- CandidateOpenLoop lifecycle states (open, done, dropped) and invariants.
- Exact byte-for-byte identifier preservation (no .strip() normalization).
- Exact 9-property schema serialization and exclusion of internal fields.
- OpenLoopExtractionConfig and OpenLoopExtractionDiagnostics defaults.
"""

from datetime import datetime
import pytest

from src.loops.models import (
    VALID_HELP_ACTIONS,
    VALID_OPEN_LOOP_STATUSES,
    VALID_OPEN_LOOP_TYPES,
    CandidateOpenLoop,
    HelpAction,
    OpenLoopExtractionConfig,
    OpenLoopExtractionDiagnostics,
)


# ============================================================================
# HelpAction Tests
# ============================================================================


def test_help_action_valid_assist():
    """Verify valid assist action with non-empty next_step and boolean needs_yes."""
    action = HelpAction(
        action="assist",
        next_step="Draft follow-up email to client",
        needs_yes=True,
    )
    assert action.validate() is True
    schema = action.to_schema_dict()
    assert schema == {
        "action": "assist",
        "next_step": "Draft follow-up email to client",
        "needs_yes": True,
    }

    action_no_yes = HelpAction(
        action="assist",
        next_step="Search for flights to Bengaluru",
        needs_yes=False,
    )
    assert action_no_yes.validate() is True
    assert action_no_yes.to_schema_dict()["needs_yes"] is False

    # Surrounding whitespace in next_step is trimmed during serialization
    action_trimmed = HelpAction(
        action="assist",
        next_step="  Prepare agenda notes  ",
        needs_yes=False,
    )
    assert action_trimmed.validate() is True
    assert action_trimmed.to_schema_dict()["next_step"] == "Prepare agenda notes"


def test_help_action_valid_remind_z():
    """Verify valid remind action with ISO-8601 UTC timestamp ending in 'Z'."""
    action = HelpAction(
        action="remind",
        remind_at="2025-04-10T14:30:00Z",
    )
    assert action.validate() is True
    schema = action.to_schema_dict()
    assert schema == {
        "action": "remind",
        "remind_at": "2025-04-10T14:30:00Z",
    }

    # Microsecond precision with Z is also valid
    action_micro = HelpAction(
        action="remind",
        remind_at="2025-04-10T14:30:00.123456Z",
    )
    assert action_micro.validate() is True


def test_help_action_valid_remind_plus_zero():
    """Verify valid remind action with ISO-8601 UTC timestamp ending in '+00:00'."""
    action = HelpAction(
        action="remind",
        remind_at="2025-04-10T14:30:00+00:00",
    )
    assert action.validate() is True
    schema = action.to_schema_dict()
    assert schema == {
        "action": "remind",
        "remind_at": "2025-04-10T14:30:00+00:00",
    }


def test_help_action_reminder_whitespace_rejected():
    """Verify remind action strictly rejects leading or trailing whitespace on remind_at."""
    # Leading whitespace before an otherwise valid UTC timestamp
    assert HelpAction(action="remind", remind_at=" 2025-04-10T14:30:00Z").validate() is False
    assert HelpAction(action="remind", remind_at="\t2025-04-10T14:30:00Z").validate() is False
    assert HelpAction(action="remind", remind_at=" 2025-04-10T14:30:00+00:00").validate() is False
    assert HelpAction(action="remind", remind_at="\n2025-04-10T14:30:00+00:00").validate() is False

    # Trailing whitespace after Z
    assert HelpAction(action="remind", remind_at="2025-04-10T14:30:00Z ").validate() is False
    assert HelpAction(action="remind", remind_at="2025-04-10T14:30:00Z\t").validate() is False
    assert HelpAction(action="remind", remind_at="2025-04-10T14:30:00Z\n").validate() is False

    # Trailing whitespace after +00:00
    assert HelpAction(action="remind", remind_at="2025-04-10T14:30:00+00:00 ").validate() is False
    assert HelpAction(action="remind", remind_at="2025-04-10T14:30:00+00:00\t").validate() is False
    assert HelpAction(action="remind", remind_at="2025-04-10T14:30:00+00:00\n").validate() is False

    # Both leading and trailing whitespace
    assert HelpAction(action="remind", remind_at=" 2025-04-10T14:30:00Z ").validate() is False

    # Verify serialization does NOT normalize whitespace; it raises ValueError
    padded_action = HelpAction(action="remind", remind_at=" 2025-04-10T14:30:00Z")
    with pytest.raises(ValueError, match="Invalid HelpAction"):
        padded_action.to_schema_dict()

    # Existing valid Z and +00:00 timestamps still pass and emit byte-for-byte unchanged
    valid_z = HelpAction(action="remind", remind_at="2025-04-10T14:30:00Z")
    assert valid_z.validate() is True
    assert valid_z.to_schema_dict()["remind_at"] == "2025-04-10T14:30:00Z"

    valid_plus_zero = HelpAction(action="remind", remind_at="2025-04-10T14:30:00+00:00")
    assert valid_plus_zero.validate() is True
    assert valid_plus_zero.to_schema_dict()["remind_at"] == "2025-04-10T14:30:00+00:00"


def test_help_action_valid_leave():
    """Verify valid leave action with non-empty reason explaining non-intervention."""
    action = HelpAction(
        action="leave",
        reason="Obligation owned by Vikram; external party task",
    )
    assert action.validate() is True
    schema = action.to_schema_dict()
    assert schema == {
        "action": "leave",
        "reason": "Obligation owned by Vikram; external party task",
    }

    # Surrounding whitespace in reason is trimmed during serialization
    action_trimmed = HelpAction(
        action="leave",
        reason="  Personal chat between third parties  ",
    )
    assert action_trimmed.validate() is True
    assert action_trimmed.to_schema_dict()["reason"] == "Personal chat between third parties"


def test_help_action_missing_required_fields():
    """Verify HelpAction rejects missing required action-specific fields."""
    # Assist missing next_step
    assert HelpAction(action="assist", next_step=None, needs_yes=True).validate() is False
    # Assist missing needs_yes
    assert HelpAction(action="assist", next_step="Review doc", needs_yes=None).validate() is False
    # Remind missing remind_at
    assert HelpAction(action="remind", remind_at=None).validate() is False
    # Leave missing reason
    assert HelpAction(action="leave", reason=None).validate() is False
    # Unknown action
    assert HelpAction(action="unknown").validate() is False
    assert HelpAction(action="").validate() is False


def test_help_action_incorrect_field_types():
    """Verify HelpAction rejects incorrect field types."""
    assert HelpAction(action=123).validate() is False  # type: ignore[arg-type]
    # next_step must be str
    assert HelpAction(action="assist", next_step=123, needs_yes=True).validate() is False  # type: ignore[arg-type]
    # needs_yes must be strictly bool (not str or int)
    assert HelpAction(action="assist", next_step="Review", needs_yes="true").validate() is False  # type: ignore[arg-type]
    assert HelpAction(action="assist", next_step="Review", needs_yes=1).validate() is False  # type: ignore[arg-type]
    assert HelpAction(action="assist", next_step="Review", needs_yes=0).validate() is False  # type: ignore[arg-type]
    # remind_at must be str
    assert HelpAction(action="remind", remind_at=1787535062).validate() is False  # type: ignore[arg-type]
    # reason must be str
    assert HelpAction(action="leave", reason=999).validate() is False  # type: ignore[arg-type]
    assert HelpAction(action="leave", reason=["Not me"]).validate() is False  # type: ignore[arg-type]


def test_help_action_empty_or_whitespace_only_next_step():
    """Verify assist action rejects empty, whitespace-only, or multiline next_step (newlines and carriage returns)."""
    assert HelpAction(action="assist", next_step="", needs_yes=True).validate() is False
    assert HelpAction(action="assist", next_step="   ", needs_yes=True).validate() is False
    assert HelpAction(action="assist", next_step="\t\n", needs_yes=True).validate() is False
    # Multiline next_step is strictly rejected (must be single-line)
    assert HelpAction(action="assist", next_step="Step 1\nStep 2", needs_yes=True).validate() is False
    assert HelpAction(action="assist", next_step="Step 1\rStep 2", needs_yes=True).validate() is False
    assert HelpAction(action="assist", next_step="Step 1\r\nStep 2", needs_yes=True).validate() is False
    assert HelpAction(action="assist", next_step="Step 1\r", needs_yes=True).validate() is False
    assert HelpAction(action="assist", next_step="Step 1\n", needs_yes=True).validate() is False

    # Valid single-line next_step passes
    assert HelpAction(
        action="assist",
        next_step="Draft follow-up email to client regarding contract terms",
        needs_yes=True,
    ).validate() is True


def test_help_action_empty_or_whitespace_only_reason():
    """Verify leave action rejects empty or whitespace-only reason."""
    assert HelpAction(action="leave", reason="").validate() is False
    assert HelpAction(action="leave", reason="   ").validate() is False
    assert HelpAction(action="leave", reason="\t\n").validate() is False


def test_help_action_invalid_reminder_datetime():
    """Verify remind action rejects malformed or unparseable datetimes."""
    # Malformed strings
    assert HelpAction(action="remind", remind_at="").validate() is False
    assert HelpAction(action="remind", remind_at="   ").validate() is False
    assert HelpAction(action="remind", remind_at="not-a-datetimeZ").validate() is False
    # Impossible calendar dates
    assert HelpAction(action="remind", remind_at="2025-02-30T10:00:00Z").validate() is False
    assert HelpAction(action="remind", remind_at="2025-13-01T10:00:00Z").validate() is False
    # Impossible clock values
    assert HelpAction(action="remind", remind_at="2025-01-01T25:00:00Z").validate() is False
    assert HelpAction(action="remind", remind_at="2025-01-01T10:65:00Z").validate() is False
    # Date only without time component
    assert HelpAction(action="remind", remind_at="2025-01-01Z").validate() is False


def test_help_action_naive_datetime_rejected():
    """Verify remind action strictly rejects naive datetimes without timezone."""
    assert HelpAction(action="remind", remind_at="2025-03-15T09:00:00").validate() is False
    assert HelpAction(action="remind", remind_at="2025-03-15 09:00:00").validate() is False


def test_help_action_non_zero_offset_rejected():
    """Verify remind action strictly rejects non-zero UTC offsets."""
    assert HelpAction(action="remind", remind_at="2025-03-15T09:00:00+05:30").validate() is False
    assert HelpAction(action="remind", remind_at="2025-03-15T09:00:00-05:00").validate() is False
    assert HelpAction(action="remind", remind_at="2025-03-15T09:00:00+01:00").validate() is False
    assert HelpAction(action="remind", remind_at="2025-03-15T09:00:00-08:00").validate() is False


def test_help_action_negative_zero_offset_rejected():
    """Verify remind action rejects '-00:00' (unknown local offset marker per RFC 3339)."""
    assert HelpAction(action="remind", remind_at="2025-03-15T09:00:00-00:00").validate() is False
    assert HelpAction(action="remind", remind_at="2025-01-01T00:00:00-00:00").validate() is False


def test_help_action_out_of_range_year():
    """Verify remind action strictly enforces the 1970-2100 astronomical calendar bounds."""
    # Before 1970
    assert HelpAction(action="remind", remind_at="1969-12-31T23:59:59Z").validate() is False
    assert HelpAction(action="remind", remind_at="1900-01-01T00:00:00Z").validate() is False
    # After 2100
    assert HelpAction(action="remind", remind_at="2101-01-01T00:00:00Z").validate() is False
    assert HelpAction(action="remind", remind_at="2200-01-01T00:00:00Z").validate() is False
    # Valid boundaries
    assert HelpAction(action="remind", remind_at="1970-01-01T00:00:00Z").validate() is True
    assert HelpAction(action="remind", remind_at="2100-12-31T23:59:59Z").validate() is True


def test_help_action_invalid_action_specific_field_combinations():
    """Verify HelpAction rejects invalid combinations of action type and fields."""
    # Assist must not have remind_at or reason
    assert HelpAction(
        action="assist",
        next_step="Draft contract",
        needs_yes=True,
        remind_at="2025-03-15T09:00:00Z",
    ).validate() is False
    assert HelpAction(
        action="assist",
        next_step="Draft contract",
        needs_yes=True,
        reason="Some reason",
    ).validate() is False

    # Remind must not have next_step, needs_yes, or reason
    assert HelpAction(
        action="remind",
        remind_at="2025-03-15T09:00:00Z",
        next_step="Call person",
    ).validate() is False
    assert HelpAction(
        action="remind",
        remind_at="2025-03-15T09:00:00Z",
        needs_yes=False,
    ).validate() is False
    assert HelpAction(
        action="remind",
        remind_at="2025-03-15T09:00:00Z",
        reason="Some reason",
    ).validate() is False

    # Leave must not have next_step, needs_yes, or remind_at
    assert HelpAction(
        action="leave",
        reason="Third party chat",
        next_step="Do task",
    ).validate() is False
    assert HelpAction(
        action="leave",
        reason="Third party chat",
        needs_yes=True,
    ).validate() is False
    assert HelpAction(
        action="leave",
        reason="Third party chat",
        remind_at="2025-03-15T09:00:00Z",
    ).validate() is False


def test_help_action_to_schema_dict_raises_on_invalid():
    """Verify HelpAction.to_schema_dict() raises ValueError when invalid."""
    invalid_actions = [
        HelpAction(action="assist", next_step="", needs_yes=True),
        HelpAction(action="remind", remind_at="2025-01-01T00:00:00"),  # naive
        HelpAction(action="remind", remind_at="2025-01-01T00:00:00-00:00"),  # -00:00
        HelpAction(action="leave", reason=""),
        HelpAction(action="unknown"),
    ]
    for action in invalid_actions:
        with pytest.raises(ValueError, match="Invalid HelpAction"):
            action.to_schema_dict()


# ============================================================================
# CandidateOpenLoop Tests
# ============================================================================


def test_candidate_open_loop_valid_open():
    """Verify valid open loop with HelpAction, due date, and closed_by=None."""
    help_act = HelpAction(action="assist", next_step="Review the deck", needs_yes=False)
    loop = CandidateOpenLoop(
        title="Review marketing proposal",
        type="commitment",
        owner="ME",
        chat_id="C01",
        source_msg="M001",
        due="2025-04-01",
        status="open",
        closed_by=None,
        help=help_act,
    )
    assert loop.validate() is True
    schema = loop.to_schema_dict()
    assert schema == {
        "title": "Review marketing proposal",
        "type": "commitment",
        "owner": "ME",
        "chat_id": "C01",
        "source_msg": "M001",
        "due": "2025-04-01",
        "status": "open",
        "closed_by": None,
        "help": {
            "action": "assist",
            "next_step": "Review the deck",
            "needs_yes": False,
        },
    }


def test_candidate_open_loop_valid_done():
    """Verify valid done loop with valid closed_by and help=None."""
    loop = CandidateOpenLoop(
        title="Send invoice to accounting",
        type="commitment",
        owner="P01",
        chat_id="C01",
        source_msg="M001",
        due=None,
        status="done",
        closed_by="M005",
        help=None,
    )
    assert loop.validate() is True
    schema = loop.to_schema_dict()
    assert schema == {
        "title": "Send invoice to accounting",
        "type": "commitment",
        "owner": "P01",
        "chat_id": "C01",
        "source_msg": "M001",
        "due": None,
        "status": "done",
        "closed_by": "M005",
        "help": None,
    }


def test_candidate_open_loop_valid_dropped():
    """Verify valid dropped loop with valid closed_by and help=None."""
    loop = CandidateOpenLoop(
        title="Book court for pickleball",
        type="followup",
        owner="ME",
        chat_id="C02",
        source_msg="M010",
        due="2025-05-15",
        status="dropped",
        closed_by="M015",
        help=None,
    )
    assert loop.validate() is True
    schema = loop.to_schema_dict()
    assert schema == {
        "title": "Book court for pickleball",
        "type": "followup",
        "owner": "ME",
        "chat_id": "C02",
        "source_msg": "M010",
        "due": "2025-05-15",
        "status": "dropped",
        "closed_by": "M015",
        "help": None,
    }


def test_candidate_open_loop_invalid_status_and_type():
    """Verify CandidateOpenLoop rejects invalid type or status enums."""
    valid_help = HelpAction(action="leave", reason="Not owned by ME")

    # Invalid status
    assert CandidateOpenLoop(
        title="Valid title",
        type="commitment",
        owner="ME",
        chat_id="C01",
        source_msg="M01",
        status="pending",  # invalid
        help=valid_help,
    ).validate() is False

    # Invalid type
    assert CandidateOpenLoop(
        title="Valid title",
        type="chore",  # invalid
        owner="ME",
        chat_id="C01",
        source_msg="M01",
        status="open",
        help=valid_help,
    ).validate() is False

    # Verify all approved types pass
    for loop_type in VALID_OPEN_LOOP_TYPES:
        loop = CandidateOpenLoop(
            title="Valid title",
            type=loop_type,
            owner="ME",
            chat_id="C01",
            source_msg="M01",
            status="open",
            help=valid_help,
        )
        assert loop.validate() is True


def test_candidate_open_loop_invalid_or_empty_identifiers():
    """Verify CandidateOpenLoop rejects missing, non-string, or empty identifiers."""
    valid_help = HelpAction(action="leave", reason="Not me")

    # Empty owner
    assert CandidateOpenLoop(
        title="Task", type="commitment", owner="", chat_id="C01", source_msg="M01", help=valid_help
    ).validate() is False

    # Non-string owner
    assert CandidateOpenLoop(
        title="Task", type="commitment", owner=None, chat_id="C01", source_msg="M01", help=valid_help  # type: ignore[arg-type]
    ).validate() is False

    # Empty chat_id
    assert CandidateOpenLoop(
        title="Task", type="commitment", owner="ME", chat_id="", source_msg="M01", help=valid_help
    ).validate() is False

    # Empty source_msg
    assert CandidateOpenLoop(
        title="Task", type="commitment", owner="ME", chat_id="C01", source_msg="", help=valid_help
    ).validate() is False


def test_candidate_open_loop_padded_identifiers_rejected():
    """Verify CandidateOpenLoop strictly rejects identifiers with leading/trailing whitespace."""
    valid_help = HelpAction(action="leave", reason="Not me")

    padded_cases = [
        # owner
        {"owner": " ME", "chat_id": "C01", "source_msg": "M01"},
        {"owner": "ME ", "chat_id": "C01", "source_msg": "M01"},
        {"owner": " P01", "chat_id": "C01", "source_msg": "M01"},
        {"owner": "P01 ", "chat_id": "C01", "source_msg": "M01"},
        {"owner": "\tME", "chat_id": "C01", "source_msg": "M01"},
        # chat_id
        {"owner": "ME", "chat_id": " C01", "source_msg": "M01"},
        {"owner": "ME", "chat_id": "C01 ", "source_msg": "M01"},
        {"owner": "ME", "chat_id": "\nC01", "source_msg": "M01"},
        # source_msg
        {"owner": "ME", "chat_id": "C01", "source_msg": " M01"},
        {"owner": "ME", "chat_id": "C01", "source_msg": "M01 "},
        {"owner": "ME", "chat_id": "C01", "source_msg": "\tM01"},
        {"owner": "ME", "chat_id": "C01", "source_msg": "M01\n"},
    ]

    for kwargs in padded_cases:
        loop = CandidateOpenLoop(
            title="Task",
            type="commitment",
            status="open",
            help=valid_help,
            **kwargs,
        )
        assert loop.validate() is False
        with pytest.raises(ValueError, match="Invalid CandidateOpenLoop"):
            loop.to_schema_dict()

    # closed_by on done loop
    done_padded = [
        CandidateOpenLoop(
            title="Task", type="commitment", owner="ME", chat_id="C01", source_msg="M01",
            status="done", closed_by=" M02"
        ),
        CandidateOpenLoop(
            title="Task", type="commitment", owner="ME", chat_id="C01", source_msg="M01",
            status="done", closed_by="M02 "
        ),
        CandidateOpenLoop(
            title="Task", type="commitment", owner="ME", chat_id="C01", source_msg="M01",
            status="done", closed_by="\tM02\n"
        ),
    ]
    for loop in done_padded:
        assert loop.validate() is False
        with pytest.raises(ValueError, match="Invalid CandidateOpenLoop"):
            loop.to_schema_dict()


def test_candidate_open_loop_exact_identifier_preservation():
    """Verify serialization preserves exact original identifier values without stripping."""
    valid_help = HelpAction(action="leave", reason="Not me")
    loop = CandidateOpenLoop(
        title="Task",
        type="commitment",
        owner="P01",
        chat_id="C01",
        source_msg="M001",
        status="open",
        help=valid_help,
    )
    schema = loop.to_schema_dict()
    assert schema["owner"] == "P01"
    assert schema["chat_id"] == "C01"
    assert schema["source_msg"] == "M001"

    # Closed loop closed_by preservation
    done_loop = CandidateOpenLoop(
        title="Task",
        type="commitment",
        owner="ME",
        chat_id="C01",
        source_msg="M001",
        status="done",
        closed_by="M009",
    )
    done_schema = done_loop.to_schema_dict()
    assert done_schema["closed_by"] == "M009"


def test_candidate_open_loop_invalid_due_date():
    """Verify CandidateOpenLoop validates due date format and calendar reality."""
    valid_help = HelpAction(action="leave", reason="Not me")

    # Invalid formats
    assert CandidateOpenLoop(
        title="Task", type="commitment", owner="ME", chat_id="C01", source_msg="M01",
        due="2025/04/01", help=valid_help
    ).validate() is False
    assert CandidateOpenLoop(
        title="Task", type="commitment", owner="ME", chat_id="C01", source_msg="M01",
        due="tomorrow", help=valid_help
    ).validate() is False
    assert CandidateOpenLoop(
        title="Task", type="commitment", owner="ME", chat_id="C01", source_msg="M01",
        due="", help=valid_help
    ).validate() is False
    assert CandidateOpenLoop(
        title="Task", type="commitment", owner="ME", chat_id="C01", source_msg="M01",
        due=" 2025-04-01 ", help=valid_help
    ).validate() is False
    assert CandidateOpenLoop(
        title="Task", type="commitment", owner="ME", chat_id="C01", source_msg="M01",
        due=123, help=valid_help  # type: ignore[arg-type]
    ).validate() is False

    # Impossible calendar dates
    assert CandidateOpenLoop(
        title="Task", type="commitment", owner="ME", chat_id="C01", source_msg="M01",
        due="2025-02-30", help=valid_help
    ).validate() is False
    assert CandidateOpenLoop(
        title="Task", type="commitment", owner="ME", chat_id="C01", source_msg="M01",
        due="2025-13-01", help=valid_help
    ).validate() is False

    # Valid due dates
    assert CandidateOpenLoop(
        title="Task", type="commitment", owner="ME", chat_id="C01", source_msg="M01",
        due="2025-04-01", help=valid_help
    ).validate() is True
    assert CandidateOpenLoop(
        title="Task", type="commitment", owner="ME", chat_id="C01", source_msg="M01",
        due=None, help=valid_help
    ).validate() is True


def test_candidate_open_loop_open_loop_invariants():
    """Verify invariants for open loops: closed_by must be None, help must be valid."""
    valid_help = HelpAction(action="leave", reason="Not me")

    # Open loop with closed_by must fail
    assert CandidateOpenLoop(
        title="Task", type="commitment", owner="ME", chat_id="C01", source_msg="M01",
        status="open", closed_by="M02", help=valid_help
    ).validate() is False

    # Open loop without help must fail
    assert CandidateOpenLoop(
        title="Task", type="commitment", owner="ME", chat_id="C01", source_msg="M01",
        status="open", closed_by=None, help=None
    ).validate() is False

    # Open loop with invalid help must fail
    invalid_help = HelpAction(action="assist", next_step="", needs_yes=True)
    assert CandidateOpenLoop(
        title="Task", type="commitment", owner="ME", chat_id="C01", source_msg="M01",
        status="open", closed_by=None, help=invalid_help
    ).validate() is False


def test_candidate_open_loop_closed_loop_invariants():
    """Verify invariants for done/dropped loops: closed_by required, help must be None."""
    valid_help = HelpAction(action="leave", reason="Not me")

    # Done loop with closed_by=None must fail
    assert CandidateOpenLoop(
        title="Task", type="commitment", owner="ME", chat_id="C01", source_msg="M01",
        status="done", closed_by=None, help=None
    ).validate() is False

    # Done loop with empty closed_by must fail
    assert CandidateOpenLoop(
        title="Task", type="commitment", owner="ME", chat_id="C01", source_msg="M01",
        status="done", closed_by="", help=None
    ).validate() is False

    # Done loop with non-null help must fail
    assert CandidateOpenLoop(
        title="Task", type="commitment", owner="ME", chat_id="C01", source_msg="M01",
        status="done", closed_by="M02", help=valid_help
    ).validate() is False

    # Dropped loop with closed_by=None must fail
    assert CandidateOpenLoop(
        title="Task", type="commitment", owner="ME", chat_id="C01", source_msg="M01",
        status="dropped", closed_by=None, help=None
    ).validate() is False

    # Dropped loop with non-null help must fail
    assert CandidateOpenLoop(
        title="Task", type="commitment", owner="ME", chat_id="C01", source_msg="M01",
        status="dropped", closed_by="M02", help=valid_help
    ).validate() is False


def test_candidate_open_loop_title_validation():
    """Verify CandidateOpenLoop validates and trims title."""
    valid_help = HelpAction(action="leave", reason="Not me")

    # Empty title
    assert CandidateOpenLoop(
        title="", type="commitment", owner="ME", chat_id="C01", source_msg="M01", help=valid_help
    ).validate() is False
    assert CandidateOpenLoop(
        title="   ", type="commitment", owner="ME", chat_id="C01", source_msg="M01", help=valid_help
    ).validate() is False

    # Title surrounding whitespace is trimmed in serialization
    loop = CandidateOpenLoop(
        title="  Submit budget proposal  ",
        type="commitment",
        owner="ME",
        chat_id="C01",
        source_msg="M01",
        help=valid_help,
    )
    assert loop.validate() is True
    assert loop.to_schema_dict()["title"] == "Submit budget proposal"


def test_candidate_open_loop_exact_nine_property_serialization():
    """Verify CandidateOpenLoop.to_schema_dict() emits exactly the 9 approved schema properties."""
    expected_keys = {
        "title",
        "type",
        "owner",
        "chat_id",
        "source_msg",
        "due",
        "status",
        "closed_by",
        "help",
    }

    # Open loop
    open_loop = CandidateOpenLoop(
        title="Plan team offsite",
        type="commitment",
        owner="ME",
        chat_id="C01",
        source_msg="M01",
        due="2025-06-01",
        status="open",
        help=HelpAction(action="assist", next_step="Research venues", needs_yes=False),
    )
    open_dict = open_loop.to_schema_dict()
    assert set(open_dict.keys()) == expected_keys
    assert len(open_dict) == 9

    # Done loop
    done_loop = CandidateOpenLoop(
        title="Send offsite dates",
        type="commitment",
        owner="ME",
        chat_id="C01",
        source_msg="M01",
        due=None,
        status="done",
        closed_by="M05",
    )
    done_dict = done_loop.to_schema_dict()
    assert set(done_dict.keys()) == expected_keys
    assert len(done_dict) == 9

    # Dropped loop
    dropped_loop = CandidateOpenLoop(
        title="Offsite booking",
        type="followup",
        owner="ME",
        chat_id="C01",
        source_msg="M01",
        due=None,
        status="dropped",
        closed_by="M06",
    )
    dropped_dict = dropped_loop.to_schema_dict()
    assert set(dropped_dict.keys()) == expected_keys
    assert len(dropped_dict) == 9


def test_candidate_open_loop_internal_attributes_excluded():
    """Verify internal provenance fields do not leak into schema output."""
    loop = CandidateOpenLoop(
        title="Prepare report",
        type="commitment",
        owner="ME",
        chat_id="C01",
        source_msg="M01",
        due="2025-04-10",
        due_anchor_msg="M02",
        status="open",
        help=HelpAction(action="leave", reason="Self managed"),
        confidence=0.95,
        source_msgs_all=["M01", "M02", "M03"],
    )
    assert loop.validate() is True
    schema = loop.to_schema_dict()
    assert "due_anchor_msg" not in schema
    assert "confidence" not in schema
    assert "source_msgs_all" not in schema
    assert len(schema) == 9


def test_candidate_open_loop_to_schema_dict_raises_on_invalid():
    """Verify CandidateOpenLoop.to_schema_dict() raises ValueError when invalid."""
    invalid_loops = [
        CandidateOpenLoop(
            title="", type="commitment", owner="ME", chat_id="C01", source_msg="M01"
        ),
        CandidateOpenLoop(
            title="Task", type="invalid_type", owner="ME", chat_id="C01", source_msg="M01"
        ),
        CandidateOpenLoop(
            title="Task", type="commitment", owner=" P01", chat_id="C01", source_msg="M01"
        ),
        CandidateOpenLoop(
            title="Task", type="commitment", owner="ME", chat_id="C01", source_msg="M01",
            status="done", closed_by=None
        ),
        CandidateOpenLoop(
            title="Task", type="commitment", owner="ME", chat_id="C01", source_msg="M01",
            status="open", closed_by=None, help=None
        ),
    ]
    for loop in invalid_loops:
        with pytest.raises(ValueError, match="Invalid CandidateOpenLoop"):
            loop.to_schema_dict()


# ============================================================================
# Config and Diagnostics Tests
# ============================================================================


def test_open_loop_extraction_config_defaults():
    """Verify OpenLoopExtractionConfig default parameter values."""
    config = OpenLoopExtractionConfig()
    assert config.min_confidence == 0.5
    assert config.conservative_closure is True
    assert config.default_remind_lead_hours == 24
    assert config.max_loops_per_chat is None
    assert config.max_open_loops is None  # Unlimited by default


def test_open_loop_extraction_config_custom_limits():
    """Verify OpenLoopExtractionConfig accepts custom and explicit limits."""
    # Explicit custom cap
    config_custom = OpenLoopExtractionConfig(max_open_loops=25, max_loops_per_chat=3)
    assert config_custom.max_open_loops == 25
    assert config_custom.max_loops_per_chat == 3

    # Explicit None (unlimited)
    config_unlimited = OpenLoopExtractionConfig(max_open_loops=None, max_loops_per_chat=None)
    assert config_unlimited.max_open_loops is None
    assert config_unlimited.max_loops_per_chat is None

    # Positive integer 1 accepted
    config_one = OpenLoopExtractionConfig(max_open_loops=1, max_loops_per_chat=1)
    assert config_one.max_open_loops == 1
    assert config_one.max_loops_per_chat == 1


@pytest.mark.parametrize("field_name", ["max_open_loops", "max_loops_per_chat"])
def test_open_loop_extraction_config_rejects_zero(field_name):
    """Verify OpenLoopExtractionConfig rejects zero with ValueError."""
    kwargs = {field_name: 0}
    with pytest.raises(ValueError, match="strictly positive integer"):
        OpenLoopExtractionConfig(**kwargs)


@pytest.mark.parametrize("field_name", ["max_open_loops", "max_loops_per_chat"])
@pytest.mark.parametrize("neg_val", [-1, -5, -100])
def test_open_loop_extraction_config_rejects_negative_integers(field_name, neg_val):
    """Verify OpenLoopExtractionConfig rejects negative integers with ValueError."""
    kwargs = {field_name: neg_val}
    with pytest.raises(ValueError, match="strictly positive integer"):
        OpenLoopExtractionConfig(**kwargs)


@pytest.mark.parametrize("field_name", ["max_open_loops", "max_loops_per_chat"])
@pytest.mark.parametrize("bool_val", [True, False])
def test_open_loop_extraction_config_rejects_booleans(field_name, bool_val):
    """Verify OpenLoopExtractionConfig explicitly rejects booleans with TypeError."""
    kwargs = {field_name: bool_val}
    with pytest.raises(TypeError, match="cannot be a boolean"):
        OpenLoopExtractionConfig(**kwargs)


@pytest.mark.parametrize("field_name", ["max_open_loops", "max_loops_per_chat"])
@pytest.mark.parametrize("float_val", [1.0, 10.5, 0.0, -2.5])
def test_open_loop_extraction_config_rejects_floats(field_name, float_val):
    """Verify OpenLoopExtractionConfig rejects float values with TypeError."""
    kwargs = {field_name: float_val}
    with pytest.raises(TypeError, match="must be an integer or None"):
        OpenLoopExtractionConfig(**kwargs)


@pytest.mark.parametrize("field_name", ["max_open_loops", "max_loops_per_chat"])
@pytest.mark.parametrize("str_val", ["10", "unlimited", "0", ""])
def test_open_loop_extraction_config_rejects_strings(field_name, str_val):
    """Verify OpenLoopExtractionConfig rejects string values with TypeError."""
    kwargs = {field_name: str_val}
    with pytest.raises(TypeError, match="must be an integer or None"):
        OpenLoopExtractionConfig(**kwargs)


@pytest.mark.parametrize("field_name", ["max_open_loops", "max_loops_per_chat"])
@pytest.mark.parametrize("bad_val", [[10], {"cap": 5}, (1, 2)])
def test_open_loop_extraction_config_rejects_other_unsupported_types(field_name, bad_val):
    """Verify OpenLoopExtractionConfig rejects unsupported types with TypeError."""
    kwargs = {field_name: bad_val}
    with pytest.raises(TypeError, match="must be an integer or None"):
        OpenLoopExtractionConfig(**kwargs)


def test_open_loop_extraction_diagnostics_defaults():
    """Verify OpenLoopExtractionDiagnostics default zero counters."""
    diag = OpenLoopExtractionDiagnostics()
    for field_name, value in diag.__dict__.items():
        assert isinstance(value, int), f"Field {field_name} must be int"
        assert value == 0, f"Field {field_name} must default to 0"

    # String representation must be clean without raw text leakage
    diag_str = str(diag)
    assert "OpenLoopExtractionDiagnostics(" in diag_str
    assert "candidates_received=0" in diag_str
    assert "final_loops_emitted=0" in diag_str

