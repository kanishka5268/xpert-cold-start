"""Unit and regression tests for ConversationContext and context builder."""

from pathlib import Path
import pytest

from src.ingestion.loader import load_dataset
from src.preprocessing.context_builder import (
    ConversationContext,
    build_chat_histories,
    build_conversation_context,
    build_participant_histories,
    get_message_window,
)


@pytest.fixture(scope="module")
def dev_dataset():
    """Load the development dataset once for read-only tests."""
    return load_dataset(Path("data/dev"))


@pytest.fixture(scope="module")
def example_dataset():
    """Load the small example dataset once for read-only tests."""
    return load_dataset(Path("example"))


def test_build_context_dev_dataset(dev_dataset):
    """Verify context builder against the full development dataset."""
    context = build_conversation_context(dev_dataset)

    summary = context.summary
    assert summary["total_chats"] == 138
    assert summary["direct_chats"] == 74
    assert summary["group_chats"] == 64
    assert summary["total_participants"] == 287
    assert summary["silent_participants"] == 5
    assert summary["total_messages"] == 4502

    # Container access
    assert "chats" in context
    assert "participants" in context
    assert len(context["chats"]) == 138
    assert len(context["participants"]) == 287


def test_chronological_ordering_chats(dev_dataset):
    """Verify that every chat history maintains chronological order."""
    context = build_conversation_context(dev_dataset)

    for cid, chat in context.chats.items():
        msgs = chat["messages"]
        assert chat["message_count"] == len(msgs)

        if msgs:
            assert chat["first_timestamp"] == msgs[0]["ts"]
            assert chat["last_timestamp"] == msgs[-1]["ts"]
            for i in range(len(msgs) - 1):
                assert msgs[i]["ts"] <= msgs[i + 1]["ts"]
        else:
            assert chat["first_timestamp"] is None
            assert chat["last_timestamp"] is None


def test_chronological_ordering_participants(dev_dataset):
    """Verify that participant-centric histories are sorted chronologically."""
    context = build_conversation_context(dev_dataset)

    for pid, part in context.participants.items():
        sent = part["messages_sent"]
        direct = part["direct_messages"]
        all_msgs = part["all_messages"]

        assert part["total_messages_sent"] == len(sent)

        for msgs in (sent, direct, all_msgs):
            for i in range(len(msgs) - 1):
                assert msgs[i]["ts"] <= msgs[i + 1]["ts"]

        if sent:
            assert part["first_sent_timestamp"] == sent[0]["ts"]
            assert part["last_sent_timestamp"] == sent[-1]["ts"]
        else:
            assert part["first_sent_timestamp"] is None
            assert part["last_sent_timestamp"] is None


def test_direct_and_group_distinction(dev_dataset):
    """Verify is_group is the authoritative discriminator across chats and participant indexes."""
    context = build_conversation_context(dev_dataset)

    # In dev dataset, C08 is a 2-person group chat ('ME', 'P11')
    c08 = context.get_chat("C08")
    assert c08 is not None
    assert c08["is_group"] is True
    assert c08["participant_count"] == 2

    # Check participant P11 categorization
    p11 = context.get_participant("P11")
    assert p11 is not None
    assert "C08" in p11["group_chat_ids"]
    assert "C08" not in p11["direct_chat_ids"]

    # Direct messages must come strictly from chats where is_group == False
    for pid, part in context.participants.items():
        for m in part["direct_messages"]:
            chat = context.get_chat(m["chat_id"])
            assert chat is not None
            assert chat["is_group"] is False


def test_silent_participants(dev_dataset):
    """Verify participants with zero sent messages are properly indexed with intact context."""
    context = build_conversation_context(dev_dataset)

    # P319 is a silent participant in C31
    p319 = context.get_participant("P319")
    assert p319 is not None
    assert p319["is_silent"] is True
    assert p319["total_messages_sent"] == 0
    assert p319["messages_sent"] == []
    assert p319["first_sent_timestamp"] is None
    assert p319["last_sent_timestamp"] is None

    # Preserves chat and name context
    assert p319["chat_ids"] == ["C31"]
    assert p319["direct_chat_names"] == ["Biren"]

    # In C31, ME sent messages, so all_messages and direct_messages for P319 must reflect that
    assert len(p319["all_messages"]) > 0
    assert len(p319["direct_messages"]) > 0

    # Querying sent messages window returns empty list without error
    assert context.get_participant_messages("P319", mode="sent", limit=5) == []


def test_empty_chat_support():
    """Verify support for chats with zero messages."""
    dataset = {
        "chats": [
            {"chat_id": "C01", "name": "Active", "is_group": False, "participants": ["ME", "P01"]},
            {"chat_id": "C02", "name": "Zero Messages", "is_group": False, "participants": ["ME", "P02"]},
        ],
        "messages": [
            {"id": "M001", "chat_id": "C01", "sender_id": "P01", "sender_name": "P1", "ts": 100, "kind": "text", "body": "Hi", "quoted_id": None}
        ],
        "chats_by_id": {
            "C01": {"chat_id": "C01", "name": "Active", "is_group": False, "participants": ["ME", "P01"]},
            "C02": {"chat_id": "C02", "name": "Zero Messages", "is_group": False, "participants": ["ME", "P02"]},
        },
        "messages_by_id": {
            "M001": {"id": "M001", "chat_id": "C01", "sender_id": "P01", "sender_name": "P1", "ts": 100, "kind": "text", "body": "Hi", "quoted_id": None}
        },
        "messages_by_chat": {
            "C01": [{"id": "M001", "chat_id": "C01", "sender_id": "P01", "sender_name": "P1", "ts": 100, "kind": "text", "body": "Hi", "quoted_id": None}],
            "C02": [],
        },
        "participants": {
            "ME": {"participant_id": "ME", "chat_ids": ["C01", "C02"], "sender_names": [], "direct_chat_names": []},
            "P01": {"participant_id": "P01", "chat_ids": ["C01"], "sender_names": ["P1"], "direct_chat_names": ["Active"]},
            "P02": {"participant_id": "P02", "chat_ids": ["C02"], "sender_names": [], "direct_chat_names": ["Zero Messages"]},
        },
    }

    context = build_conversation_context(dataset)

    c02 = context.get_chat("C02")
    assert c02 is not None
    assert c02["message_count"] == 0
    assert c02["messages"] == []
    assert c02["first_timestamp"] is None
    assert c02["last_timestamp"] is None

    # Window query returns empty list
    assert context.get_chat_messages("C02", limit=10) == []

    # P02 has sent 0 messages and received 0 messages
    p02 = context.get_participant("P02")
    assert p02 is not None
    assert p02["is_silent"] is True
    assert p02["messages_sent"] == []
    assert p02["all_messages"] == []
    assert p02["direct_messages"] == []


def test_message_window_limits_and_filtering():
    """Verify get_message_window limits, range filters, kinds filtering, and reverse ordering."""
    messages = [
        {"id": "M01", "ts": 100, "kind": "text", "body": "1"},
        {"id": "M02", "ts": 110, "kind": "text", "body": "2"},
        {"id": "M03", "ts": 120, "kind": "image_caption", "body": "[photo]"},
        {"id": "M04", "ts": 130, "kind": "text", "body": "4"},
        {"id": "M05", "ts": 140, "kind": "document", "body": "doc.pdf"},
        {"id": "M06", "ts": 150, "kind": "text", "body": "6"},
    ]

    # 1. No filter returns all
    assert get_message_window(messages) == messages

    # 2. Limit takes the most recent N messages in chronological order
    window = get_message_window(messages, limit=3)
    assert len(window) == 3
    assert [m["id"] for m in window] == ["M04", "M05", "M06"]

    # 3. Limit larger than length returns all
    assert len(get_message_window(messages, limit=100)) == 6

    # 4. Limit <= 0 returns empty list
    assert get_message_window(messages, limit=0) == []
    assert get_message_window(messages, limit=-5) == []

    # 5. Timestamp range filtering
    ranged = get_message_window(messages, since_ts=110, until_ts=140)
    assert [m["id"] for m in ranged] == ["M02", "M03", "M04", "M05"]

    # 6. Kind filtering
    texts = get_message_window(messages, kinds={"text"})
    assert len(texts) == 4
    assert all(m["kind"] == "text" for m in texts)

    # 7. Combined filter and limit
    combined = get_message_window(messages, kinds={"text"}, limit=2)
    assert [m["id"] for m in combined] == ["M04", "M06"]

    # 8. Reverse ordering
    rev = get_message_window(messages, limit=3, reverse=True)
    assert [m["id"] for m in rev] == ["M06", "M05", "M04"]


def test_message_window_missing_ts_with_since_ts():
    """Verify that a message missing a ts key is excluded when since_ts is supplied."""
    msg_without_ts = {"id": "M_NO_TS", "kind": "text", "body": "no ts"}
    msg_with_ts = {"id": "M_VALID", "ts": 150, "kind": "text", "body": "valid"}
    messages = [msg_without_ts, msg_with_ts]

    # Without timestamp filter: both are preserved
    assert get_message_window(messages) == messages

    # With since_ts filter: missing ts message must be excluded
    result = get_message_window(messages, since_ts=100)
    assert len(result) == 1
    assert result[0]["id"] == "M_VALID"


def test_message_window_none_ts_with_until_ts():
    """Verify that a message with ts=None is excluded when until_ts is supplied."""
    msg_none_ts = {"id": "M_NONE_TS", "ts": None, "kind": "text", "body": "none ts"}
    msg_with_ts = {"id": "M_VALID", "ts": 50, "kind": "text", "body": "valid"}
    messages = [msg_none_ts, msg_with_ts]

    # Without timestamp filter: both are preserved
    assert get_message_window(messages) == messages

    # With until_ts filter: ts=None message must be excluded
    result = get_message_window(messages, until_ts=100)
    assert len(result) == 1
    assert result[0]["id"] == "M_VALID"


def test_message_window_invalid_ts_with_range():
    """Verify that messages with non-integer ts (string, bool, float) are excluded under range filter."""
    messages = [
        {"id": "M_STR", "ts": "120", "body": "string ts"},
        {"id": "M_BOOL", "ts": True, "body": "bool ts"},
        {"id": "M_FLOAT", "ts": 120.5, "body": "float ts"},
        {"id": "M_VALID", "ts": 120, "body": "int ts"},
    ]

    # Without timestamp filter: all are preserved
    assert get_message_window(messages) == messages

    # With range filter: only valid integer ts message passes
    result = get_message_window(messages, since_ts=100, until_ts=150)
    assert len(result) == 1
    assert result[0]["id"] == "M_VALID"


def test_message_window_inclusive_boundaries():
    """Verify that boundary timestamps equal to since_ts or until_ts are included."""
    messages = [
        {"id": "M_BEFORE", "ts": 99},
        {"id": "M_LOWER", "ts": 100},
        {"id": "M_MID", "ts": 150},
        {"id": "M_UPPER", "ts": 200},
        {"id": "M_AFTER", "ts": 201},
    ]

    # Range covering boundaries: 100 and 200 must be included
    ranged = get_message_window(messages, since_ts=100, until_ts=200)
    assert [m["id"] for m in ranged] == ["M_LOWER", "M_MID", "M_UPPER"]

    # Lower boundary only
    since_only = get_message_window(messages, since_ts=150)
    assert [m["id"] for m in since_only] == ["M_MID", "M_UPPER", "M_AFTER"]

    # Upper boundary only
    until_only = get_message_window(messages, until_ts=150)
    assert [m["id"] for m in until_only] == ["M_BEFORE", "M_LOWER", "M_MID"]


def test_participant_modes_and_queries(example_dataset):
    """Verify participant message queries across all, direct, and sent modes."""
    context = build_conversation_context(example_dataset)

    # In example dataset, Karan (P01) is in C01 (direct) and C03 (group Ghar)
    p01 = context.get_participant("P01")
    assert p01 is not None

    all_msgs = context.get_participant_messages("P01", mode="all")
    direct_msgs = context.get_participant_messages("P01", mode="direct")
    sent_msgs = context.get_participant_messages("P01", mode="sent")

    assert len(all_msgs) > len(direct_msgs)
    assert len(all_msgs) > len(sent_msgs)
    assert all(m["sender_id"] == "P01" for m in sent_msgs)
    assert all(m["chat_id"] == "C01" for m in direct_msgs)

    # Recent window on participant messages
    recent_direct = context.get_participant_messages("P01", mode="direct", limit=2)
    assert len(recent_direct) == 2
    assert recent_direct == direct_msgs[-2:]

    # Invalid mode raises ValueError
    with pytest.raises(ValueError, match="Invalid mode 'unknown'"):
        context.get_participant_messages("P01", mode="unknown")

    # Unknown participant returns empty list
    assert context.get_participant_messages("P_DOES_NOT_EXIST") == []


def test_preservation_of_all_message_fields(dev_dataset):
    """Verify message IDs, chat IDs, sender IDs, sender names, timestamps, kinds, bodies, and quoted_id are intact."""
    context = build_conversation_context(dev_dataset)

    sample_msg = context.chats["C01"]["messages"][0]
    expected_fields = {"id", "chat_id", "sender_id", "sender_name", "ts", "kind", "body", "quoted_id"}
    for f in expected_fields:
        assert f in sample_msg

    assert sample_msg["id"] == "M00001"
    assert sample_msg["chat_id"] == "C01"
    assert sample_msg["sender_id"] == "P01"
    assert sample_msg["sender_name"] == "Nirant Vakil"
    assert sample_msg["ts"] == 1787535062
    assert sample_msg["kind"] == "text"
    assert "Wow! No listed bank has even 1% net NPA" in sample_msg["body"]
    assert sample_msg["quoted_id"] is None


def test_context_serializability(example_dataset):
    """Verify to_dict returns valid dictionary without crashing."""
    context = build_conversation_context(example_dataset)
    d = context.to_dict()

    assert "chats" in d
    assert "participants" in d
    assert "summary" in d
    assert d["summary"]["total_chats"] == 6

