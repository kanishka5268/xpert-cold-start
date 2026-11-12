"""Tests for the data ingestion and validation module."""

import json
from pathlib import Path
import pytest

from src.ingestion.loader import DataValidationError, load_dataset


def test_load_dev_dataset():
    """Verify loading and index integrity on the development dataset."""
    dev_dir = Path("data/dev")
    result = load_dataset(dev_dir)

    # 1. Counts and high level checks
    assert len(result["chats"]) == 138
    assert len(result["messages"]) == 4502
    assert len(result["chats_by_id"]) == 138
    assert len(result["messages_by_id"]) == 4502
    assert len(result["messages_by_chat"]) == 138

    summary = result["validation_summary"]
    assert summary["total_chats"] == 138
    assert summary["total_messages"] == 4502
    assert summary["direct_chats"] == 74
    assert summary["group_chats"] == 64
    assert summary["unique_participants"] == 287
    assert summary["unique_senders"] == 282
    assert summary["timestamp_ordering_status"] == "non_decreasing"
    assert summary["message_kinds"] == {
        "text": 3802,
        "image_caption": 606,
        "document": 94,
    }

    # 2. Seed structure preservation
    seed = result["seed"]
    assert "circles" in seed
    assert len(seed["circles"]) == 10
    assert seed["organisations"] == ["PostMyAd", "NEXUS"]
    assert seed["goals"] == []

    # 3. Silent participant registration
    participants = result["participants"]
    assert len(participants) == 287
    assert "ME" in participants
    assert "Arjun Khetan" in participants["ME"]["sender_names"]
    assert participants["ME"]["direct_chat_names"] == []

    # P319 is a silent participant in C31 (name: "Biren")
    assert "P319" in participants
    assert participants["P319"]["chat_ids"] == ["C31"]
    assert participants["P319"]["sender_names"] == []
    assert participants["P319"]["direct_chat_names"] == ["Biren"]

    # 4. Messages by chat ordering
    for cid, msgs in result["messages_by_chat"].items():
        assert len(msgs) > 0
        for i in range(len(msgs) - 1):
            assert msgs[i]["ts"] <= msgs[i + 1]["ts"]


def test_load_example_dataset():
    """Verify loading and index integrity on the example dataset."""
    example_dir = Path("example")
    result = load_dataset(example_dir)

    assert len(result["chats"]) == 6
    assert len(result["messages"]) == 29

    summary = result["validation_summary"]
    assert summary["total_chats"] == 6
    assert summary["total_messages"] == 29
    assert summary["direct_chats"] == 4
    assert summary["group_chats"] == 2
    assert summary["unique_participants"] == 9
    assert summary["unique_senders"] == 9
    assert summary["timestamp_ordering_status"] == "strictly_sorted"

    # Seed
    assert result["seed"]["circles"] == {"P01": "spouse"}
    assert result["seed"]["organisations"] == ["Brightside Design"]
    assert len(result["seed"]["goals"]) == 1
    assert result["seed"]["goals"][0]["title"] == "Brightside to 10 retainer clients by March"


def test_duplicate_chat_id(tmp_path):
    """Verify duplicate chat ID raises DataValidationError."""
    chats = [
        {"chat_id": "C01", "name": "Chat One", "is_group": False, "participants": ["ME", "P01"]},
        {"chat_id": "C01", "name": "Chat Duplicate", "is_group": False, "participants": ["ME", "P02"]},
    ]
    (tmp_path / "chats.jsonl").write_text("\n".join(json.dumps(c) for c in chats), encoding="utf-8")
    (tmp_path / "messages.jsonl").write_text("", encoding="utf-8")

    with pytest.raises(DataValidationError, match="Duplicate chat ID 'C01'"):
        load_dataset(tmp_path)


def test_duplicate_message_id(tmp_path):
    """Verify duplicate message ID raises DataValidationError."""
    chats = [{"chat_id": "C01", "name": "Chat One", "is_group": False, "participants": ["ME", "P01"]}]
    messages = [
        {"id": "M001", "chat_id": "C01", "sender_id": "P01", "sender_name": "P1", "ts": 100, "kind": "text", "body": "Hi", "quoted_id": None},
        {"id": "M001", "chat_id": "C01", "sender_id": "ME", "sender_name": "ME", "ts": 101, "kind": "text", "body": "Hi again", "quoted_id": None},
    ]
    (tmp_path / "chats.jsonl").write_text("\n".join(json.dumps(c) for c in chats), encoding="utf-8")
    (tmp_path / "messages.jsonl").write_text("\n".join(json.dumps(m) for m in messages), encoding="utf-8")

    with pytest.raises(DataValidationError, match="Duplicate message ID 'M001'"):
        load_dataset(tmp_path)


def test_missing_chat_reference(tmp_path):
    """Verify message referencing unknown chat raises DataValidationError."""
    chats = [{"chat_id": "C01", "name": "Chat One", "is_group": False, "participants": ["ME", "P01"]}]
    messages = [
        {"id": "M001", "chat_id": "C999", "sender_id": "P01", "sender_name": "P1", "ts": 100, "kind": "text", "body": "Hi", "quoted_id": None},
    ]
    (tmp_path / "chats.jsonl").write_text("\n".join(json.dumps(c) for c in chats), encoding="utf-8")
    (tmp_path / "messages.jsonl").write_text("\n".join(json.dumps(m) for m in messages), encoding="utf-8")

    with pytest.raises(DataValidationError, match="references unknown chat 'C999'"):
        load_dataset(tmp_path)


def test_invalid_sender_membership(tmp_path):
    """Verify sender not belonging to chat participants raises DataValidationError."""
    chats = [{"chat_id": "C01", "name": "Chat One", "is_group": False, "participants": ["ME", "P01"]}]
    messages = [
        {"id": "M001", "chat_id": "C01", "sender_id": "P02", "sender_name": "P2", "ts": 100, "kind": "text", "body": "Intruder", "quoted_id": None},
    ]
    (tmp_path / "chats.jsonl").write_text("\n".join(json.dumps(c) for c in chats), encoding="utf-8")
    (tmp_path / "messages.jsonl").write_text("\n".join(json.dumps(m) for m in messages), encoding="utf-8")

    with pytest.raises(DataValidationError, match="sender 'P02' who is not in participants of chat 'C01'"):
        load_dataset(tmp_path)


def test_missing_required_fields_in_chat(tmp_path):
    """Verify missing required chat fields and invalid types raise DataValidationError."""
    base_chat = {"chat_id": "C01", "name": "Chat One", "is_group": False, "participants": ["ME", "P01"]}

    # Missing chat_id
    bad_chat = dict(base_chat)
    del bad_chat["chat_id"]
    (tmp_path / "chats.jsonl").write_text(json.dumps(bad_chat), encoding="utf-8")
    (tmp_path / "messages.jsonl").write_text("", encoding="utf-8")
    with pytest.raises(DataValidationError, match="Missing required field 'chat_id'"):
        load_dataset(tmp_path)

    # is_group must be boolean
    bad_chat = dict(base_chat, is_group="false")
    (tmp_path / "chats.jsonl").write_text(json.dumps(bad_chat), encoding="utf-8")
    with pytest.raises(DataValidationError, match="must be a boolean"):
        load_dataset(tmp_path)

    # participants must be list
    bad_chat = dict(base_chat, participants="ME, P01")
    (tmp_path / "chats.jsonl").write_text(json.dumps(bad_chat), encoding="utf-8")
    with pytest.raises(DataValidationError, match="must be a list"):
        load_dataset(tmp_path)


def test_missing_required_fields_in_message(tmp_path):
    """Verify missing required message fields and invalid types raise DataValidationError."""
    chats = [{"chat_id": "C01", "name": "Chat One", "is_group": False, "participants": ["ME", "P01"]}]
    (tmp_path / "chats.jsonl").write_text("\n".join(json.dumps(c) for c in chats), encoding="utf-8")

    base_msg = {
        "id": "M001", "chat_id": "C01", "sender_id": "P01", "sender_name": "P1",
        "ts": 100, "kind": "text", "body": "Hi", "quoted_id": None
    }

    # Missing body
    bad_msg = dict(base_msg)
    del bad_msg["body"]
    (tmp_path / "messages.jsonl").write_text(json.dumps(bad_msg), encoding="utf-8")
    with pytest.raises(DataValidationError, match="Missing required field 'body'"):
        load_dataset(tmp_path)

    # ts must be int, not string
    bad_msg = dict(base_msg, ts="100")
    (tmp_path / "messages.jsonl").write_text(json.dumps(bad_msg), encoding="utf-8")
    with pytest.raises(DataValidationError, match="Field 'ts' .* must be an integer"):
        load_dataset(tmp_path)

    # ts must not be bool (in Python bool is a subclass of int)
    bad_msg = dict(base_msg, ts=True)
    (tmp_path / "messages.jsonl").write_text(json.dumps(bad_msg), encoding="utf-8")
    with pytest.raises(DataValidationError, match="Field 'ts' .* must be an integer"):
        load_dataset(tmp_path)


def test_unicode_and_multiline_body_preservation(tmp_path):
    """Verify that multiline messages and diverse unicode scripts are preserved verbatim."""
    chats = [{"chat_id": "C01", "name": "Family 🏠", "is_group": False, "participants": ["ME", "P01"]}]
    unicode_body = (
        "Line 1: Namaste 🙏\n"
        "Line 2: यहाँ वन्दे मातरम् हिंदी में है।\n"
        "Line 3: Kaomoji (⁠˃͈⁠ ⁠દ⁠ ⁠˂͈⁠ ⁠༶⁠ ⁠) and special ~symbols~."
    )
    messages = [
        {"id": "M001", "chat_id": "C01", "sender_id": "P01", "sender_name": "Karan ❤️", "ts": 100, "kind": "text", "body": unicode_body, "quoted_id": None}
    ]
    (tmp_path / "chats.jsonl").write_text("\n".join(json.dumps(c) for c in chats), encoding="utf-8")
    (tmp_path / "messages.jsonl").write_text("\n".join(json.dumps(m) for m in messages), encoding="utf-8")

    result = load_dataset(tmp_path)
    loaded_msg = result["messages_by_id"]["M001"]
    assert loaded_msg["body"] == unicode_body


def test_silent_participant_registration(tmp_path):
    """Verify participants who never sent messages are properly indexed."""
    chats = [
        {"chat_id": "C01", "name": "Direct Active", "is_group": False, "participants": ["ME", "P01"]},
        {"chat_id": "C02", "name": "Direct Silent Contact", "is_group": False, "participants": ["ME", "P_SILENT"]},
        {"chat_id": "C03", "name": "Group Chat", "is_group": True, "participants": ["ME", "P01", "P_GROUP_SILENT"]},
    ]
    # Messages only from P01
    messages = [
        {"id": "M001", "chat_id": "C01", "sender_id": "P01", "sender_name": "P1 Profile", "ts": 100, "kind": "text", "body": "Hi", "quoted_id": None}
    ]
    (tmp_path / "chats.jsonl").write_text("\n".join(json.dumps(c) for c in chats), encoding="utf-8")
    (tmp_path / "messages.jsonl").write_text("\n".join(json.dumps(m) for m in messages), encoding="utf-8")

    result = load_dataset(tmp_path)
    participants = result["participants"]

    assert "P_SILENT" in participants
    assert participants["P_SILENT"]["chat_ids"] == ["C02"]
    assert participants["P_SILENT"]["sender_names"] == []
    assert participants["P_SILENT"]["direct_chat_names"] == ["Direct Silent Contact"]

    assert "P_GROUP_SILENT" in participants
    assert participants["P_GROUP_SILENT"]["chat_ids"] == ["C03"]
    assert participants["P_GROUP_SILENT"]["sender_names"] == []
    assert participants["P_GROUP_SILENT"]["direct_chat_names"] == []


def test_distinction_two_person_group_vs_direct_chat(tmp_path):
    """Verify is_group is the authoritative discriminator, not participant count."""
    chats = [
        # 2 participants, is_group=True (Formal group)
        {"chat_id": "C_GRP", "name": "Project Alpha Group", "is_group": True, "participants": ["ME", "P01"]},
        # 2 participants, is_group=False (Direct chat)
        {"chat_id": "C_DIR", "name": "P01 Personal", "is_group": False, "participants": ["ME", "P01"]},
    ]
    messages = [
        {"id": "M001", "chat_id": "C_GRP", "sender_id": "P01", "sender_name": "P1", "ts": 100, "kind": "text", "body": "Group msg", "quoted_id": None},
        {"id": "M002", "chat_id": "C_DIR", "sender_id": "P01", "sender_name": "P1", "ts": 101, "kind": "text", "body": "Direct msg", "quoted_id": None},
    ]
    (tmp_path / "chats.jsonl").write_text("\n".join(json.dumps(c) for c in chats), encoding="utf-8")
    (tmp_path / "messages.jsonl").write_text("\n".join(json.dumps(m) for m in messages), encoding="utf-8")

    result = load_dataset(tmp_path)
    summary = result["validation_summary"]

    assert summary["group_chats"] == 1
    assert summary["direct_chats"] == 1

    # direct_chat_names must only capture the direct chat title, NOT the group title
    p01 = result["participants"]["P01"]
    assert p01["direct_chat_names"] == ["P01 Personal"]
    assert "Project Alpha Group" not in p01["direct_chat_names"]


def test_malformed_json_and_blank_lines(tmp_path):
    """Verify clear error with line number for malformed JSON and blank lines skipped."""
    chats_content = (
        '{"chat_id": "C01", "name": "Valid 1", "is_group": false, "participants": ["ME", "P01"]}\n'
        '\n'  # blank line
        '   \n'  # whitespace line
        '{"chat_id": "C02", "name": "Malformed", \n'  # broken json
    )
    (tmp_path / "chats.jsonl").write_text(chats_content, encoding="utf-8")
    (tmp_path / "messages.jsonl").write_text("", encoding="utf-8")

    with pytest.raises(DataValidationError, match="Malformed JSON in 'chats.jsonl' at line 4"):
        load_dataset(tmp_path)


def test_quoted_message_references(tmp_path):
    """Verify quoted_id validation for both valid and broken quotes."""
    chats = [{"chat_id": "C01", "name": "Chat", "is_group": False, "participants": ["ME", "P01"]}]
    messages = [
        {"id": "M001", "chat_id": "C01", "sender_id": "P01", "sender_name": "P1", "ts": 100, "kind": "text", "body": "First", "quoted_id": None},
        {"id": "M002", "chat_id": "C01", "sender_id": "ME", "sender_name": "ME", "ts": 105, "kind": "text", "body": "Reply", "quoted_id": "M001"},
        {"id": "M003", "chat_id": "C01", "sender_id": "P01", "sender_name": "P1", "ts": 110, "kind": "text", "body": "Broken reply", "quoted_id": "M_UNKNOWN"},
    ]
    (tmp_path / "chats.jsonl").write_text("\n".join(json.dumps(c) for c in chats), encoding="utf-8")
    (tmp_path / "messages.jsonl").write_text("\n".join(json.dumps(m) for m in messages), encoding="utf-8")

    # strict_quotes=False (default): registers warning
    result = load_dataset(tmp_path, strict_quotes=False)
    warnings = result["validation_summary"]["validation_warnings"]
    assert any("quotes non-existent message 'M_UNKNOWN'" in w for w in warnings)

    # strict_quotes=True: raises DataValidationError
    with pytest.raises(DataValidationError, match="quotes non-existent message 'M_UNKNOWN'"):
        load_dataset(tmp_path, strict_quotes=True)


def test_duplicate_participant_ids(tmp_path):
    """Verify duplicate participant IDs within a chat raise DataValidationError."""
    chats = [
        {"chat_id": "C01", "name": "Chat", "is_group": False, "participants": ["ME", "P01", "P01"]}
    ]
    (tmp_path / "chats.jsonl").write_text("\n".join(json.dumps(c) for c in chats), encoding="utf-8")
    (tmp_path / "messages.jsonl").write_text("", encoding="utf-8")

    with pytest.raises(DataValidationError, match="Duplicate participant ID 'P01' in chat 'C01'"):
        load_dataset(tmp_path)


def test_zero_message_chats_in_messages_by_chat(tmp_path):
    """Verify chats with zero messages have an empty list in messages_by_chat."""
    chats = [
        {"chat_id": "C01", "name": "Active Chat", "is_group": False, "participants": ["ME", "P01"]},
        {"chat_id": "C02", "name": "Quiet Chat", "is_group": False, "participants": ["ME", "P02"]},
    ]
    messages = [
        {"id": "M001", "chat_id": "C01", "sender_id": "P01", "sender_name": "P1", "ts": 100, "kind": "text", "body": "Hi", "quoted_id": None}
    ]
    (tmp_path / "chats.jsonl").write_text("\n".join(json.dumps(c) for c in chats), encoding="utf-8")
    (tmp_path / "messages.jsonl").write_text("\n".join(json.dumps(m) for m in messages), encoding="utf-8")

    result = load_dataset(tmp_path)
    by_chat = result["messages_by_chat"]

    assert "C01" in by_chat and len(by_chat["C01"]) == 1
    assert "C02" in by_chat and by_chat["C02"] == []


def test_quoted_cross_chat_references(tmp_path):
    """Verify cross-chat quoted message references warn in non-strict and fail in strict mode."""
    chats = [
        {"chat_id": "C01", "name": "Chat 1", "is_group": False, "participants": ["ME", "P01"]},
        {"chat_id": "C02", "name": "Chat 2", "is_group": False, "participants": ["ME", "P02"]},
    ]
    messages = [
        {"id": "M001", "chat_id": "C01", "sender_id": "P01", "sender_name": "P1", "ts": 100, "kind": "text", "body": "In C1", "quoted_id": None},
        {"id": "M002", "chat_id": "C02", "sender_id": "P02", "sender_name": "P2", "ts": 105, "kind": "text", "body": "In C2 quoting C1", "quoted_id": "M001"},
    ]
    (tmp_path / "chats.jsonl").write_text("\n".join(json.dumps(c) for c in chats), encoding="utf-8")
    (tmp_path / "messages.jsonl").write_text("\n".join(json.dumps(m) for m in messages), encoding="utf-8")

    # Non-strict mode: warns
    result = load_dataset(tmp_path, strict_quotes=False)
    warnings = result["validation_summary"]["validation_warnings"]
    assert any("quotes message 'M001' from chat 'C01'" in w for w in warnings)

    # Strict mode: raises DataValidationError
    with pytest.raises(DataValidationError, match="quotes message 'M001' from chat 'C01'"):
        load_dataset(tmp_path, strict_quotes=True)


def test_quoted_future_message_references(tmp_path):
    """Verify future-timestamp quoted references warn in non-strict and fail in strict mode."""
    chats = [
        {"chat_id": "C01", "name": "Chat 1", "is_group": False, "participants": ["ME", "P01"]},
    ]
    messages = [
        {"id": "M001", "chat_id": "C01", "sender_id": "P01", "sender_name": "P1", "ts": 100, "kind": "text", "body": "Quoting future", "quoted_id": "M002"},
        {"id": "M002", "chat_id": "C01", "sender_id": "ME", "sender_name": "ME", "ts": 200, "kind": "text", "body": "Future message", "quoted_id": None},
    ]
    (tmp_path / "chats.jsonl").write_text("\n".join(json.dumps(c) for c in chats), encoding="utf-8")
    (tmp_path / "messages.jsonl").write_text("\n".join(json.dumps(m) for m in messages), encoding="utf-8")

    # Non-strict mode: warns
    result = load_dataset(tmp_path, strict_quotes=False)
    warnings = result["validation_summary"]["validation_warnings"]
    assert any("quotes future message 'M002'" in w for w in warnings)

    # Strict mode: raises DataValidationError
    with pytest.raises(DataValidationError, match="quotes future message 'M002'"):
        load_dataset(tmp_path, strict_quotes=True)


def test_quoted_self_references(tmp_path):
    """Verify self-referencing quoted message warns in non-strict and fails in strict mode."""
    chats = [
        {"chat_id": "C01", "name": "Chat 1", "is_group": False, "participants": ["ME", "P01"]},
    ]
    messages = [
        {"id": "M001", "chat_id": "C01", "sender_id": "P01", "sender_name": "P1", "ts": 100, "kind": "text", "body": "Self quote", "quoted_id": "M001"},
    ]
    (tmp_path / "chats.jsonl").write_text("\n".join(json.dumps(c) for c in chats), encoding="utf-8")
    (tmp_path / "messages.jsonl").write_text("\n".join(json.dumps(m) for m in messages), encoding="utf-8")

    # Non-strict mode: warns
    result = load_dataset(tmp_path, strict_quotes=False)
    warnings = result["validation_summary"]["validation_warnings"]
    assert any("quotes itself" in w for w in warnings)

    # Strict mode: raises DataValidationError
    with pytest.raises(DataValidationError, match="quotes itself"):
        load_dataset(tmp_path, strict_quotes=True)


def test_forward_compatibility_unknown_message_kind(tmp_path):
    """Verify unknown message kinds are accepted for forward compatibility."""
    chats = [{"chat_id": "C01", "name": "Chat", "is_group": False, "participants": ["ME", "P01"]}]
    messages = [
        {"id": "M001", "chat_id": "C01", "sender_id": "P01", "sender_name": "P1", "ts": 100, "kind": "voice_note", "body": "[voice]", "quoted_id": None},
        {"id": "M002", "chat_id": "C01", "sender_id": "ME", "sender_name": "ME", "ts": 105, "kind": "poll", "body": "[poll: Dinner?]", "quoted_id": None},
    ]
    (tmp_path / "chats.jsonl").write_text("\n".join(json.dumps(c) for c in chats), encoding="utf-8")
    (tmp_path / "messages.jsonl").write_text("\n".join(json.dumps(m) for m in messages), encoding="utf-8")

    result = load_dataset(tmp_path)
    assert result["validation_summary"]["message_kinds"] == {"voice_note": 1, "poll": 1}
    assert result["messages_by_id"]["M001"]["kind"] == "voice_note"
