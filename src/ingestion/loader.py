"""Data ingestion and validation module for WhatsApp conversations."""

from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union


class DataValidationError(ValueError):
    """Raised when data loading, schema, or integrity validation fails."""
    pass


def _read_jsonl(file_path: Path) -> List[Dict[str, Any]]:
    """Read a JSONL file, ignoring empty lines, with line-numbered error reporting."""
    if not file_path.exists():
        raise FileNotFoundError(f"Required file not found: {file_path}")

    records: List[Dict[str, Any]] = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as err:
                raise DataValidationError(
                    f"Malformed JSON in '{file_path.name}' at line {line_no}: {err}"
                ) from err

            if not isinstance(item, dict):
                raise DataValidationError(
                    f"Expected JSON object in '{file_path.name}' at line {line_no}, got {type(item).__name__}"
                )
            records.append(item)
    return records


def _read_json(file_path: Path) -> Dict[str, Any]:
    """Read a JSON file with error reporting."""
    if not file_path.exists():
        return {}
    with open(file_path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as err:
            raise DataValidationError(
                f"Malformed JSON in '{file_path.name}': {err}"
            ) from err

    if not isinstance(data, dict):
        raise DataValidationError(
            f"Expected JSON object in '{file_path.name}', got {type(data).__name__}"
        )
    return data


def _validate_chat(chat: Dict[str, Any], line_no: int, file_name: str) -> None:
    """Validate chat schema and fields."""
    required = ("chat_id", "name", "is_group", "participants")
    for field in required:
        if field not in chat:
            raise DataValidationError(
                f"Missing required field '{field}' in '{file_name}' at line {line_no}"
            )

    chat_id = chat["chat_id"]
    if not isinstance(chat_id, str) or not chat_id.strip():
        raise DataValidationError(
            f"Field 'chat_id' in '{file_name}' at line {line_no} must be a non-empty string"
        )

    if not isinstance(chat["name"], str):
        raise DataValidationError(
            f"Field 'name' in '{file_name}' at line {line_no} must be a string"
        )

    if not isinstance(chat["is_group"], bool):
        raise DataValidationError(
            f"Field 'is_group' in '{file_name}' at line {line_no} must be a boolean"
        )

    participants = chat["participants"]
    if not isinstance(participants, list):
        raise DataValidationError(
            f"Field 'participants' in '{file_name}' at line {line_no} must be a list"
        )
    if not participants:
        raise DataValidationError(
            f"Field 'participants' in '{file_name}' at line {line_no} cannot be empty"
        )
    seen_participants: Set[str] = set()
    for p in participants:
        if not isinstance(p, str) or not p.strip():
            raise DataValidationError(
                f"Participant IDs in '{file_name}' at line {line_no} must be non-empty strings"
            )
        if p in seen_participants:
            raise DataValidationError(
                f"Duplicate participant ID '{p}' in chat '{chat_id}' in '{file_name}' at line {line_no}"
            )
        seen_participants.add(p)


def _validate_message(msg: Dict[str, Any], line_no: int, file_name: str) -> None:
    """Validate message schema and fields."""
    required = ("id", "chat_id", "sender_id", "sender_name", "ts", "kind", "body")
    for field in required:
        if field not in msg:
            raise DataValidationError(
                f"Missing required field '{field}' in '{file_name}' at line {line_no}"
            )

    mid = msg["id"]
    if not isinstance(mid, str) or not mid.strip():
        raise DataValidationError(
            f"Field 'id' in '{file_name}' at line {line_no} must be a non-empty string"
        )

    cid = msg["chat_id"]
    if not isinstance(cid, str) or not cid.strip():
        raise DataValidationError(
            f"Field 'chat_id' in '{file_name}' at line {line_no} must be a non-empty string"
        )

    sid = msg["sender_id"]
    if not isinstance(sid, str) or not sid.strip():
        raise DataValidationError(
            f"Field 'sender_id' in '{file_name}' at line {line_no} must be a non-empty string"
        )

    if not isinstance(msg["sender_name"], str):
        raise DataValidationError(
            f"Field 'sender_name' in '{file_name}' at line {line_no} must be a string"
        )

    ts = msg["ts"]
    if isinstance(ts, bool) or not isinstance(ts, int):
        raise DataValidationError(
            f"Field 'ts' in '{file_name}' at line {line_no} must be an integer"
        )

    kind = msg["kind"]
    if not isinstance(kind, str) or not kind.strip():
        raise DataValidationError(
            f"Field 'kind' in '{file_name}' at line {line_no} must be a non-empty string"
        )

    if not isinstance(msg["body"], str):
        raise DataValidationError(
            f"Field 'body' in '{file_name}' at line {line_no} must be a string"
        )

    quoted_id = msg.get("quoted_id")
    if quoted_id is not None:
        if not isinstance(quoted_id, str) or not quoted_id.strip():
            raise DataValidationError(
                f"Field 'quoted_id' in '{file_name}' at line {line_no} must be null or a non-empty string"
            )


def load_dataset(
    input_dir: Union[str, Path],
    strict_quotes: bool = False,
) -> Dict[str, Any]:
    """
    Load and validate a WhatsApp conversation dataset.

    Args:
        input_dir: Path to directory containing chats.jsonl, messages.jsonl, seed.json.
        strict_quotes: If True, broken quoted_id references raise DataValidationError.
                       If False, broken references are appended to validation_warnings.

    Returns:
        Structured dictionary with chats, messages, seed, indexes, and validation_summary.
    """
    input_path = Path(input_dir)
    if not input_path.exists() or not input_path.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {input_path}")

    chats_path = input_path / "chats.jsonl"
    messages_path = input_path / "messages.jsonl"
    seed_path = input_path / "seed.json"

    # 1. Load raw records
    chats = _read_jsonl(chats_path)
    messages = _read_jsonl(messages_path)
    seed = _read_json(seed_path)

    validation_warnings: List[str] = []
    if not seed_path.exists():
        validation_warnings.append(f"seed.json not found in '{input_path}'")

    # 2. Validate chats and build chats_by_id
    chats_by_id: Dict[str, Dict[str, Any]] = {}
    direct_chat_count = 0
    group_chat_count = 0

    for idx, chat in enumerate(chats, start=1):
        _validate_chat(chat, idx, "chats.jsonl")
        cid = chat["chat_id"]
        if cid in chats_by_id:
            raise DataValidationError(
                f"Duplicate chat ID '{cid}' found in 'chats.jsonl' at line {idx}"
            )
        chats_by_id[cid] = chat

        # Use is_group as the authoritative discriminator
        if chat["is_group"]:
            group_chat_count += 1
        else:
            direct_chat_count += 1

    # 3. Validate messages and build messages_by_id and messages_by_chat
    messages_by_id: Dict[str, Dict[str, Any]] = {}
    # Initialize messages_by_chat with an empty list for every chat to ensure zero-message chats are represented
    messages_by_chat: Dict[str, List[Dict[str, Any]]] = {
        cid: [] for cid in chats_by_id
    }
    unique_senders: Set[str] = set()
    message_kinds: Counter[str] = Counter()

    for idx, msg in enumerate(messages, start=1):
        _validate_message(msg, idx, "messages.jsonl")
        mid = msg["id"]
        if mid in messages_by_id:
            raise DataValidationError(
                f"Duplicate message ID '{mid}' found in 'messages.jsonl' at line {idx}"
            )

        cid = msg["chat_id"]
        if cid not in chats_by_id:
            raise DataValidationError(
                f"Message '{mid}' at line {idx} references unknown chat '{cid}'"
            )

        sid = msg["sender_id"]
        chat = chats_by_id[cid]
        if sid not in chat["participants"]:
            raise DataValidationError(
                f"Message '{mid}' at line {idx} has sender '{sid}' who is not in participants of chat '{cid}'"
            )

        messages_by_id[mid] = msg
        messages_by_chat[cid].append(msg)
        unique_senders.add(sid)
        message_kinds[msg["kind"]] += 1

    # 4. Validate quoted_id references
    for msg in messages:
        qid = msg.get("quoted_id")
        if qid is not None:
            if qid == msg["id"]:
                warning = f"Message '{msg['id']}' quotes itself"
                if strict_quotes:
                    raise DataValidationError(warning)
                validation_warnings.append(warning)
            elif qid not in messages_by_id:
                warning = f"Message '{msg['id']}' quotes non-existent message '{qid}'"
                if strict_quotes:
                    raise DataValidationError(warning)
                validation_warnings.append(warning)
            else:
                target = messages_by_id[qid]
                if target["chat_id"] != msg["chat_id"]:
                    warning = (
                        f"Message '{msg['id']}' in chat '{msg['chat_id']}' quotes message '{qid}' from chat '{target['chat_id']}'"
                    )
                    if strict_quotes:
                        raise DataValidationError(warning)
                    validation_warnings.append(warning)
                if target["ts"] > msg["ts"]:
                    warning = (
                        f"Message '{msg['id']}' (ts={msg['ts']}) quotes future message '{qid}' (ts={target['ts']})"
                    )
                    if strict_quotes:
                        raise DataValidationError(warning)
                    validation_warnings.append(warning)

    # 5. Check and preserve chronological ordering
    is_strictly_sorted = True
    is_non_decreasing = True
    inversions = 0

    for i in range(len(messages) - 1):
        curr_ts = messages[i]["ts"]
        next_ts = messages[i + 1]["ts"]
        if curr_ts >= next_ts:
            is_strictly_sorted = False
        if curr_ts > next_ts:
            is_non_decreasing = False
            inversions += 1

    if not messages or len(messages) == 1:
        timestamp_ordering_status = "strictly_sorted"
    elif is_strictly_sorted:
        timestamp_ordering_status = "strictly_sorted"
    elif is_non_decreasing:
        timestamp_ordering_status = "non_decreasing"
    else:
        timestamp_ordering_status = "unordered"
        validation_warnings.append(
            f"Messages are not in chronological order: {inversions} timestamp inversion(s) detected. Messages stably sorted by timestamp."
        )
        # Stably sort messages by timestamp
        messages = sorted(messages, key=lambda m: m["ts"])
        # Re-sort messages_by_chat stably by timestamp
        for cid in messages_by_chat:
            messages_by_chat[cid] = sorted(messages_by_chat[cid], key=lambda m: m["ts"])

    # 6. Build participants index
    # Must include every participant appearing in chats (including ME and silent participants)
    participants: Dict[str, Dict[str, Any]] = {}

    for chat in chats:
        cid = chat["chat_id"]
        is_grp = chat["is_group"]
        c_name = chat["name"].strip() if isinstance(chat.get("name"), str) else ""
        c_parts = chat["participants"]

        direct_contact_id: Optional[str] = None
        if not is_grp:
            others = [p for p in c_parts if p != "ME"]
            if len(others) == 1:
                direct_contact_id = others[0]

        for pid in c_parts:
            if pid not in participants:
                participants[pid] = {
                    "participant_id": pid,
                    "chat_ids": [],
                    "sender_names": set(),
                    "direct_chat_names": set(),
                }
            if cid not in participants[pid]["chat_ids"]:
                participants[pid]["chat_ids"].append(cid)

            # Store direct-chat contact names where available
            if not is_grp and c_name:
                if pid == direct_contact_id:
                    participants[pid]["direct_chat_names"].add(c_name)
                elif direct_contact_id is None and pid != "ME":
                    participants[pid]["direct_chat_names"].add(c_name)

    # Observe sender names from messages
    for msg in messages:
        sid = msg["sender_id"]
        sname = msg.get("sender_name")
        if isinstance(sname, str):
            clean_sname = sname.strip()
            if clean_sname and sid in participants:
                participants[sid]["sender_names"].add(clean_sname)

    # Convert sets to deterministic, JSON-serializable sorted lists
    for p in participants.values():
        p["sender_names"] = sorted(p["sender_names"])
        p["direct_chat_names"] = sorted(p["direct_chat_names"])

    # 7. Build validation summary
    validation_summary = {
        "total_chats": len(chats),
        "total_messages": len(messages),
        "direct_chats": direct_chat_count,
        "group_chats": group_chat_count,
        "unique_participants": len(participants),
        "unique_senders": len(unique_senders),
        "message_kinds": dict(message_kinds),
        "timestamp_ordering_status": timestamp_ordering_status,
        "validation_warnings": validation_warnings,
    }

    return {
        "chats": chats,
        "messages": messages,
        "seed": seed,
        "chats_by_id": chats_by_id,
        "messages_by_id": messages_by_id,
        "messages_by_chat": dict(messages_by_chat),
        "participants": participants,
        "validation_summary": validation_summary,
    }

