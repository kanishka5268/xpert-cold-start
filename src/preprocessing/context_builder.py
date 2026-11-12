"""Conversation context builder for chat-level and participant-centric histories."""

from collections import defaultdict
from typing import Any, Collection, Dict, List, Optional, Set, Union


def get_message_window(
    messages: List[Dict[str, Any]],
    limit: Optional[int] = None,
    since_ts: Optional[int] = None,
    until_ts: Optional[int] = None,
    kinds: Optional[Collection[str]] = None,
    reverse: bool = False,
) -> List[Dict[str, Any]]:
    """
    Return a window of messages filtered by timestamp range and kinds, with an optional recent-message limit.

    Args:
        messages: List of message dictionaries in chronological order.
        limit: Maximum number of recent messages to return (takes the last N messages).
               If None, returns all matching messages. If <= 0, returns an empty list.
        since_ts: Optional inclusive lower bound Unix timestamp.
        until_ts: Optional inclusive upper bound Unix timestamp.
        kinds: Optional collection of message kinds to include (e.g. {"text", "image_caption"}).
        reverse: If True, returns messages ordered from newest to oldest. Default is False (chronological).

    Returns:
        List of matching message dictionaries.
    """
    if limit is not None and limit <= 0:
        return []

    kinds_set = set(kinds) if kinds is not None else None

    # Filter messages
    filtered: List[Dict[str, Any]] = []
    has_ts_filter = since_ts is not None or until_ts is not None

    for msg in messages:
        if has_ts_filter:
            ts = msg.get("ts")
            if not isinstance(ts, int) or isinstance(ts, bool):
                continue
            if since_ts is not None and ts < since_ts:
                continue
            if until_ts is not None and ts > until_ts:
                continue

        if kinds_set is not None and msg.get("kind") not in kinds_set:
            continue
        filtered.append(msg)

    # Take the last `limit` messages (the most recent window)
    if limit is not None and len(filtered) > limit:
        filtered = filtered[-limit:]

    if reverse:
        return list(reversed(filtered))
    return filtered


def build_chat_histories(dataset: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Build chronological chat-level histories using chats and messages_by_chat.

    Args:
        dataset: Loaded dataset dictionary from load_dataset().

    Returns:
        Dictionary mapping chat_id -> chat history dict.
    """
    chats_by_id = dataset.get("chats_by_id", {})
    messages_by_chat = dataset.get("messages_by_chat", {})

    chat_histories: Dict[str, Dict[str, Any]] = {}

    for cid, chat in chats_by_id.items():
        # Retrieve messages for this chat; guaranteed chronological from ingestion
        msgs = list(messages_by_chat.get(cid, []))

        first_ts = msgs[0]["ts"] if msgs else None
        last_ts = msgs[-1]["ts"] if msgs else None

        chat_histories[cid] = {
            "chat_id": cid,
            "name": chat.get("name", ""),
            "is_group": chat.get("is_group", False),
            "participants": list(chat.get("participants", [])),
            "participant_count": len(chat.get("participants", [])),
            "messages": msgs,
            "message_count": len(msgs),
            "first_timestamp": first_ts,
            "last_timestamp": last_ts,
        }

    return chat_histories


def build_participant_histories(dataset: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Build participant-centric histories using the participants index, chats, and messages.

    Args:
        dataset: Loaded dataset dictionary from load_dataset().

    Returns:
        Dictionary mapping participant_id -> participant history dict.
    """
    chats_by_id = dataset.get("chats_by_id", {})
    participants_info = dataset.get("participants", {})
    all_messages = dataset.get("messages", [])

    # Index messages by sender and by participating chat in single chronological pass
    messages_sent_by_pid: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    all_messages_by_pid: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    direct_messages_by_pid: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for msg in all_messages:
        sid = msg.get("sender_id")
        cid = msg.get("chat_id")
        if sid:
            messages_sent_by_pid[sid].append(msg)

        chat = chats_by_id.get(cid)
        if chat:
            is_grp = chat.get("is_group", False)
            for pid in chat.get("participants", []):
                all_messages_by_pid[pid].append(msg)
                if not is_grp:
                    direct_messages_by_pid[pid].append(msg)

    participant_histories: Dict[str, Dict[str, Any]] = {}

    for pid, pdata in participants_info.items():
        p_chat_ids = pdata.get("chat_ids", [])

        direct_chat_ids = [
            cid for cid in p_chat_ids
            if cid in chats_by_id and not chats_by_id[cid].get("is_group", False)
        ]
        group_chat_ids = [
            cid for cid in p_chat_ids
            if cid in chats_by_id and chats_by_id[cid].get("is_group", False)
        ]

        sent = messages_sent_by_pid.get(pid, [])
        direct_msgs = direct_messages_by_pid.get(pid, [])
        all_msgs = all_messages_by_pid.get(pid, [])

        first_sent_ts = sent[0]["ts"] if sent else None
        last_sent_ts = sent[-1]["ts"] if sent else None

        participant_histories[pid] = {
            "participant_id": pid,
            "is_self": pid == "ME",
            "is_silent": len(sent) == 0,
            "sender_names": list(pdata.get("sender_names", [])),
            "direct_chat_names": list(pdata.get("direct_chat_names", [])),
            "chat_ids": list(p_chat_ids),
            "direct_chat_ids": direct_chat_ids,
            "group_chat_ids": group_chat_ids,
            "messages_sent": sent,
            "total_messages_sent": len(sent),
            "direct_messages": direct_msgs,
            "all_messages": all_msgs,
            "first_sent_timestamp": first_sent_ts,
            "last_sent_timestamp": last_sent_ts,
        }

    return participant_histories


class ConversationContext:
    """
    Unified container providing access to chat-level and participant-centric histories
    along with message-window filtering.
    """

    def __init__(
        self,
        dataset: Dict[str, Any],
        chats: Dict[str, Dict[str, Any]],
        participants: Dict[str, Dict[str, Any]],
    ):
        self.dataset = dataset
        self.chats = chats
        self.participants = participants
        self._summary = self._compute_summary()

    def _compute_summary(self) -> Dict[str, Any]:
        direct_count = sum(1 for c in self.chats.values() if not c["is_group"])
        group_count = sum(1 for c in self.chats.values() if c["is_group"])
        silent_count = sum(1 for p in self.participants.values() if p["is_silent"])
        return {
            "total_chats": len(self.chats),
            "direct_chats": direct_count,
            "group_chats": group_count,
            "total_participants": len(self.participants),
            "silent_participants": silent_count,
            "total_messages": len(self.dataset.get("messages", [])),
        }

    @property
    def summary(self) -> Dict[str, Any]:
        """Summary metrics of the conversation context."""
        return dict(self._summary)

    def get_chat(self, chat_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve chat-level history by chat_id, or None if not found."""
        return self.chats.get(chat_id)

    def get_chat_messages(
        self,
        chat_id: str,
        limit: Optional[int] = None,
        since_ts: Optional[int] = None,
        until_ts: Optional[int] = None,
        kinds: Optional[Collection[str]] = None,
        reverse: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve a window of messages for a specific chat.

        Args:
            chat_id: Chat identifier.
            limit: Maximum number of recent messages to return.
            since_ts: Optional inclusive lower bound Unix timestamp.
            until_ts: Optional inclusive upper bound Unix timestamp.
            kinds: Optional collection of message kinds to include.
            reverse: If True, return newest first; otherwise chronological.

        Returns:
            List of message dictionaries.
        """
        chat = self.get_chat(chat_id)
        if not chat:
            return []
        return get_message_window(
            chat["messages"],
            limit=limit,
            since_ts=since_ts,
            until_ts=until_ts,
            kinds=kinds,
            reverse=reverse,
        )

    def get_participant(self, participant_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve participant-centric history by participant_id, or None if not found."""
        return self.participants.get(participant_id)

    def get_participant_messages(
        self,
        participant_id: str,
        mode: str = "all",
        limit: Optional[int] = None,
        since_ts: Optional[int] = None,
        until_ts: Optional[int] = None,
        kinds: Optional[Collection[str]] = None,
        reverse: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve a window of messages for a specific participant.

        Args:
            participant_id: Participant identifier.
            mode: One of:
                  - "all": all messages in any chat where participant is a member.
                  - "direct": messages in direct chats involving participant.
                  - "sent": messages sent by participant.
            limit: Maximum number of recent messages to return.
            since_ts: Optional inclusive lower bound Unix timestamp.
            until_ts: Optional inclusive upper bound Unix timestamp.
            kinds: Optional collection of message kinds to include.
            reverse: If True, return newest first; otherwise chronological.

        Returns:
            List of message dictionaries.
        """
        part = self.get_participant(participant_id)
        if not part:
            return []

        if mode == "all":
            msgs = part["all_messages"]
        elif mode == "direct":
            msgs = part["direct_messages"]
        elif mode == "sent":
            msgs = part["messages_sent"]
        else:
            raise ValueError(f"Invalid mode '{mode}'. Expected 'all', 'direct', or 'sent'.")

        return get_message_window(
            msgs,
            limit=limit,
            since_ts=since_ts,
            until_ts=until_ts,
            kinds=kinds,
            reverse=reverse,
        )

    def __getitem__(self, key: str) -> Any:
        if key == "chats":
            return self.chats
        if key == "participants":
            return self.participants
        if key == "summary":
            return self.summary
        if key == "dataset":
            return self.dataset
        raise KeyError(f"Invalid key '{key}'. Available: 'chats', 'participants', 'summary', 'dataset'.")

    def __contains__(self, key: str) -> bool:
        return key in ("chats", "participants", "summary", "dataset")

    def to_dict(self) -> Dict[str, Any]:
        """Convert context to a serializable dictionary representation."""
        return {
            "chats": self.chats,
            "participants": self.participants,
            "summary": self.summary,
        }


def build_conversation_context(dataset: Dict[str, Any]) -> ConversationContext:
    """
    Build a unified ConversationContext from a dataset loaded by load_dataset().

    Args:
        dataset: Loaded dataset dictionary containing chats, messages, participants, etc.

    Returns:
        ConversationContext instance with indexed chat-level and participant-centric histories.
    """
    if not isinstance(dataset, dict):
        raise TypeError(f"Expected dataset dictionary, got {type(dataset).__name__}")

    chat_histories = build_chat_histories(dataset)
    participant_histories = build_participant_histories(dataset)

    return ConversationContext(
        dataset=dataset,
        chats=chat_histories,
        participants=participant_histories,
    )

