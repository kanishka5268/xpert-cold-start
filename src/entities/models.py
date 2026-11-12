"""Strongly typed dataclasses, configuration, and validation for People, Organisation, and Chat entities."""

from dataclasses import dataclass, field
import math
import time
from typing import Any, Dict, List, Optional, Set


# Official schema-compliant enums
VALID_CIRCLES: Set[str] = {
    "spouse", "child", "parent", "in-law", "sibling", "relative",
    "friend", "colleague", "client", "vendor", "partner", "staff"
}

VALID_ORG_TYPES: Set[str] = {
    "company", "residence", "vacation home", "workspace", "society", "school", "other"
}

VALID_ORG_SIDES: Set[str] = {"home", "work"}

VALID_ORG_ROLES: Set[str] = {
    "owner", "partner", "helper", "resident", "parent", "member", "other"
}

VALID_ORG_LINKS: Set[str] = {"runs", "supports"}


def is_valid_confidence(conf: Any, min_threshold: float) -> bool:
    """
    Validate that confidence is a finite numeric value in [0.0, 1.0] and >= min_threshold.

    Rejects:
    - None
    - booleans (True/False)
    - strings, dicts, lists
    - NaN, positive/negative infinity
    - values below 0.0 or above 1.0
    - values strictly below min_threshold
    """
    if conf is None or isinstance(conf, bool):
        return False
    if not isinstance(conf, (int, float)):
        return False
    if math.isnan(conf) or math.isinf(conf):
        return False
    if conf < 0.0 or conf > 1.0:
        return False
    return conf >= min_threshold


def is_valid_llm_person_record(p_rec: Any, min_conf: float) -> bool:
    """
    Validate an InterpretedPerson record for LLM-derived circle resolution.

    Requirements:
    1. p_rec exists
    2. p_rec.circle is in VALID_CIRCLES
    3. p_rec.confidence is a valid finite numeric value >= min_conf
    4. p_rec.evidence_msgs has at least one valid message reference
    """
    if not p_rec:
        return False
    circle = getattr(p_rec, "circle", None)
    if circle not in VALID_CIRCLES:
        return False
    conf = getattr(p_rec, "confidence", None)
    if not is_valid_confidence(conf, min_conf):
        return False
    ev_msgs = getattr(p_rec, "evidence_msgs", None)
    if not ev_msgs or not isinstance(ev_msgs, (list, set, tuple)):
        return False
    if not any(isinstance(m, str) and m.strip() for m in ev_msgs):
        return False
    return True


def is_valid_llm_org_link(p_rec: Any, min_conf: float) -> bool:
    """
    Validate that an InterpretedPerson record has a valid LLM-grounded organisation link.

    Requirements:
    1. p_rec exists and has non-empty associated_org_names
    2. p_rec.confidence is a valid finite numeric value >= min_conf
    """
    if not p_rec:
        return False
    org_names = getattr(p_rec, "associated_org_names", None)
    if not org_names or not isinstance(org_names, (list, set, tuple)):
        return False
    if not any(isinstance(o, str) and o.strip() for o in org_names):
        return False
    conf = getattr(p_rec, "confidence", None)
    return is_valid_confidence(conf, min_conf)


@dataclass(frozen=True)
class EntityExtractionConfig:
    """Configurable calibration parameters for entity extraction and synthesis."""

    # Message engagement floor for substantive group participation
    min_substantive_group_messages: int = 3

    # Minimum ratio of non-ME chat participants belonging to an organisation to infer chat association
    min_org_participant_overlap: float = 0.60

    # Minimum model-reported confidence floors
    min_llm_circle_confidence: float = 0.65
    min_llm_org_confidence: float = 0.70

    # Safety and pruning flags
    prune_zero_member_non_seed_orgs: bool = True
    strict_staff_looks_after: bool = True


class EntityExtractionError(ValueError):
    """Raised when entity extraction or chat routing encounters an unresolvable structural error."""
    pass


class UnresolvableChatMappingError(EntityExtractionError):
    """Raised when a chat cannot be mapped to any valid person or organisation target without guessing."""
    pass


@dataclass
class UnresolvableSoloChatDiagnostic:
    """Diagnostic record for a chat containing only 'ME' that lacks independently grounded evidence."""
    chat_id: str
    reason: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class EntityExtractionDiagnostics:
    """Detailed operational diagnostics collected during entity extraction."""
    solo_me_chats_unresolvable: List[UnresolvableSoloChatDiagnostic] = field(default_factory=list)
    pruned_orphan_org_names: List[str] = field(default_factory=list)
    unsupported_circle_people_omitted: List[str] = field(default_factory=list)


@dataclass
class EntitiesResult:
    """Strongly typed output of the entity extraction layer with referential integrity."""
    people: List[Dict[str, Any]]
    organisations: List[Dict[str, Any]]
    chats: List[Dict[str, Any]]
    diagnostics: EntityExtractionDiagnostics = field(default_factory=EntityExtractionDiagnostics)

    def to_dict(self) -> Dict[str, Any]:
        """Convert extracted entities to final dictionary representation."""
        return {
            "people": self.people,
            "organisations": self.organisations,
            "chats": self.chats,
        }


def validate_entities_referential_integrity(
    result: EntitiesResult,
    input_chat_ids: Set[str],
    input_person_ids: Set[str],
) -> List[str]:
    """
    Validate that extracted entities strictly satisfy all schema constraints and referential integrity.

    Args:
        result: EntitiesResult containing people, organisations, and chats.
        input_chat_ids: Exhaustive set of valid chat IDs from input.
        input_person_ids: Exhaustive set of valid non-ME participant IDs from input.

    Returns:
        List of validation error strings. An empty list signifies complete structural validity.
    """
    errors: List[str] = []

    org_ids: Set[str] = set()
    for idx, org in enumerate(result.organisations):
        where = f"organisations[{idx}]"
        if not isinstance(org, dict):
            errors.append(f"{where}: expected object, got {type(org).__name__}")
            continue
        oid = org.get("id")
        if not oid or not isinstance(oid, str):
            errors.append(f"{where}: missing or invalid id")
            continue
        if oid in org_ids:
            errors.append(f'{where}: duplicate organisation id "{oid}"')
        org_ids.add(oid)

        if not org.get("name") or not isinstance(org["name"], str):
            errors.append(f"{where}: missing or invalid name")
        if org.get("type") not in VALID_ORG_TYPES:
            errors.append(f"{where}: invalid type '{org.get('type')}', expected one of {sorted(VALID_ORG_TYPES)}")
        if org.get("side") not in VALID_ORG_SIDES:
            errors.append(f"{where}: invalid side '{org.get('side')}', expected 'home' or 'work'")
        if org.get("role") not in VALID_ORG_ROLES:
            errors.append(f"{where}: invalid role '{org.get('role')}', expected one of {sorted(VALID_ORG_ROLES)}")

    people_ids: Set[str] = set()
    for idx, person in enumerate(result.people):
        where = f"people[{idx}]"
        if not isinstance(person, dict):
            errors.append(f"{where}: expected object, got {type(person).__name__}")
            continue
        pid = person.get("id")
        if not pid or not isinstance(pid, str):
            errors.append(f"{where}: missing or invalid id")
            continue
        if pid == "ME":
            errors.append(f"{where}: 'ME' must never be included in people")
        if pid not in input_person_ids:
            errors.append(f'{where}: unknown participant id "{pid}" not in input')
        if pid in people_ids:
            errors.append(f'{where}: duplicate person id "{pid}"')
        people_ids.add(pid)

        if person.get("circle") not in VALID_CIRCLES:
            errors.append(f"{where}: invalid circle '{person.get('circle')}', expected one of {sorted(VALID_CIRCLES)}")

        for link_idx, link in enumerate(person.get("links", []) or []):
            link_where = f"{where}.links[{link_idx}]"
            target_org = link.get("organisation")
            if target_org not in org_ids:
                errors.append(f'{link_where}: references unknown organisation "{target_org}"')
            if link.get("link") not in VALID_ORG_LINKS:
                errors.append(f"{link_where}: invalid link type '{link.get('link')}', expected 'runs' or 'supports'")

        for other_idx, other in enumerate(person.get("looks_after", []) or []):
            if other not in input_person_ids:
                errors.append(f'{where}.looks_after[{other_idx}]: references unknown person "{other}"')

    tied_chats: Set[str] = set()
    for idx, chat in enumerate(result.chats):
        where = f"chats[{idx}]"
        if not isinstance(chat, dict):
            errors.append(f"{where}: expected object, got {type(chat).__name__}")
            continue
        cid = chat.get("chat_id")
        if not cid or not isinstance(cid, str):
            errors.append(f"{where}: missing or invalid chat_id")
            continue
        if cid not in input_chat_ids:
            errors.append(f'{where}: unknown chat_id "{cid}" not in input')
        if cid in tied_chats:
            errors.append(f'{where}: chat "{cid}" is tied multiple times')
        tied_chats.add(cid)

        has_p = "person" in chat
        has_o = "organisation" in chat
        if has_p == has_o:
            errors.append(f"{where}: chat must have exactly one of 'person' or 'organisation'")
        elif has_p:
            p_target = chat["person"]
            if p_target not in input_person_ids or p_target == "ME":
                errors.append(f'{where}: person target "{p_target}" is invalid or "ME"')
        elif has_o:
            o_target = chat["organisation"]
            if o_target not in org_ids:
                errors.append(f'{where}: organisation target "{o_target}" is not an emitted organisation')

    missing = input_chat_ids - tied_chats
    if missing:
        errors.append(f"{len(missing)} chat(s) from input are missing from output: {sorted(missing)[:5]}")

    return errors

