"""Organisation canonical resolution, entity disambiguation, and deterministic ID allocation."""

from dataclasses import dataclass, field
import hashlib
import re
from typing import Dict, List, Optional, Set, Tuple, Union


GENUINE_LEGAL_SUFFIXES: Set[str] = {
    "pvt ltd",
    "private limited",
    "ltd",
    "limited",
    "llp",
    "inc",
    "incorporated",
    "corp",
    "corporation",
    "co",
    "company",
    "plc",
    "gmbh",
}

CONTEXTUAL_DESCRIPTORS: Set[str] = {
    "team",
    "chat",
    "channel",
    "updates",
}

MEANINGFUL_NAME_COMPONENTS: Set[str] = {
    "design",
    "labs",
    "technologies",
    "solutions",
    "media",
    "group",
    "studio",
    "daas",
    "capital",
    "services",
    "consulting",
    "enterprises",
    "holdings",
    "agency",
}

# Preserve backward compatibility for any external import of LEGAL_SUFFIXES:
LEGAL_SUFFIXES: Set[str] = GENUINE_LEGAL_SUFFIXES


@dataclass
class CandidateOrganisation:
    """Provisional organisation candidate tracked across the extraction lifecycle."""
    name: str
    is_seed: bool
    org_type: str = "other"
    side: str = "home"
    role: str = "other"
    attached_chats: Set[str] = field(default_factory=set)
    linked_people: Set[str] = field(default_factory=set)
    evidence_msgs: Set[str] = field(default_factory=set)
    confidence: float = 1.0
    id: Optional[str] = None

    @property
    def member_count(self) -> int:
        """Total distinct attached chats and linked people."""
        return len(self.attached_chats) + len(self.linked_people)


def generate_org_id(canonical_name: str, allocated_ids: Set[str]) -> str:
    """
    Generate a stable, deterministic, collision-safe organisation ID.

    Format: org_<clean_slug>
    If slug collides with an existing organisation, appends a deterministic
    6-character SHA-256 hash of the canonical name.

    Args:
        canonical_name: Canonical string name of the organisation.
        allocated_ids: Set of IDs already allocated in this run.

    Returns:
        Deterministic, collision-safe ID string.
    """
    clean_slug = re.sub(r"[^a-z0-9]+", "_", canonical_name.lower().strip()).strip("_")
    if not clean_slug:
        clean_slug = "org"

    base_id = f"org_{clean_slug}"
    if base_id not in allocated_ids:
        allocated_ids.add(base_id)
        return base_id

    # Collision detected: append stable 6-char SHA-256 hash
    h = hashlib.sha256(canonical_name.encode("utf-8")).hexdigest()[:6]
    cand_id = f"{base_id}_{h}"
    if cand_id not in allocated_ids:
        allocated_ids.add(cand_id)
        return cand_id

    # If still colliding, append incremental numerical suffix
    idx = 1
    while f"{cand_id}_{idx}" in allocated_ids:
        idx += 1
    final_id = f"{cand_id}_{idx}"
    allocated_ids.add(final_id)
    return final_id


def normalize_org_name(name: str) -> str:
    """Clean extra punctuation, quotes, and whitespace from organisation strings."""
    if not name:
        return ""
    # Strip quotes, brackets, and extraneous symbols
    cleaned = re.sub(r'[\'"`\[\]{}()]', "", name).strip()
    return re.sub(r"\s+", " ", cleaned)


def strip_legal_suffixes(name: str) -> str:
    """
    Strip only genuine corporate legal entity suffixes for alias matching.
    Preserves meaningful name components (e.g. Design, Labs, Technologies, Group, Studio).
    """
    cleaned = normalize_org_name(name)
    tokens = cleaned.split()
    if not tokens:
        return cleaned

    # Check two-token legal suffixes first (e.g. "pvt ltd", "private limited")
    if len(tokens) >= 2:
        last_two = f"{tokens[-2]} {tokens[-1]}".lower()
        if last_two in GENUINE_LEGAL_SUFFIXES:
            tokens = tokens[:-2]

    # Check single-token legal suffixes (e.g. "ltd", "inc", "corp", "co", "company", "llp")
    if tokens and tokens[-1].lower() in GENUINE_LEGAL_SUFFIXES:
        tokens.pop()

    return " ".join(tokens) if tokens else cleaned


def are_distinct_entities(
    name1: str,
    name2: str,
    chats1: Optional[Set[str]] = None,
    chats2: Optional[Set[str]] = None,
    people1: Optional[Set[str]] = None,
    people2: Optional[Set[str]] = None,
) -> bool:
    """
    Determine whether two organisation names represent distinct entities.

    Two names are distinct if:
    1. Their normalized forms differ after stripping ONLY genuine legal suffixes.
    2. Any substantive token differences exist (including meaningful business descriptors).
    3. Seed precedence or distinct ventures must never be falsely collapsed.
    """
    norm1 = normalize_org_name(name1).lower()
    norm2 = normalize_org_name(name2).lower()

    if norm1 == norm2:
        return False

    stripped1 = strip_legal_suffixes(name1).lower()
    stripped2 = strip_legal_suffixes(name2).lower()

    if stripped1 == stripped2:
        # Names differ ONLY by genuine legal suffixes (e.g. "PostMyAd" vs "PostMyAd Pvt Ltd")
        return False

    tokens1 = set(stripped1.split())
    tokens2 = set(stripped2.split())
    substantive_diff = (tokens1 ^ tokens2) - GENUINE_LEGAL_SUFFIXES

    if substantive_diff:
        # If there are substantive different tokens (e.g. 'daas', 'design', 'labs'),
        # they are distinct entities!
        return True

    return False


def match_canonical_seed_org(raw_name: str, seed_orgs: List[str]) -> Optional[str]:
    """
    Match a raw organisation name against foundational seed organisations.
    Only strips genuine legal suffixes; preserves distinct ventures.
    """
    if not raw_name or not seed_orgs:
        return None

    clean_raw = normalize_org_name(raw_name)
    raw_lower = clean_raw.lower()
    raw_stripped = strip_legal_suffixes(clean_raw).lower()

    # 1. Exact match (case-insensitive)
    for s in seed_orgs:
        if raw_lower == s.lower():
            return s

    # 2. Stripped genuine legal suffix match
    for s in seed_orgs:
        s_stripped = strip_legal_suffixes(s).lower()
        if raw_stripped == s_stripped:
            return s

    # 3. Substring match ONLY if remainder is a genuine legal suffix or informal descriptor
    for s in seed_orgs:
        s_lower = s.lower()
        if raw_lower.startswith(s_lower):
            remainder = raw_lower[len(s_lower):].strip()
            if remainder in GENUINE_LEGAL_SUFFIXES or remainder in CONTEXTUAL_DESCRIPTORS:
                return s
        if s_lower.startswith(raw_lower):
            remainder = s_lower[len(raw_lower):].strip()
            if remainder in GENUINE_LEGAL_SUFFIXES:
                return s

    return None


def find_candidate_org(
    raw_name: str,
    candidates: Union[Dict[str, CandidateOrganisation], List[CandidateOrganisation]],
    seed_org_names: Optional[List[str]] = None,
) -> Optional[CandidateOrganisation]:
    """
    Match an observed organisation mention against candidate organisations.

    Supports seed canonical matching, stripped genuine legal suffix matching,
    while strictly preserving distinct ventures.
    """
    if not raw_name or not isinstance(raw_name, str):
        return None

    clean_raw = normalize_org_name(raw_name).strip()
    if not clean_raw:
        return None

    cand_list: List[CandidateOrganisation] = (
        list(candidates.values()) if isinstance(candidates, dict) else list(candidates)
    )

    # 1. Match seed organisations if seed names provided (seed precedence anchoring)
    if seed_org_names:
        matched_seed = match_canonical_seed_org(clean_raw, seed_org_names)
        if matched_seed:
            for cand in cand_list:
                if cand.name.lower() == matched_seed.lower():
                    return cand

    raw_lower = clean_raw.lower()
    raw_stripped = strip_legal_suffixes(clean_raw).lower()

    # 2. Exact match against candidate names (case-insensitive)
    for cand in cand_list:
        if cand.name.lower() == raw_lower:
            return cand

    # 3. Stripped genuine legal suffix match (e.g. "PostMyAd Pvt Ltd" vs "PostMyAd")
    for cand in cand_list:
        cand_stripped = strip_legal_suffixes(cand.name).lower()
        if raw_stripped == cand_stripped:
            return cand

    # 4. Suffix match ONLY if remainder is a genuine legal suffix
    for cand in cand_list:
        cand_lower = cand.name.lower()
        if raw_lower.startswith(cand_lower):
            remainder = raw_lower[len(cand_lower):].strip()
            if remainder in GENUINE_LEGAL_SUFFIXES:
                return cand
        if cand_lower.startswith(raw_lower):
            remainder = cand_lower[len(raw_lower):].strip()
            if remainder in GENUINE_LEGAL_SUFFIXES:
                return cand

    return None
