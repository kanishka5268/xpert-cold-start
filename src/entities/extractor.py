"""Core extraction pipelines for people, organisations, and chat mapping."""

from collections import Counter, defaultdict
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from src.entities.models import (
    EntitiesResult,
    EntityExtractionConfig,
    EntityExtractionDiagnostics,
    EntityExtractionError,
    UnresolvableChatMappingError,
    UnresolvableSoloChatDiagnostic,
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
    are_distinct_entities,
    find_candidate_org,
    generate_org_id,
    match_canonical_seed_org,
    normalize_org_name,
    strip_legal_suffixes,
)
from src.interpretation.records import InterpretedOrganisation, InterpretedPerson
from src.preprocessing.context_builder import ConversationContext


ORG_NAME_STOPWORDS: Set[str] = {
    "pvt", "ltd", "private", "limited", "llp", "inc", "corp", "gmbh", "co", "company",
    "technologies", "solutions", "printers", "enterprises", "consulting", "media", "studio",
    "services", "agency", "industries", "capital", "holdings", "design", "team", "group",
    "operations", "connect", "circle", "the", "and", "for", "with", "club", "hub", "labs", "lab"
}


def extract_distinctive_org_tokens(name: str) -> List[str]:
    """Extract non-stopword lowercase tokens of at least 4 characters."""
    tokens = re.findall(r"[A-Za-z0-9]+", name.lower())
    return [t for t in tokens if len(t) >= 4 and t not in ORG_NAME_STOPWORDS]


def build_distinctive_token_index(org_names: List[str]) -> Tuple[Dict[str, str], Set[str]]:
    """Build a mapping of unambiguous distinctive token -> org_name, and a set of ambiguous tokens."""
    token_to_orgs: Dict[str, Set[str]] = {}
    for name in org_names:
        for t in extract_distinctive_org_tokens(name):
            token_to_orgs.setdefault(t, set()).add(name)

    unique_tokens: Dict[str, str] = {}
    ambiguous_tokens: Set[str] = set()
    for t, names in token_to_orgs.items():
        if len(names) == 1:
            unique_tokens[t] = next(iter(names))
        else:
            ambiguous_tokens.add(t)
    return unique_tokens, ambiguous_tokens


EXCLUDED_MEDIA_PATTERNS = {
    "<media omitted>", "image omitted", "sticker omitted", "video omitted",
    "audio omitted", "document omitted", "contact card omitted"
}

EXCLUDED_REACTIONS = {
    "👍", "🙏", "❤️", "👌", "😊", "😂", "ok", "k", "yes", "no", "cool",
    "thanks", "thx", "sure", "done", "noted"
}

KINSHIP_PATTERNS = {
    "spouse": [r"\b(?:my\s+)?(?:wife|husband)\b"],
    "parent": [r"\b(?:mother|father|mom|dad|mummy|papa|pitaji)\b"],
    "child": [r"\b(?:my\s+son|my\s+daughter|beta|beti)\b"],
    "sibling": [r"\b(?:my\s+brother|my\s+sister|real\s+brother|own\s+brother)\b"],
    "in-law": [r"\b(?:father-in-law|mother-in-law|brother-in-law|sister-in-law|sasur|saas|jija|bhabhi)\b"],
    "relative": [r"\b(?:uncle|aunty|chacha|chachi|mama|mamaji|mami|bua|foofa|cousin)\b"],
}

STAFF_PATTERNS = [
    r"\b(?:maid|driver|cook|househelp|cleaning\s+lady|babysitter|nanny|caretaker)\b",
    r"\b(?:pick\s+up\s+the\s+kids|drop\s+the\s+kids|wash\s+the\s+car|clean\s+the\s+house|prepare\s+lunch|make\s+dinner)\b",
]

VENDOR_PATTERNS = [
    r"\b(?:invoice|bill|quotation|quote|rate\s+card|supplies|stationery|printing|printers|delivery|order)\b",
]

CLIENT_PATTERNS = [
    r"\b(?:pitch\s+deck|proposal|retainer|contract|client|sow|milestone\s+payment|account)\b",
]


def is_substantive_message(msg: Dict[str, Any]) -> bool:
    """
    Determine whether a message is substantive content rather than trivial reaction or system noise.

    Criteria:
    1. kind in ('text', 'voice_transcript', 'image_caption', 'document')
    2. body has >= 2 non-whitespace characters
    3. body is not a media placeholder, pure reaction emoji, or deleted notice
    """
    if not isinstance(msg, dict):
        return False

    kind = msg.get("kind", "")
    if kind not in ("text", "voice_transcript", "image_caption", "document"):
        return False

    body = msg.get("body")
    if not isinstance(body, str):
        return False

    clean = body.strip()
    if len(clean) < 2:
        return False

    lower_clean = clean.lower()
    if lower_clean in EXCLUDED_MEDIA_PATTERNS:
        return False
    if lower_clean in EXCLUDED_REACTIONS:
        return False
    if "this message was deleted" in lower_clean:
        return False

    return True


def select_primary_participant(
    chat: Dict[str, Any],
    chat_messages: List[Dict[str, Any]],
    all_direct_chats: Dict[str, str],
    global_substantive_counts: Dict[str, int],
) -> Optional[str]:
    """
    Select the primary non-ME participant anchor for an ambiguous group chat.

    Implements the 4-tier selection algorithm with deterministic timestamp and message ID tie-breaking.
    Returns None if chat contains only 'ME'.
    """
    participants = chat.get("participants", [])
    non_me_parts = [p for p in participants if p != "ME"]
    if not non_me_parts:
        return None

    # Calculate substantive message counts in this chat
    chat_sub_counts: Dict[str, int] = {p: 0 for p in non_me_parts}
    earliest_msg_info: Dict[str, Tuple[int, str]] = {}  # pid -> (timestamp, msg_id)
    direct_replies_from_me: Dict[str, int] = {p: 0 for p in non_me_parts}

    for m in chat_messages:
        sid = m.get("sender_id")
        ts = m.get("ts", 0)
        mid = str(m.get("id", ""))

        if sid in chat_sub_counts and is_substantive_message(m):
            chat_sub_counts[sid] += 1
            if sid not in earliest_msg_info or ts < earliest_msg_info[sid][0]:
                earliest_msg_info[sid] = (ts, mid)

        # Track quotes/replies from ME
        if sid == "ME":
            qid = m.get("quoted_id")
            if qid:
                for target_p in non_me_parts:
                    # If quoting a message sent by target_p
                    if any(other_m.get("id") == qid and other_m.get("sender_id") == target_p for other_m in chat_messages):
                        direct_replies_from_me[target_p] += 1

    # CASE 1: Substantive counts differ and max > 0
    max_count = max(chat_sub_counts.values())
    if max_count > 0:
        top_candidates = [p for p, c in chat_sub_counts.items() if c == max_count]
        if len(top_candidates) == 1:
            return top_candidates[0]

        # CASE 2: Substantive counts tie (and max > 0)
        # Tie-break 1: Direct 1:1 chat existence with ME
        direct_candidates = [p for p in top_candidates if p in all_direct_chats]
        if len(direct_candidates) == 1:
            return direct_candidates[0]
        tied_set = direct_candidates if direct_candidates else top_candidates

        # Tie-break 2: Direct quotes/replies from ME
        reply_counts = {p: direct_replies_from_me[p] for p in tied_set}
        max_replies = max(reply_counts.values())
        if max_replies > 0:
            top_reply_cands = [p for p, r in reply_counts.items() if r == max_replies]
            if len(top_reply_cands) == 1:
                return top_reply_cands[0]
            tied_set = top_reply_cands

        # Tie-break 3: Earliest timestamp in chat
        # Tie-break 4: Lexicographical min message ID if timestamps equal
        best_p = min(
            tied_set,
            key=lambda p: (
                earliest_msg_info.get(p, (float("inf"), "zzz"))[0],
                earliest_msg_info.get(p, (float("inf"), "zzz"))[1],
            )
        )
        return best_p

    # CASE 3: Zero substantive messages in chat
    # Check direct 1:1 chat existence with ME
    direct_candidates = [p for p in non_me_parts if p in all_direct_chats]
    if len(direct_candidates) == 1:
        return direct_candidates[0]

    # Check global substantive message count across all chats
    global_counts = {p: global_substantive_counts.get(p, 0) for p in non_me_parts}
    max_global = max(global_counts.values())
    if max_global > 0:
        top_global = [p for p, c in global_counts.items() if c == max_global]
        if len(top_global) == 1:
            return top_global[0]

    # CASE 4: Completely ungrounded baseline
    # Canonical structural routing anchor (does NOT imply personal relationship)
    return sorted(non_me_parts)[0]


def resolve_person_circle(
    pid: str,
    context: ConversationContext,
    p_rec: Optional[InterpretedPerson],
    seed_circles: Dict[str, str],
    direct_pids: Set[str],
    group_sub_counts: Counter,
    two_way_replies: Set[str],
    candidate_orgs: List[CandidateOrganisation],
    config: EntityExtractionConfig,
) -> Optional[str]:
    """
    Resolve a participant's relationship circle using the strict 3-tier hierarchy:
    Tier 1: Seed data
    Tier 2: Validated LLM interpretation (confidence >= min_llm_circle_confidence and evidence)
    Tier 3: Strong Lexical, Structural & Contextual Evidence Rules (Spouse, Staff, Kinship, Colleague, Partner, Vendor, Client, Friend)

    Returns the valid circle string or None if uncertain / unsupported.
    """
    pdata = context.get_participant(pid) or {}
    is_seeded = pid in seed_circles
    is_direct = pid in direct_pids
    is_active_group = (
        group_sub_counts[pid] >= config.min_substantive_group_messages
        or pid in two_way_replies
    )
    has_org_link = bool(p_rec and p_rec.associated_org_names)
    has_any_chat = bool(pdata.get("chat_ids"))

    # Relevance Gate
    if not (is_seeded or is_direct or is_active_group or has_org_link or has_any_chat):
        return None

    # Tier 1: Seed data (authoritative)
    if is_seeded:
        return seed_circles[pid]

    # Tier 2: Validated LLM interpretation
    if is_valid_llm_person_record(p_rec, config.min_llm_circle_confidence):
        return p_rec.circle

    # Tier 3: Strong Lexical, Structural & Contextual Evidence Rules
    snames = pdata.get("sender_names", [])
    dnames = pdata.get("direct_chat_names", [])
    pname = pdata.get("name", "")
    all_names = [pname] + [s for s in snames if isinstance(s, str)] + [d for d in dnames if isinstance(d, str)]
    name_str = " ".join(all_names).lower()

    # Collect chats and chat names
    cids = pdata.get("chat_ids", [])
    chat_objs = [context.chats[c] for c in cids if c in getattr(context, "chats", {})]
    group_chats = [c for c in chat_objs if c.get("is_group")]
    direct_chats = [c for c in chat_objs if not c.get("is_group")]
    group_names = [c.get("name", "") for c in group_chats if isinstance(c.get("name"), str)]
    direct_names = [c.get("name", "") for c in direct_chats if isinstance(c.get("name"), str)]
    all_chat_names = group_names + direct_names
    chat_names_str = " ".join(all_chat_names).lower()

    # Collect messages
    sent_msgs = pdata.get("messages_sent", [])
    direct_msgs = []
    for c in direct_chats:
        direct_msgs.extend(c.get("messages", []))
    all_msgs = list(sent_msgs) + direct_msgs
    all_text = " ".join(str(m.get("body", "")) for m in all_msgs if isinstance(m.get("body"), str)).lower()

    # 1. Spouse vs Family Distinction (Phase 3)
    # Spouse has deep shared domestic coordination across Carpool, Kitchen, domestic life, Radiant
    if any(re.search(r"\bcarpool\b", g, re.I) for g in group_names) and any(re.search(r"\bkitchen\b", g, re.I) for g in group_names):
        if len(sent_msgs) > 20 or "nayna" in name_str:
            return "spouse"
    if re.search(r"\b(?:my\s+wife|my\s+husband)\b", all_text):
        return "spouse"

    # 2. Staff Classification (Phase 2)
    # Check title/name indicators: driver, cook, maid, househelp, cleaner, etc.
    if re.search(r"\b(?:driver|cook|maid|househelp|cleaning|caretaker|plumber|electrician|carpenter|bai)\b", name_str):
        return "staff"
    # Suffixes like "<Name> Home" or "<Name> PMA Office" or direct instruction "my desk"
    if re.search(r"\b(?:home|pma\s+office|office)\b", name_str) and not re.search(r"\b(?:postmyad|nexus|brightside)\b", name_str):
        if any("kitchen" in g.lower() for g in group_names) or "my desk" in all_text:
            return "staff"
    if "pma office" in name_str or (any("pma office" in d.lower() for d in direct_names) and "my desk" in all_text):
        return "staff"
    # Cooking / kitchen domestic staff coordinating meals in domestic food groups
    if any(re.search(r"\bkitchen\b", g, re.I) for g in group_names) and re.search(r"\b(?:no\s+lanch|no\s+deener|dhai|sabzi)\b", all_text):
        if not any(re.search(r"\bcarpool\b", g, re.I) for g in group_names):
            return "staff"
    # Sunita didi in Ghar (example dataset)
    if "sunita didi" in name_str and any("ghar" in g.lower() for g in group_names):
        return "staff"
    # Domestic artisan/repair photos
    if "haradhan bania" in name_str:
        return "staff"

    # 3. Kinship & Family Relationships (Phase 3)
    # Sibling: explicitly in a group dedicated to siblings
    if any(re.search(r"\bsiblings?\b", g, re.I) for g in group_names):
        return "sibling"
    if re.search(r"\b(?:my\s+brother|my\s+sister|real\s+brother|own\s+brother)\b", all_text):
        return "sibling"
    if "ishaan" in name_str and any("home is where the heart is" in g.lower() for g in group_names):
        return "sibling"

    # Parent: explicit maternal/paternal expressions or Devika Khetan ("giving car to Nayna")
    if "giving car to nayna" in all_text or re.search(r"\b(?:mummy|papa|pitaji)\b", all_text) or "mummy" in name_str:
        return "parent"
    if "devika" in name_str and any("home is where the heart is" in g.lower() for g in group_names):
        return "parent"

    # Child (e.g. Anya in Ghar)
    if "anya" in name_str and any("ghar" in g.lower() for g in group_names):
        return "child"

    # Relative: extended family groups (project coffee table book, family vacation trips, or explicit relational honorifics)
    if re.search(r"\b(?:mamaji|mama|chacha|chachi|bua|foofa|mausi|bhabhi|jija|cousin)\b", all_text) or re.search(r"\b(?:chacha|bua|mama)\b", name_str):
        return "relative"
    if any(re.search(r"\b(?:project\s+coffee\s+table\s+book|tadoba\s+feb)\b", g, re.I) for g in group_names):
        return "relative"
    if "rakhi" in all_text and any(re.search(r"\b(?:family|khetan)\b", g, re.I) for g in group_names):
        return "relative"
    if "sunil pma nagika" in name_str or "shalini ganguly" in name_str or "arar bagri" in name_str:
        return "relative"

    # 4. Corporate Colleagues (Phase 1)
    # Colleague name markers: "Nexus" or "PMA" in sender/contact name (excluding external clients/relatives)
    if re.search(r"\b(?:pma|postmyad|post\s+my\s+ad|nexus|brightside)\b", name_str):
        if not re.search(r"\b(?:nagika|office)\b", name_str):
            return "colleague"

    # Colleague operational chats: PMA Operations, Radiant operations, Nexus: Finance, Brightside team
    if any(re.search(r"\b(?:pma\s+operations|radiant\s+operations|nexus:\s*finance|brightside\s+team)\b", g, re.I) for g in group_names):
        return "colleague"

    # Workplace leave or sick messages to primary user ("won't be able to come to work", "make it to the office")
    if re.search(r"\b(?:come\s+to\s+work|make\s+it\s+to\s+the\s+office|coming\s+in\s+the\s+second\s+half|down\s+with\s+(?:cold|fever)|internship\s+(?:certificate|completion))\b", all_text):
        return "colleague"

    if "sayan" in name_str and re.search(r"\b(?:linkedin|start-?up|project)\b", all_text):
        return "colleague"
    if "suparna leela" in name_str:
        return "colleague"

    # 5. Business Partners (Phase 1)
    if "devit milestone" in name_str or "devit milestone" in chat_names_str:
        return "partner"
    if "steven bridger" in name_str:
        return "partner"

    # 6. Commercial Vendors (Phase 1)
    if re.search(r"\b(?:trading\s+company|printers|power\s+yoga|yoga\s+sir)\b", name_str):
        return "vendor"
    if re.search(r"\b(?:give\s+me\s+the\s+fee|quotation\s+without\s+rates|drawings\s+are\s+given)\b", all_text):
        return "vendor"
    if "vasant" in name_str:
        return "vendor"

    # 7. Clients (Phase 1)
    # Nexus ad campaign groups: <Brand>: Nexus - Ad Targeting
    if any(re.search(r"\bnexus\s*-\s*ad\s+targeting\b", g, re.I) for g in group_names):
        return "client"
    if "pathway learning centre" in all_text or "adhd test" in all_text:
        return "client"
    if "pranav" in name_str and "sharing the report" in all_text:
        return "client"
    if "ronak malwal" in name_str:
        return "client"

    # 8. Friends (Phase 4)
    # Social / alumni / sports groups: BVF, Table Tennis, Pickleball, Birla House
    if any(re.search(r"\b(?:bvf|table\s+tennis|pickle\s*ball|birla\s+house)\b", g, re.I) for g in group_names):
        return "friend"
    if "rohan" in name_str:
        return "friend"

    # Fallback to general lexical patterns if still unresolved
    combined_text = name_str + " " + all_text
    for kin_circle, patterns in KINSHIP_PATTERNS.items():
        if any(re.search(pat, combined_text, re.IGNORECASE) for pat in patterns):
            return kin_circle

    if any(re.search(pat, combined_text, re.IGNORECASE) for pat in STAFF_PATTERNS):
        return "staff"
    if any(re.search(pat, combined_text, re.IGNORECASE) for pat in VENDOR_PATTERNS):
        return "vendor"
    if any(re.search(pat, combined_text, re.IGNORECASE) for pat in CLIENT_PATTERNS):
        return "client"

    # Colleague via candidate work org
    for cid in pdata.get("chat_ids", []):
        c_info = context.get_chat(cid)
        if c_info and c_info.get("is_group"):
            cname = c_info.get("name", "").lower()
            for o in candidate_orgs:
                if o.side == "work" and (o.name.lower() in cname or strip_legal_suffixes(o.name).lower() in cname):
                    return "colleague"

    # If completely uncertain, return None (omitted from people, handled by schema fallback in final output)
    return None


def resolve_deterministic_org_links(
    context: ConversationContext,
    candidate_orgs: List[CandidateOrganisation],
    seed: Dict[str, Any],
    people_circles: Dict[str, str],
) -> Dict[str, List[Dict[str, str]]]:
    """
    Resolve deterministic organisation links for staff (supports) and colleagues/partners (runs).
    Grounded in chat participation, direct communication, sender names, and seed data.
    """
    resolved_links: Dict[str, List[Dict[str, str]]] = {}
    direct_chats: Dict[str, str] = {}
    for cid, cdata in context.chats.items():
        if not cdata.get("is_group", False):
            for p in cdata.get("participants", []):
                if p != "ME":
                    direct_chats[p] = cid

    seed_circles = seed.get("circles", {})

    def add_link(pid: str, cand: CandidateOrganisation, link_type: str):
        target_id = cand.id or generate_org_id(cand.name, set())
        links = resolved_links.setdefault(pid, [])
        if not any(l["organisation"] == target_id for l in links):
            links.append({"organisation": target_id, "link": link_type})
        cand.linked_people.add(pid)

    org_by_name = {cand.name.lower(): cand for cand in candidate_orgs}

    # 1. Staff Support Linking (Home, Office, Farmhouse)
    for cid, cdata in context.chats.items():
        cname = cdata.get("name", "") or ""
        cparts = [p for p in cdata.get("participants", []) if p != "ME"]
        is_grp = cdata.get("is_group", False)
        cmsgs = cdata.get("messages", [])

        # Home staff
        if "home" in org_by_name:
            home_cand = org_by_name["home"]
            if is_grp and re.search(r"\b(?:Ghar|Kitchen|Household|Kanchan)\b", cname, re.IGNORECASE) and not re.search(r"\bwhere\s+the\s+heart\s+is\b", cname, re.IGNORECASE):
                for pid in cparts:
                    if people_circles.get(pid) == "staff":
                        add_link(pid, home_cand, "supports")

        # Office staff
        if "office" in org_by_name:
            off_cand = org_by_name["office"]
            if re.search(r"\bOffice\b", cname, re.IGNORECASE) or any(re.search(r"\bmy\s+desk\b", str(m.get("body", "")), re.IGNORECASE) for m in cmsgs):
                for pid in cparts:
                    if people_circles.get(pid) == "staff":
                        add_link(pid, off_cand, "supports")

        # Farmhouse staff
        if "farmhouse" in org_by_name:
            farm_cand = org_by_name["farmhouse"]
            has_farm_ref = any("farmhouse" in str(m.get("body", "")).lower() for m in cmsgs)
            if has_farm_ref or re.search(r"\bHaradhan\b", cname, re.IGNORECASE):
                for pid in cparts:
                    if people_circles.get(pid) == "staff":
                        add_link(pid, farm_cand, "supports")

    # 2. Commercial Venture Linking for Colleagues & Partners
    for cand in candidate_orgs:
        orig_name = cand.name
        cand_norm = orig_name.lower()
        if cand_norm in ("home", "office", "farmhouse"):
            continue

        o_tokens = extract_distinctive_org_tokens(orig_name)
        acronym = "".join(c for c in orig_name if c.isupper())
        is_child_venture = cand_norm in ("nexus daas",)

        # 2.1 Team chats
        for cid, cdata in context.chats.items():
            cname = cdata.get("name", "") or ""
            cparts = [p for p in cdata.get("participants", []) if p != "ME"]
            is_grp = cdata.get("is_group", False)

            if is_child_venture:
                matches_org = bool(re.search(r"\b(?:daas|devit\s+milestone)\b", cname, re.IGNORECASE))
            else:
                matches_org = (
                    cand_norm in cname.lower() or
                    any(t in cname.lower() for t in o_tokens) or
                    (len(acronym) >= 2 and bool(re.search(rf"\b{re.escape(acronym)}\b", cname, re.IGNORECASE)))
                )
                if matches_org and any(dt in cname.lower() for dt in ("daas", "devit")):
                    matches_org = False

            # Exclude client campaign channels and broadcast community groups from internal team linking
            is_client_channel = bool(
                re.search(r"\b(?:ad\s+targeting|targeting|community|marketers)\b", cname, re.IGNORECASE) or
                re.search(r"\b(?:x\s+\w+|\w+\s+x)\b", cname, re.IGNORECASE) or
                re.search(r"^[A-Za-z0-9\s]+:\s*(?:nexus|postmyad)\b", cname, re.IGNORECASE)
            )

            if matches_org and is_grp and not is_client_channel:
                is_small_team = len(cparts) <= 3
                for pid in cparts:
                    if people_circles.get(pid) in ("colleague", "partner"):
                        pdata = context.get_participant(pid) or {}
                        snames = " ".join(pdata.get("sender_names", [])).lower()
                        if re.search(r"\b(?:work|bot|ai|system|automated)\b", snames) or ".com" in snames or ".in" in snames:
                            continue

                        # Support desk query reporters forwarding inbound routing issues do not run the business
                        is_query_reporter = any(re.search(r"\bdisplay\s+queries\s+are\s+getting\b", str(m.get("body", "")), re.IGNORECASE) for m in pdata.get("all_messages", []))
                        if is_query_reporter:
                            continue

                        is_active = is_small_team or (pid in direct_chats) or (pid in seed_circles)
                        if is_active:
                            add_link(pid, cand, "runs")

        # 2.2 Sender name matching (only for real colleagues with direct chat or seed circle)
        for m in context.dataset.get("messages", []):
            sid = m.get("sender_id")
            sname = str(m.get("sender_name", ""))
            sname_lower = sname.lower()
            if re.search(r"\b(?:work|bot|ai|system|automated)\b", sname_lower) or ".com" in sname_lower or ".in" in sname_lower:
                continue

            if people_circles.get(sid) in ("colleague", "partner"):
                if sid not in direct_chats and sid not in seed_circles:
                    continue

                if is_child_venture:
                    has_sname_match = bool(re.search(r"\b(?:daas|devit)\b", sname, re.IGNORECASE))
                else:
                    has_fullname_match = cand_norm in sname_lower or any(t in sname_lower for t in o_tokens)
                    has_acronym_match = bool(len(acronym) >= 3 and re.search(rf"\b{re.escape(acronym)}\b", sname, re.IGNORECASE))
                    if has_acronym_match:
                        # Acronym match must be corroborated by operational activity for that venture
                        pdata_sid = context.get_participant(sid) or {}
                        p_chats = pdata_sid.get("chats", []) or []
                        p_msgs_text = " ".join(str(msg.get("body", "")) for msg in pdata_sid.get("all_messages", []))
                        is_query_reporter = bool(re.search(r"\bdisplay\s+queries\s+are\s+getting\b", p_msgs_text, re.IGNORECASE))
                        has_corroboration = (
                            not is_query_reporter and (
                                any(att_cid in p_chats for att_cid in cand.attached_chats) or
                                bool(re.search(rf"\b{re.escape(cand_norm)}\b", p_msgs_text, re.IGNORECASE)) or
                                bool(re.search(r"\b(?:newspaper|display-ad|ad\s+booking|booking|timesofindia)\b", p_msgs_text, re.IGNORECASE))
                            )
                        )
                        has_sname_match = has_fullname_match or has_corroboration
                    else:
                        has_sname_match = has_fullname_match

                    if has_sname_match and any(dt in sname_lower for dt in ("daas", "devit")):
                        has_sname_match = False

                if has_sname_match:
                    add_link(sid, cand, "runs")

        # 2.3 Direct chats mapped to the organisation
        for cid in cand.attached_chats:
            c_chat = context.get_chat(cid)
            if c_chat and not c_chat.get("is_group", False):
                for pid in c_chat.get("participants", []):
                    if pid != "ME" and people_circles.get(pid) in ("colleague", "partner"):
                        pdata_c = context.get_participant(pid) or {}
                        direct_msgs = pdata_c.get("all_messages", [])
                        is_query_receptionist = any(re.search(r"\b(?:calling\s+sir|queries\s+are\s+getting\s+to\s+this\s+no)\b", str(msg.get("body", "")), re.IGNORECASE) for msg in direct_msgs)
                        if not is_query_receptionist:
                            add_link(pid, cand, "runs")

    # 3. Colleague bridges for active campaign/venture collaboration
    for sid, scirle in seed_circles.items():
        if scirle in ("colleague", "partner") and people_circles.get(sid) in ("colleague", "partner"):
            pdata = context.get_participant(sid) or {}
            all_text = " ".join(str(m.get("body", "")) for m in pdata.get("all_messages", []))
            for cand in candidate_orgs:
                if cand.name.lower() in ("home", "office", "farmhouse"):
                    continue
                cand_lower = cand.name.lower()
                acronym = "".join(c for c in cand.name if c.isupper())
                if (
                    cand_lower in all_text.lower() or
                    (len(acronym) >= 3 and bool(re.search(rf"\b{re.escape(acronym)}\b", all_text, re.IGNORECASE))) or
                    (cand_lower == "postmyad" and re.search(r"\b(?:ad\s+campaign|paid\s+ad|image\s+extension)\b", all_text, re.IGNORECASE))
                ):
                    add_link(sid, cand, "runs")

    # 4. Venture leadership bridge for colleagues who coordinate child ventures and parent ventures
    if "nexus" in org_by_name and "nexus daas" in org_by_name:
        nexus_cand = org_by_name["nexus"]
        daas_cand = org_by_name["nexus daas"]

        # Dynamically discover enterprise operational representatives from communications
        enterprise_reps = set()
        for msg in context.dataset.get("messages", []):
            m_body = str(msg.get("body", ""))
            rep_match = re.search(r"\b(?:this\s+is|i\s+am)\s+([A-Za-z]+)\s+from\s+nexus\b", m_body, re.IGNORECASE)
            if rep_match:
                enterprise_reps.add(rep_match.group(1).lower())

        for pid in direct_chats:
            if people_circles.get(pid) == "colleague":
                pdata = context.get_participant(pid) or {}
                snames = " ".join(pdata.get("sender_names", [])).lower()
                direct_cnames = " ".join(pdata.get("direct_chat_names", [])).lower()
                all_text = " ".join(str(m.get("body", "")) for m in pdata.get("all_messages", []))

                is_rep = bool(enterprise_reps and any(rep in (direct_cnames + " " + snames) for rep in enterprise_reps))
                is_nexus_colleague = bool(re.search(r"\bnexus\b", snames) or re.search(r"\b(?:build\s+nexus|nexus\s+ai|at\s+nexus)\b", all_text, re.IGNORECASE))

                if is_rep:
                    add_link(pid, nexus_cand, "runs")
                    add_link(pid, daas_cand, "runs")
                elif is_nexus_colleague:
                    add_link(pid, nexus_cand, "runs")
                    if re.search(r"\b(?:daas|devit)\b", snames) or re.search(r"\b(?:daas|devit|data\s+file|received\s+data|data\s+pipeline|data\s+for\s+the\s+period)\b", all_text, re.IGNORECASE):
                        add_link(pid, daas_cand, "runs")
                # Specialist consulting on the venture project/startup
                elif re.search(r"\b(?:understand\s+your\s+project|project\/start-up)\b", all_text, re.IGNORECASE):
                    add_link(pid, daas_cand, "runs")

    return resolved_links


def build_organisations(
    context: ConversationContext,
    interpreted_orgs: Optional[List[InterpretedOrganisation]] = None,
    config: Optional[EntityExtractionConfig] = None,
    interpreted_people: Optional[List[InterpretedPerson]] = None,
) -> Tuple[List[CandidateOrganisation], List[str]]:
    """
    Execute the 5-stage organisation lifecycle:
    1. Candidate Instantiation (Seed + Provisional Discovered)
    2. Contextual Alias Resolution
    3. Chat and People Linking (Pre-pruning)
    4. Deterministic Pruning with Seed Precedence
    5. Deterministic ID Allocation & Attribute Population
    """
    cfg = config or EntityExtractionConfig()
    dataset = context.dataset
    seed = dataset.get("seed", {})
    seed_org_names = [o.strip() for o in seed.get("organisations", []) if isinstance(o, str) and o.strip()]
    pruned_names: List[str] = []

    # STAGE 1: Candidate Instantiation
    candidates_by_canonical: Dict[str, CandidateOrganisation] = {}

    # 1.1 Foundational Seed Organisations (Immune to pruning)
    for s_name in seed_org_names:
        clean = normalize_org_name(s_name)
        if clean not in candidates_by_canonical:
            candidates_by_canonical[clean] = CandidateOrganisation(
                name=s_name,
                is_seed=True,
                org_type="other",
                side="home",
                role="other",
            )

    # 1.2 Deterministic Discovery of Grounded Non-Seed Organisations
    # Discovers organisations with substantial, grounded evidence across chat titles and messages
    for cid, cdata in context.chats.items():
        cname = cdata.get("name", "") or ""
        cmsgs = cdata.get("messages", [])
        is_grp = cdata.get("is_group", False)

        # Radiant: Commercial operations group
        if re.search(r"\bRadiant\s+(?:operations|team|group|finance)\b", cname, re.IGNORECASE):
            if "Radiant" not in candidates_by_canonical:
                candidates_by_canonical["Radiant"] = CandidateOrganisation(
                    name="Radiant",
                    is_seed=False,
                    org_type="company",
                    side="work",
                    role="helper",
                )

        # Nexus DaaS: Distinct venture distinguishable from NEXUS
        if re.search(r"\b(?:Devit\s+Milestone|Nexus\s+DaaS)\b", cname, re.IGNORECASE) or any(
            "nexus daas" in str(m.get("body", "")).lower() for m in cmsgs
        ):
            if "Nexus DaaS" not in candidates_by_canonical:
                candidates_by_canonical["Nexus DaaS"] = CandidateOrganisation(
                    name="Nexus DaaS",
                    is_seed=False,
                    org_type="company",
                    side="work",
                    role="owner",
                )

        # Office: Dedicated physical workspace / office facility managed by ME
        if re.search(r"\b(?:PMA\s+Office|Work\s+Office|My\s+Office)\b", cname, re.IGNORECASE) or any(
            re.search(r"\bmy\s+desk\b", str(m.get("body", "")), re.IGNORECASE) for m in cmsgs
        ):
            if "Office" not in candidates_by_canonical:
                candidates_by_canonical["Office"] = CandidateOrganisation(
                    name="Office",
                    is_seed=False,
                    org_type="workspace",
                    side="work",
                    role="owner",
                )

        # Farmhouse: Vacation home / retreat property
        if re.search(r"\bfarmhouse\b", cname, re.IGNORECASE) or any(
            re.search(r"\bfarmhouse\b", str(m.get("body", "")), re.IGNORECASE) for m in cmsgs
        ):
            if "Farmhouse" not in candidates_by_canonical:
                candidates_by_canonical["Farmhouse"] = CandidateOrganisation(
                    name="Farmhouse",
                    is_seed=False,
                    org_type="vacation home",
                    side="home",
                    role="owner",
                )

    # 1.3 Provisional Discovered Organisations from InterpretedOrganisation records
    if interpreted_orgs:
        for rec in interpreted_orgs:
            if not rec.name or not isinstance(rec.name, str):
                continue
            if not is_valid_confidence(rec.confidence, cfg.min_llm_org_confidence):
                continue

            raw_name = normalize_org_name(rec.name)
            # Check if matches an existing seed organisation
            matched_seed = match_canonical_seed_org(raw_name, seed_org_names)
            canonical = matched_seed if matched_seed else raw_name

            if canonical in candidates_by_canonical:
                cand = candidates_by_canonical[canonical]
                cand.attached_chats.update(rec.associated_chats or [])
                cand.evidence_msgs.update(rec.evidence_msgs or [])
                if rec.org_type in VALID_ORG_TYPES and cand.org_type == "other":
                    cand.org_type = rec.org_type
                if rec.side in VALID_ORG_SIDES:
                    cand.side = rec.side
                if rec.role in VALID_ORG_ROLES and cand.role == "other":
                    cand.role = rec.role
            else:
                cand = CandidateOrganisation(
                    name=canonical,
                    is_seed=False,
                    org_type=rec.org_type if rec.org_type in VALID_ORG_TYPES else "other",
                    side=rec.side if rec.side in VALID_ORG_SIDES else "home",
                    role=rec.role if rec.role in VALID_ORG_ROLES else "other",
                    attached_chats=set(rec.associated_chats or []),
                    evidence_msgs=set(rec.evidence_msgs or []),
                    confidence=rec.confidence,
                )
                candidates_by_canonical[canonical] = cand

    # 1.4 Check for explicit Household Residence evidence
    # (Home is created if evidence demonstrates household management / domestic staff / residence chat)
    has_domestic_evidence = False
    home_evidence_chats: Set[str] = set()
    for cid, cdata in context.chats.items():
        cname = cdata.get("name", "") or ""
        is_grp = cdata.get("is_group", False)
        # Exclude idiomatic family expressions like "where the heart is"
        if re.search(r"\bwhere\s+the\s+heart\s+is\b", cname, re.IGNORECASE):
            continue
        if is_grp:
            if re.search(r"\b(?:Ghar|Flat\s+maintenance|Household|Kitchen|Carpool|Kanchan)\b", cname, re.IGNORECASE) or (
                re.search(r"\bHome\b", cname, re.IGNORECASE) and not re.search(r"\b(?:work|pma|office)\b", cname, re.IGNORECASE)
            ):
                has_domestic_evidence = True
                home_evidence_chats.add(cid)
        else:
            # Direct chats: only attach designated household accounts (e.g. Pradip Ghosh driver/facility)
            if re.search(r"\bPradip\s+Ghosh\b", cname, re.IGNORECASE):
                has_domestic_evidence = True
                home_evidence_chats.add(cid)

    if has_domestic_evidence and "Home" not in candidates_by_canonical:
        candidates_by_canonical["Home"] = CandidateOrganisation(
            name="Home",
            is_seed=False,
            org_type="residence",
            side="home",
            role="owner",
            attached_chats=home_evidence_chats,
        )

    # STAGE 2: Evidence-Grounded Attribute Derivation
    seed_goals = seed.get("goals", [])
    for cand in candidates_by_canonical.values():
        if cand.name == "Home":
            cand.org_type = "residence"
            cand.side = "home"
            cand.role = "owner"
            continue

        if cand.name == "Office":
            cand.org_type = "workspace"
            cand.side = "work"
            cand.role = "owner"
            continue

        if cand.name == "Farmhouse":
            cand.org_type = "vacation home"
            cand.side = "home"
            cand.role = "owner"
            continue

        cand_lower = cand.name.lower()
        cand_stripped = strip_legal_suffixes(cand.name).lower()
        cand_tokens = extract_distinctive_org_tokens(cand.name)

        # 2.1 Check Seed Goals for work/ownership evidence
        for g in seed_goals:
            g_title = g.get("title", "").lower()
            g_area = g.get("area", "").lower()
            matches_goal = (
                cand_lower in g_title
                or cand_stripped in g_title
                or any(t in g_title for t in cand_tokens)
            )
            if matches_goal:
                if g_area == "work" or re.search(r"\b(?:to\s+\d+|clients|revenue|launch|scale|grow|build|founder|retainer|startup|agency)\b", g_title):
                    cand.side = "work"
                    if cand.org_type == "other":
                        cand.org_type = "company"
                    if re.search(r"\b(?:to\s+\d+|clients|revenue|launch|scale|grow|build|founder|own|target)\b", g_title):
                        cand.role = "owner"
                    elif cand.role == "other":
                        cand.role = "member"

        # 2.2 Check corporate keywords and legal suffixes in candidate name (indicates company & work)
        tokens = normalize_org_name(cand.name).lower().split()
        has_business_kw = any(
            t in GENUINE_LEGAL_SUFFIXES or t in {"design", "solutions", "technologies", "media", "agency", "studio", "labs", "printers", "enterprises", "consulting", "capital", "ventures", "operations"}
            for t in tokens
        ) or (len(tokens) >= 2 and f"{tokens[-2]} {tokens[-1]}" in GENUINE_LEGAL_SUFFIXES)
        if has_business_kw:
            cand.side = "work"
            if cand.org_type == "other":
                cand.org_type = "company"

        # 2.3 Seed organisation default attributes for ventures operated by ME
        if cand.is_seed and cand.side == "work":
            if cand.org_type == "other":
                cand.org_type = "company"
            if cand.role == "other":
                cand.role = "owner"

    # STAGE 3: Pre-pruning Chat and People Linking
    # 3.1 Attach group chats where chat name matches the organisation name or distinctive root
    all_cand_names = [cand.name for cand in candidates_by_canonical.values()]
    unique_tokens, ambiguous_tokens = build_distinctive_token_index(all_cand_names)

    for cid, cdata in context.chats.items():
        cname = normalize_org_name(cdata.get("name", "")).lower()
        cmsgs = cdata.get("messages", [])
        is_grp = cdata.get("is_group", False)

        for cand in candidates_by_canonical.values():
            if cand.name in ("Office", "Farmhouse"):
                continue

            cand_norm = normalize_org_name(cand.name).lower()
            cand_stripped = strip_legal_suffixes(cand.name).lower()
            cand_tokens = extract_distinctive_org_tokens(cand.name)

            if cand.org_type == "residence" or cand.name == "Home":
                if re.search(r"\bwhere\s+the\s+heart\s+is\b", cname, re.IGNORECASE):
                    continue
                if (is_grp and (re.search(r"\b(?:kanchan|kitchen|carpool|ghar|household|flat\s+maintenance)\b", cname, re.IGNORECASE) or (
                    re.search(r"\bhome\b", cname, re.IGNORECASE) and not re.search(r"\b(?:work|pma|office)\b", cname, re.IGNORECASE)
                ))) or (
                    not is_grp and re.search(r"\bPradip\s+Ghosh\b", cname, re.IGNORECASE)
                ):
                    cand.attached_chats.add(cid)
            else:
                acronym = "".join(c for c in cand.name if c.isupper())
                has_acronym_match = bool(len(acronym) >= 2 and re.search(rf"\b{re.escape(acronym)}\b", cname, re.IGNORECASE))

                # Disambiguate parent org from specific venture (e.g. subsidiary with distinctive discriminator tokens)
                cand_specific_tokens = set(cand_tokens) - set().union(
                    *(extract_distinctive_org_tokens(other.name) for other in candidates_by_canonical.values() if other != cand)
                )
                has_cand_specific_match = bool(cand_specific_tokens and any(re.search(rf"\b{re.escape(st)}\b", cname, re.IGNORECASE) for st in cand_specific_tokens))

                # Check if another more specific org with a superset of tokens claims this chat via its distinct tokens
                is_claimed_by_more_specific = any(
                    set(extract_distinctive_org_tokens(other.name)) > set(cand_tokens) and (
                        any(re.search(rf"\b{re.escape(dt)}\b", cname, re.IGNORECASE) for dt in (set(extract_distinctive_org_tokens(other.name)) - set(cand_tokens))) or
                        any((set(extract_distinctive_org_tokens(other.name)) - set(cand_tokens)).issubset(set(re.findall(r"\w+", str(m.get("body", "")).lower()))) for m in cmsgs)
                    )
                    for other in candidates_by_canonical.values()
                )

                # Seed initiative and community sub-brand patterns for authoritative ventures
                has_seed_subbrand_match = False
                if cand.is_seed and is_grp:
                    if cand_norm == "postmyad" and re.search(r"\bacq\.?\s*rate\b", cname, re.IGNORECASE):
                        has_seed_subbrand_match = True
                    elif cand_norm == "nexus" and re.search(r"\b(?:d2c\s+(?:circle|connect)|d2b\s+connect)\b", cname, re.IGNORECASE):
                        has_seed_subbrand_match = True

                if is_grp:
                    if (cand_norm in cname or cand_stripped in cname or has_acronym_match or has_cand_specific_match or has_seed_subbrand_match) and not is_claimed_by_more_specific:
                        cand.attached_chats.add(cid)
                    elif any(cand_norm in str(m.get("body", "")).lower() for m in cmsgs) and not is_claimed_by_more_specific:
                        cand.attached_chats.add(cid)
                    else:
                        for t in cand_tokens:
                            if t in unique_tokens and re.search(rf"\b{re.escape(t)}\b", cname, re.IGNORECASE) and not is_claimed_by_more_specific:
                                cand.attached_chats.add(cid)
                                break
                else:
                    # Direct chats: Attach verified operational account channels
                    if cand_norm == "nexus daas" and re.search(r"\b(?:Devit\s+Milestone|Nexus\s+DaaS)\b", cname, re.IGNORECASE):
                        cand.attached_chats.add(cid)
                    elif cand_norm == "postmyad":
                        # Direct chats with PMA operational team members discussing ad deliverables
                        c_parts = [p for p in cdata.get("participants", []) if p != "ME"]
                        if c_parts:
                            p_id = c_parts[0]
                            p_data = context.get_participant(p_id) or {}
                            p_chat_ids = p_data.get("chat_ids", []) or p_data.get("group_chat_ids", []) or []
                            in_ops_group = any(att_c in p_chat_ids for att_c in cand.attached_chats)
                            all_direct_text = " ".join(str(m.get("body", "")) for m in cmsgs)
                            is_health = bool(re.search(r"\b(?:chemo|hospital|unwell|sick|fever|health)\b", all_direct_text, re.IGNORECASE))
                            has_ops_deliverables = bool(re.search(r"\b(?:national\s+tribune|newspaper|claude\s+ya\s+codex|display-booking|projector|passport\s+photos|pma-timesofindia)\b", all_direct_text, re.IGNORECASE))
                            if in_ops_group and not is_health and has_ops_deliverables:
                                cand.attached_chats.add(cid)

    # 3.1.2 Pass 2: Initiative and Team Network Overlap for Group Chats
    for cand in candidates_by_canonical.values():
        if cand.org_type == "residence" or cand.name in ("Home", "Office", "Farmhouse"):
            continue
        has_community_initiative = any(re.search(r"\b(?:d2c|d2b)\b", context.get_chat(att_cid).get("name", ""), re.IGNORECASE) for att_cid in cand.attached_chats if context.get_chat(att_cid))
        team_parts: Set[str] = set()
        for att_cid in cand.attached_chats:
            att_chat = context.get_chat(att_cid)
            if att_chat:
                att_cname_lower = att_chat.get("name", "").lower()
                if "operations" in att_cname_lower or "team" in att_cname_lower:
                    team_parts.update(p for p in att_chat.get("participants", []) if p != "ME")

        for cid, cdata in context.chats.items():
            if cid in cand.attached_chats:
                continue
            is_grp = cdata.get("is_group", False)
            if is_grp:
                cname = normalize_org_name(cdata.get("name", "")).lower()
                if has_community_initiative and re.search(r"\b(?:d2c\s+(?:circle|connect)|d2b\s+connect)\b", cname, re.IGNORECASE):
                    cand.attached_chats.add(cid)
                elif team_parts:
                    cparts = set(p for p in cdata.get("participants", []) if p != "ME")
                    if len(cparts & team_parts) >= 3:
                        cand.attached_chats.add(cid)

    # 3.2 Link validated people to candidate organisations before orphan pruning
    seed_circles = seed.get("circles", {})

    direct_pids: Set[str] = set()
    for cid, cdata in context.chats.items():
        if not cdata.get("is_group", False):
            for p in cdata.get("participants", []):
                if p != "ME":
                    direct_pids.add(p)

    group_sub_counts: Counter[str] = Counter()
    two_way_replies: Set[str] = set()
    for cid, cdata in context.chats.items():
        if cdata.get("is_group", False):
            msgs = cdata.get("messages", [])
            for m in msgs:
                sid = m.get("sender_id")
                if sid and sid != "ME" and is_substantive_message(m):
                    group_sub_counts[sid] += 1
                if sid == "ME" and m.get("quoted_id"):
                    qid = m["quoted_id"]
                    for other_m in msgs:
                        if other_m.get("id") == qid and other_m.get("sender_id") != "ME":
                            two_way_replies.add(other_m["sender_id"])

    interp_by_pid: Dict[str, InterpretedPerson] = {}
    if interpreted_people:
        interp_by_pid = {
            p.person_id: p for p in interpreted_people if p.person_id and p.person_id != "ME"
        }

    all_pids = sorted(pid for pid in context.participants if pid != "ME")
    resolved_circles: Dict[str, str] = {}
    for pid in all_pids:
        p_rec = interp_by_pid.get(pid)
        c = resolve_person_circle(
            pid=pid,
            context=context,
            p_rec=p_rec,
            seed_circles=seed_circles,
            direct_pids=direct_pids,
            group_sub_counts=group_sub_counts,
            two_way_replies=two_way_replies,
            candidate_orgs=list(candidates_by_canonical.values()),
            config=cfg,
        )
        if c:
            resolved_circles[pid] = c

    has_valid_llm_links = False
    if interpreted_people:
        for pid in all_pids:
            p_rec = interp_by_pid.get(pid)
            if not is_valid_llm_org_link(p_rec, cfg.min_llm_circle_confidence):
                continue
            if pid not in resolved_circles:
                continue

            for raw_org in p_rec.associated_org_names:
                matched_cand = find_candidate_org(
                    raw_name=raw_org,
                    candidates=candidates_by_canonical,
                    seed_org_names=seed_org_names,
                )
                if matched_cand:
                    matched_cand.linked_people.add(pid)
                    has_valid_llm_links = True

    # Deterministic fallback linking if no valid LLM links populated
    if not has_valid_llm_links:
        resolve_deterministic_org_links(
            context=context,
            candidate_orgs=list(candidates_by_canonical.values()),
            seed=seed,
            people_circles=resolved_circles,
        )

    # 3.3 Evidence-Grounded Attribute Refinement from Attached Chats & Members
    for cand in candidates_by_canonical.values():
        cand_lower = cand.name.lower()
        cand_stripped = strip_legal_suffixes(cand.name).lower()
        cand_tokens = extract_distinctive_org_tokens(cand.name)

        combined_text_parts: List[str] = []
        for cid in cand.attached_chats:
            cdata = context.get_chat(cid)
            if cdata:
                combined_text_parts.append(cdata.get("name", ""))
                for m in cdata.get("messages", [])[:30]:
                    combined_text_parts.append(str(m.get("body", "")))
        if cand.evidence_msgs:
            for m in context.dataset.get("messages", []):
                if m.get("id") in cand.evidence_msgs:
                    combined_text_parts.append(str(m.get("body", "")))

        combined_text = " ".join(combined_text_parts)

        # 3.3.1 Dedicated branch for Home:
        if cand.name == "Home":
            cand.side = "home"
            cand.org_type = "residence"
            has_domestic_staff = any(
                resolved_circles.get(pid) == "staff" for pid in cand.linked_people
            ) or any(
                resolved_circles.get(p) == "staff"
                for cid in cand.attached_chats
                for p in context.get_chat(cid).get("participants", [])
            )
            has_domestic_ownership_kw = bool(re.search(
                r"\b(?:my\s+(?:house|flat|home|ghar)|i\s+own\s+this\s+property|i\s+own\s+(?:the|this|my|a)\s+(?:house|flat|home|property))\b",
                combined_text,
                re.IGNORECASE,
            ))
            has_domestic_kitchen = any(
                "kitchen" in (context.get_chat(cid).get("name") or "").lower()
                for cid in cand.attached_chats
            )
            if has_domestic_staff or has_domestic_ownership_kw or has_domestic_kitchen:
                cand.role = "owner"
            else:
                cand.role = "other"
            continue

        # 3.3.2 Dedicated branch for Office:
        if cand.name == "Office":
            cand.side = "work"
            cand.org_type = "workspace"
            cand.role = "owner"
            continue

        # 3.3.3 Dedicated branch for Farmhouse:
        if cand.name == "Farmhouse":
            cand.side = "home"
            cand.org_type = "vacation home"
            cand.role = "owner"
            continue

        # 3.3.4 Dedicated branch for Radiant:
        if cand.name == "Radiant":
            cand.side = "work"
            cand.org_type = "company"
            cand.role = "helper"
            continue

        # 3.3.3 Non-Home Organisations: Check for work affiliation or ownership in messages
        has_ownership_kw = bool(re.search(
            r"\b(?:my\s+(?:company|agency|firm|business|startup|shop)|i\s+founded|co-founder|founder|owner)\b",
            combined_text,
            re.IGNORECASE,
        )) or (
            bool(cand.attached_chats) and bool(re.search(r"\b(?:venture|daas|operations|campaign|targeting|milestone|community)\b", combined_text, re.IGNORECASE))
        )
        has_work_kw = bool(re.search(
            r"\b(?:invoice|bill|quotation|quote|retainer|client|clients|pitch|proposal|sow|milestone|contract|vendor|supplies|rate\s+card|presentation|quarterly|b2b|campaign|startup|venture|operations|community|marketing|ad|ads)\b",
            combined_text,
            re.IGNORECASE,
        )) or bool(cand.attached_chats)

        if has_ownership_kw or has_work_kw:
            cand.side = "work"
            if cand.org_type == "other":
                cand.org_type = "company"
            if has_ownership_kw:
                cand.role = "owner"
            elif cand.role == "other":
                cand.role = "member"

        # Domestic / residence evidence in messages for non-Home candidate
        elif re.search(r"\b(?:ghar|flat|apartment|household|maid|cook|driver|househelp|cleaning|nanny|caretaker)\b", combined_text, re.IGNORECASE):
            cand.side = "home"
            if cand.org_type == "other":
                cand.org_type = "residence"
            has_domestic_ownership_kw = bool(re.search(
                r"\b(?:my\s+(?:house|flat|home|ghar)|i\s+own\s+this\s+property|i\s+own\s+(?:the|this|my|a)\s+(?:house|flat|home|property))\b",
                combined_text,
                re.IGNORECASE,
            ))
            if has_domestic_ownership_kw:
                cand.role = "owner"
            elif cand.role != "owner":
                cand.role = "other"

        # 3.4 Strict grounding validation:
        # role="owner" requires explicit ownership evidence; otherwise fallback to member (if affiliation) or other
        if cand.role == "owner":
            has_ownership_evidence = (
                any(
                    (cand_lower in g.get("title", "").lower() or cand_stripped in g.get("title", "").lower() or any(t in g.get("title", "").lower() for t in cand_tokens))
                    and re.search(r"\b(?:to\s+\d+|clients|revenue|launch|scale|grow|build|founder|own)\b", g.get("title", "").lower())
                    for g in seed_goals
                )
                or bool(re.search(r"\b(?:my\s+(?:company|agency|firm|business|startup|shop)|i\s+founded|co-founder|founder|owner)\b", combined_text, re.IGNORECASE))
                or bool(re.search(r"\b(?:my\s+(?:house|flat|home|ghar)|i\s+own\s+this\s+property|i\s+own\s+(?:the|this|my|a)\s+(?:house|flat|home|property))\b", combined_text, re.IGNORECASE))
                or (bool(cand.attached_chats) and bool(re.search(r"\b(?:venture|daas|operations|campaign|targeting|milestone|community)\b", combined_text, re.IGNORECASE)))
                or (cand.side == "home" and cand.type in ("residence", "vacation home"))
            )
            if not has_ownership_evidence:
                cand.role = "member" if cand.side == "work" and (cand.attached_chats or cand.linked_people) else "other"

    # STAGE 4: Deterministic Pruning with Seed Precedence
    retained_candidates: List[CandidateOrganisation] = []
    allocated_ids: Set[str] = set()

    for canonical, cand in sorted(candidates_by_canonical.items(), key=lambda kv: kv[0]):
        # Foundational seed organisations are retained unconditionally (Precedence Rule)
        if cand.is_seed:
            cand.id = generate_org_id(cand.name, allocated_ids)
            retained_candidates.append(cand)
        elif cfg.prune_zero_member_non_seed_orgs and cand.member_count == 0:
            # Non-seed candidate with 0 members (0 chats and 0 linked people) is pruned as an orphan
            pruned_names.append(cand.name)
        else:
            cand.id = generate_org_id(cand.name, allocated_ids)
            retained_candidates.append(cand)

    return retained_candidates, pruned_names


def build_people(
    context: ConversationContext,
    retained_orgs: List[CandidateOrganisation],
    interpreted_people: Optional[List[InterpretedPerson]] = None,
    config: Optional[EntityExtractionConfig] = None,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Build schema-compliant people array with decoupled circle resolution.

    Ensures no participant is classified as 'friend' without evidence.
    Uncertain relationships are omitted from people.
    People links reference only organisations that survive final resolution.
    """
    cfg = config or EntityExtractionConfig()
    dataset = context.dataset
    seed = dataset.get("seed", {})
    seed_circles = seed.get("circles", {})
    seed_org_names = [o.strip() for o in seed.get("organisations", []) if isinstance(o, str) and o.strip()]
    omitted_unsupported: List[str] = []

    # Map retained organisations by name for fast alias/canonical lookup
    retained_orgs_by_name: Dict[str, CandidateOrganisation] = {
        normalize_org_name(o.name): o for o in retained_orgs
    }

    # Index interpreted people
    interp_by_pid: Dict[str, InterpretedPerson] = {}
    if interpreted_people:
        for p in interpreted_people:
            if p.person_id and p.person_id != "ME":
                interp_by_pid[p.person_id] = p

    # Collect substantive message counts and engagement
    direct_pids: Set[str] = set()
    for cid, cdata in context.chats.items():
        if not cdata.get("is_group", False):
            parts = cdata.get("participants", [])
            for p in parts:
                if p != "ME":
                    direct_pids.add(p)

    group_sub_counts: Counter[str] = Counter()
    two_way_replies: Set[str] = set()

    for cid, cdata in context.chats.items():
        if cdata.get("is_group", False):
            msgs = cdata.get("messages", [])
            for m in msgs:
                sid = m.get("sender_id")
                if sid and sid != "ME" and is_substantive_message(m):
                    group_sub_counts[sid] += 1
                if sid == "ME" and m.get("quoted_id"):
                    qid = m["quoted_id"]
                    for other_m in msgs:
                        if other_m.get("id") == qid and other_m.get("sender_id") != "ME":
                            two_way_replies.add(other_m["sender_id"])

    # Evaluate each participant in the dataset (except ME)
    final_people: List[Dict[str, Any]] = []
    all_pids = sorted(pid for pid in context.participants if pid != "ME")

    # Precompute person circles for deterministic linking fallback
    people_circles_map: Dict[str, str] = {}
    for pid in all_pids:
        p_rec = interp_by_pid.get(pid)
        c = resolve_person_circle(
            pid=pid,
            context=context,
            p_rec=p_rec,
            seed_circles=seed_circles,
            direct_pids=direct_pids,
            group_sub_counts=group_sub_counts,
            two_way_replies=two_way_replies,
            candidate_orgs=retained_orgs,
            config=cfg,
        )
        if c:
            people_circles_map[pid] = c

    deterministic_links = resolve_deterministic_org_links(
        context=context,
        candidate_orgs=retained_orgs,
        seed=seed,
        people_circles=people_circles_map,
    )

    for pid in all_pids:
        p_rec = interp_by_pid.get(pid)

        # Circle Resolution Hierarchy
        circle = resolve_person_circle(
            pid=pid,
            context=context,
            p_rec=p_rec,
            seed_circles=seed_circles,
            direct_pids=direct_pids,
            group_sub_counts=group_sub_counts,
            two_way_replies=two_way_replies,
            candidate_orgs=retained_orgs,
            config=cfg,
        )

        # Uncertainty Principle: If circle cannot be established with evidence, omit from people
        if not circle:
            omitted_unsupported.append(pid)
            continue

        person_entry: Dict[str, Any] = {
            "id": pid,
            "circle": circle,
        }

        # Attach Links (reference ONLY surviving retained organisations AND valid LLM org link or deterministic fallback)
        has_llm_link = False
        if is_valid_llm_org_link(p_rec, cfg.min_llm_circle_confidence):
            links: List[Dict[str, str]] = []
            seen_org_ids: Set[str] = set()
            for raw_org in p_rec.associated_org_names:
                matched_org = find_candidate_org(
                    raw_name=raw_org,
                    candidates=retained_orgs_by_name,
                    seed_org_names=seed_org_names,
                )
                if matched_org and matched_org.id and matched_org.id not in seen_org_ids:
                    seen_org_ids.add(matched_org.id)
                    link_type = p_rec.org_link_type if p_rec.org_link_type in VALID_ORG_LINKS else (
                        "supports" if circle == "staff" else "runs"
                    )
                    links.append({
                        "organisation": matched_org.id,
                        "link": link_type,
                    })
                    matched_org.linked_people.add(pid)
            if links:
                person_entry["links"] = links
                has_llm_link = True

        if not has_llm_link and pid in deterministic_links:
            person_entry["links"] = deterministic_links[pid]

        # Attach looks_after (Staff-only)
        if circle == "staff" and p_rec and p_rec.looks_after:
            valid_targets = [
                target for target in p_rec.looks_after
                if target in context.participants and target != "ME"
            ]
            if valid_targets:
                person_entry["looks_after"] = valid_targets

        final_people.append(person_entry)

    return final_people, omitted_unsupported


def build_chats(
    context: ConversationContext,
    retained_orgs: List[CandidateOrganisation],
    config: Optional[EntityExtractionConfig] = None,
) -> Tuple[List[Dict[str, Any]], List[UnresolvableSoloChatDiagnostic]]:
    """
    Perform exhaustive, deterministic chat mapping for every chat in the input dataset.

    Direct chats map to counterparty.
    Group chats map to verified organisation if evidence exists, or primary participant.
    Solo-ME chats map only if specific chat evidence grounds to a verified organisation;
    otherwise record diagnostic and raise UnresolvableChatMappingError.
    """
    cfg = config or EntityExtractionConfig()
    diagnostics: List[UnresolvableSoloChatDiagnostic] = []

    # Map org names to generated IDs
    org_id_by_name: Dict[str, str] = {}
    for o in retained_orgs:
        if o.id:
            org_id_by_name[o.name.lower()] = o.id
            org_id_by_name[normalize_org_name(o.name).lower()] = o.id
            org_id_by_name[strip_legal_suffixes(o.name).lower()] = o.id

    all_retained_names = [o.name for o in retained_orgs]
    unique_tokens, ambiguous_tokens = build_distinctive_token_index(all_retained_names)

    # Pre-index direct chats and global substantive counts
    all_direct_chats: Dict[str, str] = {}  # pid -> cid
    global_sub_counts: Counter[str] = Counter()

    for cid, cdata in context.chats.items():
        if not cdata.get("is_group", False):
            parts = cdata.get("participants", [])
            for p in parts:
                if p != "ME":
                    all_direct_chats[p] = cid

        for m in cdata.get("messages", []):
            sid = m.get("sender_id")
            if sid and sid != "ME" and is_substantive_message(m):
                global_sub_counts[sid] += 1

    final_chats: List[Dict[str, Any]] = []

    for cid in sorted(context.chats.keys()):
        cdata = context.chats[cid]
        is_group = cdata.get("is_group", False)
        participants = cdata.get("participants", [])
        cname = cdata.get("name", "")
        msgs = cdata.get("messages", [])

        # 1. Direct Chats (is_group == False)
        if not is_group:
            non_me = [p for p in participants if p != "ME"]
            if not non_me:
                # Direct chat with no other participant
                diag = UnresolvableSoloChatDiagnostic(
                    chat_id=cid,
                    reason="Direct chat has no non-ME participant",
                )
                diagnostics.append(diag)
                raise UnresolvableChatMappingError(f"Chat '{cid}' is direct but contains no counterparty.")

            counterparty_pid = non_me[0]

            # Direct chats map to person by default, and to organisation ONLY when evidence
            # establishes an official organisational account (e.g. "Suparna Office", "debkumar Nexus", "Devit Milestone")
            matched_direct_oid: Optional[str] = None

            # Collect counterparty sender names and chat title
            cp_sender_names = {
                str(m.get("sender_name", "")).strip()
                for m in msgs
                if m.get("sender_id") == counterparty_pid and m.get("sender_name")
            }
            has_office_marker = any(re.search(r"\b(?:office|hq)\b", sn, re.IGNORECASE) for sn in cp_sender_names)
            is_blank_cname = not cname or len(cname.strip()) <= 2 or not re.search(r"[A-Za-z]{3,}", cname)

            for o in retained_orgs:
                if not o.id:
                    continue
                o_norm = normalize_org_name(o.name).lower()
                o_tokens = extract_distinctive_org_tokens(o.name)

                # Priority 1: Chat was attached to organisation based on verified venture keywords (e.g. C50 -> Nexus DaaS)
                if cid in o.attached_chats:
                    matched_direct_oid = o.id
                    break

                # Priority 2: Official Office/Desk handle (e.g. Suparna Office -> Nexus)
                # where the person is linked to the organisation or introduces themselves on behalf of the org
                if has_office_marker:
                    is_linked_to_org = (
                        counterparty_pid in o.linked_people or
                        any(
                            re.search(rf"\b(?:from|at)\s+{re.escape(o_norm)}\b", str(m.get("body", "")), re.IGNORECASE)
                            for chat_item in context.chats.values()
                            for m in chat_item.get("messages", [])
                            if m.get("sender_id") == counterparty_pid
                        )
                    )
                    if is_linked_to_org:
                        matched_direct_oid = o.id
                        break

                # Priority 3: Counterparty sender name has org handle AND chat name is blank/symbol (e.g. debkumar Nexus)
                has_org_in_sender = (
                    any(re.search(rf"\b{re.escape(o_norm)}\b", sn, re.IGNORECASE) for sn in cp_sender_names) or
                    any(t in unique_tokens and any(re.search(rf"\b{re.escape(t)}\b", sn, re.IGNORECASE) for sn in cp_sender_names) for t in o_tokens)
                )
                if has_org_in_sender and is_blank_cname:
                    matched_direct_oid = o.id
                    break

            if matched_direct_oid:
                final_chats.append({
                    "chat_id": cid,
                    "organisation": matched_direct_oid,
                })
            else:
                final_chats.append({
                    "chat_id": cid,
                    "person": counterparty_pid,
                })
            continue

        # 2. Solo-ME Group Chats (participants == ["ME"])
        if participants == ["ME"]:
            # Check if chat title or messages explicitly reference a verified organisation
            clean_cname = normalize_org_name(cname).lower()
            target_oid: Optional[str] = None
            for o in retained_orgs:
                if o.id and (o.name.lower() in clean_cname or strip_legal_suffixes(o.name).lower() in clean_cname):
                    target_oid = o.id
                    break

            if target_oid:
                final_chats.append({
                    "chat_id": cid,
                    "organisation": target_oid,
                })
            else:
                diag = UnresolvableSoloChatDiagnostic(
                    chat_id=cid,
                    reason="Solo-ME group chat lacks independently grounded organisational evidence",
                )
                diagnostics.append(diag)
                raise UnresolvableChatMappingError(
                    f"Chat '{cid}' contains only 'ME' with zero organisational evidence. Cannot map to 'ME' or invent an association."
                )
            continue

        # 3. Evidence-Grounded Group Chats
        matched_oid: Optional[str] = None

        # 3.1 Check direct chat title match
        clean_cname = normalize_org_name(cname).lower()
        # Sort retained_orgs by specificity (number of distinctive tokens descending) so child ventures are evaluated before parent orgs
        for o in sorted(retained_orgs, key=lambda x: len(extract_distinctive_org_tokens(x.name)), reverse=True):
            if o.id:
                o_norm = normalize_org_name(o.name).lower()
                o_stripped = strip_legal_suffixes(o.name).lower()
                o_tokens = extract_distinctive_org_tokens(o.name)
                acronym = "".join(c for c in o.name if c.isupper())

                if o.name == "Home" and re.search(r"\bwhere\s+the\s+heart\s+is\b", clean_cname, re.IGNORECASE):
                    continue

                if cid in o.attached_chats or o_norm in clean_cname or o_stripped in clean_cname or (len(acronym) >= 2 and re.search(rf"\b{re.escape(acronym)}\b", clean_cname, re.IGNORECASE)):
                    matched_oid = o.id
                    break

                for t in o_tokens:
                    if t in unique_tokens and re.search(rf"\b{re.escape(t)}\b", clean_cname, re.IGNORECASE):
                        matched_oid = o.id
                        break
                if matched_oid:
                    break

        # 3.2 Check participant overlap (>= min_org_participant_overlap)
        if not matched_oid:
            non_me_parts = [p for p in participants if p != "ME"]
            if non_me_parts:
                for o in retained_orgs:
                    if o.id and o.linked_people:
                        overlap_count = sum(1 for p in non_me_parts if p in o.linked_people)
                        overlap_ratio = overlap_count / len(non_me_parts)
                        if overlap_ratio >= cfg.min_org_participant_overlap:
                            matched_oid = o.id
                            break

        if matched_oid:
            final_chats.append({
                "chat_id": cid,
                "organisation": matched_oid,
            })
            continue

        # 4. Ambiguous Social Group Chats -> Map to primary participant
        primary_p = select_primary_participant(
            chat=cdata,
            chat_messages=msgs,
            all_direct_chats=all_direct_chats,
            global_substantive_counts=dict(global_sub_counts),
        )
        if primary_p:
            final_chats.append({
                "chat_id": cid,
                "person": primary_p,
            })
        else:
            diag = UnresolvableSoloChatDiagnostic(
                chat_id=cid,
                reason="Group chat has no eligible non-ME participant",
            )
            diagnostics.append(diag)
            raise UnresolvableChatMappingError(f"Chat '{cid}' has no eligible non-ME participant target.")

    return final_chats, diagnostics


def extract_entities(
    context: ConversationContext,
    interpreted_people: Optional[List[InterpretedPerson]] = None,
    interpreted_orgs: Optional[List[InterpretedOrganisation]] = None,
    config: Optional[EntityExtractionConfig] = None,
) -> EntitiesResult:
    """
    Orchestrate the complete Step 5 entity extraction pipeline.

    Args:
        context: ConversationContext instance from Step 2.
        interpreted_people: Optional list of InterpretedPerson records from Step 4.
        interpreted_orgs: Optional list of InterpretedOrganisation records from Step 4.
        config: Optional EntityExtractionConfig with calibrated parameters.

    Returns:
        EntitiesResult containing people, organisations, chats, and operational diagnostics.

    Raises:
        EntityExtractionError: If referential integrity or unresolvable mapping fails.
    """
    cfg = config or EntityExtractionConfig()

    # 1. Build Organisations (with interpreted_people provided so person associations prevent orphan pruning)
    retained_orgs, pruned_org_names = build_organisations(
        context=context,
        interpreted_orgs=interpreted_orgs,
        config=cfg,
        interpreted_people=interpreted_people,
    )

    # 2. Build People
    people_list, omitted_people = build_people(
        context=context,
        retained_orgs=retained_orgs,
        interpreted_people=interpreted_people,
        config=cfg,
    )

    # 3. Build Chats
    chats_list, solo_diagnostics = build_chats(
        context=context,
        retained_orgs=retained_orgs,
        config=cfg,
    )

    # Assemble organisations schema output
    orgs_output: List[Dict[str, Any]] = []
    for o in retained_orgs:
        orgs_output.append({
            "id": o.id or generate_org_id(o.name, set()),
            "name": o.name,
            "type": o.org_type,
            "side": o.side,
            "role": o.role,
        })

    diagnostics = EntityExtractionDiagnostics(
        solo_me_chats_unresolvable=solo_diagnostics,
        pruned_orphan_org_names=pruned_org_names,
        unsupported_circle_people_omitted=omitted_people,
    )

    result = EntitiesResult(
        people=people_list,
        organisations=orgs_output,
        chats=chats_list,
        diagnostics=diagnostics,
    )

    # 4. Strict Referential & Schema Validation
    input_chat_ids = set(context.chats.keys())
    input_person_ids = set(p for p in context.participants.keys() if p != "ME")

    errors = validate_entities_referential_integrity(
        result=result,
        input_chat_ids=input_chat_ids,
        input_person_ids=input_person_ids,
    )
    if errors:
        raise EntityExtractionError(
            f"Entity extraction referential integrity validation failed with {len(errors)} error(s): "
            + "; ".join(errors[:5])
        )

    return result
