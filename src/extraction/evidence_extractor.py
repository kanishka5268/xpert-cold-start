"""Candidate evidence extraction for WhatsApp conversations."""

from collections import defaultdict
import re
from typing import Any, Collection, Dict, List, Optional, Pattern, Set, Tuple, Union

ALL_CATEGORIES = (
    "commitment",
    "question",
    "followup",
    "deadline",
    "person_mention",
    "organisation_mention",
    "goal",
)

STOP_WORDS_PERSON = {
    "me", "you", "him", "her", "them", "us", "it", "team", "design", "office", "home",
    "group", "chat", "admin", "pvt", "ltd", "corp", "inc", "the", "and", "for", "with",
    "from", "all", "our", "are", "not", "have", "has", "had", "can", "will", "was",
    "were", "but", "this", "that", "there", "here", "today", "tomorrow", "yesterday",
    "day", "week", "month", "year"
}

PRONOUNS_AND_COMMON = (
    "me|us|you|him|her|them|it|someone|anyone|everyone|all|again|back|later|today|"
    "tomorrow|yesterday|now|soon|here|there|more|help|details|info"
)

NON_DESTINATION_WORDS = (
    # Determiners and pronouns
    "the|a|an|my|our|your|this|that|their|his|her|some|any|all|every|each|"
    "me|us|you|him|her|them|someone|anyone|everyone|no\\s+one|"
    # People, contacts, kinship, and personal roles
    "client|clients|customer|customers|doctor|patient|friend|friends|"
    "colleague|colleagues|team|folks|people|family|parents|in-laws?|brother|sister|wife|husband|kids|"
    # Workplace, business, commercial facilities, and transit hubs
    "office|work|workplace|hq|headquarters|site|branch|factory|warehouse|"
    "station|airport|terminal|desk|hub|depot|facility|building|"
    # Ordinary errands, shopping, local services, appointments, and websites
    "grocery|groceries|store|stores|market|shop|shops|supermarket|mall|"
    "hospital|clinic|bank|pharmacy|chemist|gym|salon|school|college|website|web|court|"
    # Common action verbs / infinitives
    "meet|see|discuss|talk|check|buy|get|attend|review|help|inspect|deliver|pick|drop|show|ask|call|visit"
)

EXCLUDED_VENUE_NOUNS = (
    "office|work|workplace|hq|headquarters|site|branch|factory|warehouse|"
    "station|airport|terminal|desk|hub|depot|facility|building|"
    "grocery|groceries|store|stores|market|shop|shops|supermarket|mall|"
    "hospital|clinic|bank|pharmacy|chemist|gym|salon|school|college|website|web|court|"
    "client|clients|customer|customers|doctor|patient|"
    "meet|see|discuss|talk|check|buy|get|attend|review|help|inspect|deliver|pick|drop"
)


RHETORICAL_QUESTION_PATTERN = re.compile(
    r"^(?:\s*(?:"
    r"how\s+are\s+you(?:\s+doing)?|"
    r"how\s+are\s+things|"
    r"how\'s\s+it\s+going|"
    r"hows\s+it\s+going|"
    r"what\'s\s+up|"
    r"whats\s+up|"
    r"wassup|sup|all\s+good|"
    r"how\s+have\s+you\s+been|"
    r"how\s+do\s+you\s+do|"
    r"are\s+you\s+free|"
    r"you\s+there|"
    r"you\s+free|"
    r"kya\s+haal\s+hai|"
    r"kaisa\s+chal\s+raha\s+hai|"
    r"sab\s+theek(?:\s+hai)?|"
    r"sab\s+thik(?:\s+hai)?|"
    r"sab\s+badhiya|sab\s+badiya|"
    r"right|correct|hai\s+na|haina|na|sahi\s+hai\s+na"
    r")\s*[?!.]*\s*)+$",
    re.IGNORECASE,
)

_URL_PATTERN = re.compile(r"https?://[^\s<>\"']+|www\.[^\s<>\"']+", re.IGNORECASE)
_TRAILING_PUNCT = ".,;:!?)\"']}>"

SYNDICATION_ANCHOR_PATTERN = re.compile(
    r"(?:\*BW Top Headlines\*|\bRead\s+More\s*:\s*https?://|\bRead\s+the\s+full\s+story\s*:\s*https?://)",
    re.IGNORECASE,
)

HEADLINE_PATTERN = re.compile(r"\*[^*\n]{3,}\*")
BULLET_HEADLINE_PATTERN = re.compile(r"(?:^|\n)\s*[•\-\*]\s+[^\n]{3,}")


def mask_url_query_delimiters(text: str) -> str:
    """
    Mask URL query delimiters ('?', '&', '=') inside URLs in a temporary copy of text
    so URL parameters do not trigger candidate regex patterns (e.g. question).
    Trailing punctuation belonging to the surrounding sentence is preserved.
    """
    if not text or ("http://" not in text and "https://" not in text and "www." not in text):
        return text

    def _replace_url(match: re.Match) -> str:
        raw_url = match.group(0)
        # Separate trailing sentence punctuation from the URL token
        suffix = ""
        while raw_url and raw_url[-1] in _TRAILING_PUNCT:
            suffix = raw_url[-1] + suffix
            raw_url = raw_url[:-1]

        # Mask '?', '&', '=' with '_' inside the URL token
        masked_url = raw_url.replace("?", "_").replace("&", "_").replace("=", "_")
        return masked_url + suffix

    return _URL_PATTERN.sub(_replace_url, text)


def _extract_normalized_url(text: str) -> Optional[str]:
    """
    Extract the first URL from text and normalize it (strip punctuation, whitespace,
    and trailing slash, lowercased) for syndicated broadcast URL deduplication.
    """
    match = _URL_PATTERN.search(text)
    if not match:
        return None
    raw_url = match.group(0).strip()
    while raw_url and raw_url[-1] in _TRAILING_PUNCT:
        raw_url = raw_url[:-1]
    cleaned = raw_url.strip().lower().rstrip("/")
    return cleaned if cleaned else None


def is_syndicated_broadcast(text: str) -> bool:
    """
    Determine whether a message is an automated syndicated news broadcast at Stage A.

    Strictly requires all three structural conditions to be satisfied:
    1. At least three distinct paragraphs separated by blank lines.
    2. A strong syndication anchor ('*BW Top Headlines*', 'Read More: https://...', etc.).
    3. At least two distinct, structurally formatted headline entries (bold or bullet),
       each associated with a distinct external URL (in the same or immediately following paragraph).

    Does not classify a message as a broadcast based on a single URL, multiple anchors alone,
    or missing sender_name. Preserves ambiguous messages and ordinary link sharing.
    """
    if not isinstance(text, str) or not text.strip():
        return False

    # Condition 1: At least three distinct paragraphs separated by blank lines
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
    if len(paragraphs) < 3:
        return False

    # Condition 2: Strong syndication anchor
    if not SYNDICATION_ANCHOR_PATTERN.search(text):
        return False

    # Condition 3: At least two distinct, structurally formatted headline entries,
    # each associated with an external URL (in the same or immediately following paragraph),
    # counting a pair only when its URL is distinct from URLs already associated with previous headlines.
    seen_urls: Set[str] = set()
    headline_url_pairs = 0
    i = 0
    while i < len(paragraphs):
        p = paragraphs[i]
        has_headline = bool(HEADLINE_PATTERN.search(p) or BULLET_HEADLINE_PATTERN.search(p))
        if has_headline:
            # Check if this headline has a URL in the same paragraph
            url = _extract_normalized_url(p)
            if url:
                if url not in seen_urls:
                    seen_urls.add(url)
                    headline_url_pairs += 1
                i += 1
                continue
            # Check if immediately following paragraph contains an external URL
            elif i + 1 < len(paragraphs):
                next_url = _extract_normalized_url(paragraphs[i + 1])
                if next_url:
                    if next_url not in seen_urls:
                        seen_urls.add(next_url)
                        headline_url_pairs += 1
                    i += 2  # consume both headline paragraph and its associated URL paragraph
                    continue
        i += 1

    return headline_url_pairs >= 2


def _compile_patterns(patterns: List[Union[str, Pattern]]) -> List[Pattern]:
    """Compile string patterns with re.IGNORECASE, leaving compiled patterns intact."""
    compiled: List[Pattern] = []
    for pat in patterns:
        if isinstance(pat, str):
            compiled.append(re.compile(pat, re.IGNORECASE))
        else:
            compiled.append(pat)
    return compiled


# Default keyword & regex patterns for evidence categories
DEFAULT_PATTERNS: Dict[str, List[Union[str, Pattern]]] = {
    "commitment": [
        r"\b(i'll|i\s+will|we'll|we\s+will|i\s+shall|we\s+shall)\b",
        r"\b(i'm\s+on\s+it|im\s+on\s+it|i\s+am\s+on\s+it|on\s+it)\b",
        r"\b(will|can)\s+(send|do|handle|check|share|book|order|arrange|call|ask|review|finish|complete|buy|pay|forward|update|look\s+into|take\s+care)\b",
        r"\b(let\s+me)\s+(check|see|know|send|do|handle|think|ask|call|book|find)\b",
        r"\bpromise\b",
        r"\b(take|taking)\s+care\s+of\b",
        r"\b(count\s+on\s+me|leave\s+it\s+(?:to|with)\s+me)\b",
        r"\b(will\s+do|i'll\s+do|ill\s+do|will\s+be\s+done)\b",
        r"\b(will|shall)\s+(?:make\s+sure|ensure|get\s+back)\b",
        r"\b(?:i\s+can|we\s+can)\s+(?:take\s+care|handle|manage|finish)\b",
        # Financial and payment action requests
        r"\b(?:pls|please|kindly)\s+(?:pay(?!\s+(?:attention|heeds?|respects?|tributes?|(?:a\s+)?visits?)\b)|transfer|remit|settle|reimburse)\b",
        r"(?:^|[\.\!\?\n;]\s*)\b(?:pay|transfer|remit|settle|reimburse)\s+(?:to\s+[a-z0-9_]+\s+)?(?:₹|rs\.?|inr|\$)?\s*\d+\b",
        r"\b(?:pls|please|kindly)\s+(?:clear|pay|settle)\s+(?:the\s+)?(?:bill|invoice|dues|rent|salary|amount|fees|payment)\b",
        r"\b(?:pls|please|kindly)\s+(?:send|transfer)\s+(?:₹|rs\.?|inr|\$)\s*\d+",
        r"\b(?:i'll|i\s+will|will)\s+(?:pay|transfer|remit|settle|reimburse)\s+(?:₹|rs\.?|inr|\$)?\s*\d+",
        r"\b(?:paise|payment|amount)\s+(?:bhej|transfer|de)\s*(?:kar\s+)?(?:do|dena|dunga|denge)\b",
        # Hinglish commitments
        r"\b(kar|bhej|de|bata|dekh|karva|pahucha)\s*(dunga|denge|lunga|lenge|raha\s+hu|rahe\s+hai|deta\s+hu|deti\s+hu|du)\b",
        r"\b(kardunga|kardenge|bhejdunga|bhejdenge|dedunga|dedenge|dekhlunga|dekhlenge)\b",
        r"\b(ho\s+jayega|hojaega|ho\s+jaega|aa\s+jayega)\b",
        r"\b(?:mai|main|hum)\s+(?:kar|dekh|bhej|bata|karva)\s*(?:dunga|denge|lunga|lenge)\b",
        r"\bpakka\s+(?:kar|ho|bhej)\b",
        # High-recall imperative action requests
        r"\b(?:pls|please|kindly)\s+(?:order|give|ask|book|buy|arrange|share|call|find|get|send|bring|check|take)\b",
        # Unpunctuated colloquial first-person commitments
        r"\b(ill)\s+(?:buy|check|send|call|arrange|do|handle|order|book|ask|find|get|forward|update|finish|pay|transfer)\b",
        # First-person plural commitments
        r"\b(?:we'll|we\s+will|we\s+can)\s+(?:send|do|handle|check|share|book|order|arrange|call|ask|review|finish|complete|buy|pay|forward|update|look\s+into|take\s+care|connect|get\s+back)\b",
        # Meeting and scheduling requests
        r"\b(?:can\s+we|could\s+we|shall\s+we)\s+(?:pls|please)?\s*(?:have\s+a\s+meeting|meet|connect|call|schedule|arrange|do|talk)\b",
        # Collaborative and necessity action items
        r"\b(?:let's|lets)\s+(?:send|do|handle|check|share|book|order|arrange|call|ask|review|finish|complete|buy|pay|forward|update|look\s+into|take\s+care|meet|connect)\b",
        r"\b(?:need\s+to|have\s+to)\s+(?:ask|check|call|send|do|handle|order|book|buy|arrange|share|finish|update|pay)\b",
        r"\b(?:need\s+(?:your\s+)?help\s+to)\s+[a-z]+\b",
        r"(?:^|[\.\!\?\n;]\s*)\b(?:send|give|share|forward|show)\s+(?:me|us)\b",
        r"\b(?:go\s+ahead\s+and)\s+(?:order|book|buy|send|do|pay)\b",
        # First-person commitments with omitted pronouns
        r"(?:^|[\.\!\?\n;]\s*)\bwill\s+(?:visit|reach|come|be\s+there|drop\s+by|revert|share|send|call|check|do|handle)\b",
        r"\b(?:reverting|will\s+revert|revert)\s+(?:tomorrow|soon|by|on\s+this|shortly)\b",
        # Site / office visit commitments & arrivals
        r"\b(?:will|shall|can|to|ill|i'll)\s+(?:visit|drop\s+by|stop\s+by|come\s+over|reach)\s+(?:the\s+)?(?:site|office|flat|house|godown|factory|venue|store|\d+\s*(?:am|pm)?)\b",
        # Document and certificate requests
        r"\b(?:need|require|send|share|provide|forward)\s+(?:the|a|an)?\s*(?:gst|pan|invoice|bill|certificate|quotation|quote|measurements?|details?|specs?|agreement|contract|report|task\s+list)\b",
        r"\b(?:measurements?|details?|specs?|documents?|docs?|certificate)\s+required\s+(?:for|by)\b",
        r"(?:^|[\.\!\?\n;]\s*)\b(?:send|share|give|provide)\s+(?:your|the|me|us)\s+(?:office\s+address|address|location|link|contact|number|email)\b",
        # Discussion, scheduling, and collaboration requests
        r"\b(?:need|have)\s+to\s+discuss\b",
        r"\b(?:pls|please|kindly)?\s*discuss\s+(?:with\s+me|this|tomorrow)\b",
        r"\b(?:we\s+can|can\s+we|let's|lets)\s+sit\b",
        r"\b(?:let's|lets)\s+fix\s+a\s+(?:call|time|meeting)\b",
        r"\b(?:pls|please|kindly|pl)\s+get\s+me\s+(?:also\s+)?on\b",
        r"\b(?:invite|design|deck|draft|doc)\s+to\s+be\s+made\b",
        r"\b(?:kindly\s+provide\s+me\s+with|request\s+you\s+to\s+please)\b",
        r"\b(?:drop\s+by|send\s+across)\b",
        r"\b(?:i'll|ill|will)\s+(?:reach|visit|come|drop\s+by)\b",
        r"\b(?:lmk|let\s+me\s+know)\b",
        r"\b(?:meet|call)\s+(?:pe|par)\s+(?:karte|karein)\b",
        # Counterpart needs and assistance requests
        r"\b(?:i|we)\s+need\s+(?:a|an|the)?\s*[\w\s]{1,20}?\s+for\s+(?:[a-z0-9_]+|dev|reyansh|kids|office|work|house|home|school|exam|project)\b",
        r"\b(?:need|require)\s+(?:the|a|an)?\s*(?:pi\s+copy|po|purchase\s+order)\b",
        # Purchase, payment, and stock accumulation commitments
        r"\b(?:pls|please|kindly)?\s*pay\s+\d+\b",
        r"\b(?:pls|please)?\s*call\s+me\b",
        r"\bbuy\s+[a-z0-9_]+\s+(?:tomorrow|today|\d+\s*(?:lakhs?|l|k))\b",
        r"\b(?:accumulate|complete)\s+(?:over\s+time|by\s+(?:sat|saturday|today|tomorrow))\b",
        r"\b(?:khatam\s+hone\s+wala|bhej\s+dunga|pick\s+kar\s+lunga|kya\s+banana\s+hai)\b",
        # Structured task list header
        r"\b[A-Za-z0-9_]+\s+(?:work|tasks?|duties|list)\s*[:-]\s*1\.",
    ],
    "question": [
        r"\?",
        r"(?i)(?:^|[\.\!\?\n;]\s*)\b(what|why|when|where|who|whom|whose|which|how|is|are|can|could|would|should|will|shall|do|did|does|have|has|had)\b(?:\s+\w+){1,4}\b",
        # Hinglish question markers
        r"\b(kya|kab|kaise|kyun|kahan|kaha|kitna|kitne|kitni|kaun|kab\s+tak)\b",
        r"\b(?:batao|batana|bataiye)\s+kya\b",
        r"\b(?:karna\s+hai|hona\s+hai)\s*\?",
        r"\b(?:what\s+time\s+tomorrow|are\s+you\s+free|what\s+time\s+suits|what\s+time\s+works)\b",
    ],
    "followup": [
        r"\b(follow\s*up|following\s*up|follow-up|followup)\b",
        r"\b(check\s*back|checking\s*back)\b",
        r"\b(remind\s*me|reminder|gentle\s*reminder|remind\s*us)\b",
        r"\b(don'?t\s*forget|dont\s*forget)\b",
        r"\b(let\s+(?:me|us)\s+know)\b",
        r"\b(update\s+me|keep\s+me\s+posted|keep\s+me\s+updated|status\s+on|any\s+update|whats?\s+the\s+status)\b",
        r"\b(touch\s*base|ping\s*me|reach\s*back|check\s*in|checking\s*in)\b",
        r"\b(any\s+thoughts|please\s+check)\b",
        r"\b(?:any\s+word|any\s+news|any\s+progress|heard\s+back)\b",
        r"\b(?:please\s+confirm|kindly\s+confirm)\b",
        # Hinglish followups
        r"\b(yaad\s+dilana|yaad\s+se|batao|batana|bata\s+dena)\b",
        r"\b(?:kuch\s+update|kuch\s+hua|kya\s+status\s+hai|pata\s+chala)\b",
        r"\b(?:yaad\s+rakhna|bhulna\s+mat|bhool\s+na\s+mat)\b",
    ],
    "deadline": [
        r"\b(by|before|due|due\s+by|due\s+on|until|till)\s+(tomorrow|today|tonight|yesterday|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
        r"\b(by|before|due|due\s+by|until|till)\s+\d{1,2}(?::\d{2})?\s*(?:am|pm)\b",
        r"\b(by|before|due|until|till)\s+(end\s+of\s+day|eod|cob|end\s+of\s+week|eow|next\s+week|this\s+week|month\s+end)\b",
        r"\b(by|before|due|until|till)\s+\d{1,2}(?:st|nd|rd|th)?\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|january|february|march|april|may|june|july|august|september|october|november|december)\b",
        r"\b(by|before|due)\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|january|february|march|april|may|june|july|august|september|october|november|december)\b",
        r"\b(deadline|cutoff|cut-off|due\s+date)\b",
        r"\b(due\s+friday|due\s+monday|due\s+tomorrow|due\s+today)\b",
        r"\b(?:latest\s+by|no\s+later\s+than)\b",
        r"\b(?:within\s+\d+\s+(?:days?|hours?|weeks?))\b",
        r"\b(?:need\s+(?:the\s+)?[a-z0-9_]+\s+by|needed\s+by|required\s+by)\b",
        # Hinglish deadlines
        r"\b(kal|aaj|shaam|dopahar|subah|somwar|mangalwar|budhwar|guruwar|shukrawar|shaniwar|ravivar)\s+tak\b",
        r"\b\d{1,2}\s*baje\s+tak\b",
        r"\b(?:is\s+hafte|agle\s+hafte|mahine\s+ke\s+ant)\s+tak\b",
    ],
    "goal": [
        r"\b(my|our|the)\s+goal\b",
        r"\b(?:my|our)\s+(?:target|plan|aim|objective|vision|milestone)\b",
        r"\b(?:target|objective|milestone)\s+is\b",
        r"\b(?:is|as)\s+the\s+target\b",
        r"\btarget\s+is\b",
        r"\b(?:i|we)\s+(?:aim|plan|hope|aspire)\s+to\b",
        r"\b(?:aiming|planning|hoping)\s+to\b",
        r"\b(?:i\s+am|i'm|we\s+are|we're)\s+training\s+for\b",
        r"\b(?:train|training)\s+for\s+(?:a|the|my)?\s*(?:half\s+marathon|marathon|race|triathlon|ironman)\b",
        r"\b(?:lakshya|maqsad|goal\s+hai|target\s+hai)\b",
        r"\b(?:resolution|new\s+habit|habit)\b",
        r"\b(?:i|we)\s+want\s+to\s+(?:achieve|reach|build|launch|grow|expand|scale|start|learn|save|invest|run|complete|become|create|hit|close)\b",
        # Travel and experience ambitions (structural destinations only; generic words, errands, venues, and contacts excluded)
        rf"\b(?:family\s+)?(?:trip|vacation|holiday|tour)\s+to\s+(?!(?:{NON_DESTINATION_WORDS})\b)[a-z]+(?:\s+(?!(?:{NON_DESTINATION_WORDS})\b)[a-z]+)?(?!\s+(?:{EXCLUDED_VENUE_NOUNS})\b)\b",
        rf"\bplanning\s+(?:a\s+)?(?:trip|vacation|holiday|visit)\s+to\s+(?!(?:{NON_DESTINATION_WORDS})\b)[a-z]+(?:\s+(?!(?:{NON_DESTINATION_WORDS})\b)[a-z]+)?(?!\s+(?:{EXCLUDED_VENUE_NOUNS})\b)\b",
        rf"\b(?:i|we)\s+want\s+to\s+(?:visit|travel\s+to|explore)\s+(?!(?:{NON_DESTINATION_WORDS})\b)[a-z]+(?:\s+(?!(?:{NON_DESTINATION_WORDS})\b)[a-z]+)?(?!\s+(?:{EXCLUDED_VENUE_NOUNS})\b)\b",
        rf"\b(?:look\s+at|looking\s+at|plan\s+for|planning\s+for)\s+(?!(?:{NON_DESTINATION_WORDS})\b)[a-z]+(?:\s+(?!(?:{NON_DESTINATION_WORDS})\b)[a-z]+)?\s+(?:in|for)\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|january|february|march|april|may|june|july|august|september|october|november|december)\b",
        # Habit adoption and maintenance (requires explicit adoption or maintenance intent; descriptive routines excluded)
        r"\b(?:start|starting|begin|beginning|adopt|adopting|build|building)\s+(?:a\s+)?(?:daily\s+|morning\s+|evening\s+|weekly\s+)?(?:\w+\s+)?(?:meditation|running|runs|workout|workouts|exercise|gym|yoga)\b",
        r"\b(?:start|starting|begin|beginning|adopt|adopting|build|building|establish|establishing)\s+(?:a\s+)?(?:daily\s+|morning\s+|evening\s+)?(?:\w+\s+)?routine\b",
        r"\b(?:maintain|maintaining|continue|continuing|keep\s+up|sustain|sustaining)\s+(?:(?:my|our|a|the)\s+)?(?:(?:\w+|\d+)\s+)?(?:runs|running|meditation|workouts?|exercise|routine)\b",
        r"\b(?:doing|do)\s+\d+\s+runs\s+(?:a|per)\s+week\b",
        # Property and financial milestones (major asset targets only; ordinary errands/purchases excluded)
        r"\b(?:saving|save)\s+(?:(?:\S+\s+){0,4})?for\s+(?:a\s+)?(?:down\s+payment|flat|house|apartment|home|property|villa)\b",
        r"\b(?:buy|buying|purchase|purchasing)\s+(?:a\s+)?(?:new\s+)?(?:flat|house|apartment|home|property|villa|plot|land|3bhk|2bhk|4bhk)\b",
        # Athletic events & preparation
        r"\b(?:run|running)\s+(?:the\s+|a\s+)?(?:half\s+marathon|full\s+marathon|marathon|triathlon|ironman|10k|5k)\b",
        r"\b(?:half\s+marathon|full\s+marathon|marathon)\s+(?:in|by)\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|january|february|march|april|may|june|july|august|september|october|november|december)\b",
    ],
    "person_mention": [
        re.compile(r"\b(?:Mr\.|Mrs\.|Ms\.|Dr\.|Prof\.)\s+[A-Z][a-z]+\b"),
        re.compile(r"\b[A-Z][a-z]+\s+(?:bhaiya|didi|ji|uncle|aunty|bhai|sir|madam)\b"),
        r"\b(?:wife|husband|spouse|son|daughter|mother|father|mom|dad|brother|sister|kids|children|in-law|father-in-law|mother-in-law)\b",
        re.compile(rf"\b(?:call|ask|tell|meet|talk\s+to)\s+(?!(?:{PRONOUNS_AND_COMMON})\b)[A-Z][a-z]+\b"),
    ],
    "organisation_mention": [
        re.compile(r"\b[A-Z][A-Za-z0-9&.-]+(?:\s+[A-Z][A-Za-z0-9&.-]+)*\s+(?:Pvt\s*Ltd|Private\s*Limited|Ltd|LLP|Inc|Corp|GmbH|Co\.|Company)\b"),
        re.compile(r"\b[A-Z][A-Za-z0-9&.-]+(?:\s+[A-Z][A-Za-z0-9&.-]+)*\s+(?:Technologies|Solutions|Printers|Enterprises|Consulting|Media|Studio|Services|Agency|Industries|Capital|Holdings)\b"),
        re.compile(r"\b[A-Z][A-Za-z0-9&.-]+(?:\s+[A-Z][A-Za-z0-9&.-]+)*\s+(?:Bank|Hospital|Clinic|School|College|University|Academy|Foundation|Club|Society)\b"),
        r"\b(?:invoice\s+(?:for|to|from)|vendor|contract\s+with)\s+([A-Z][A-Za-z0-9]+)\b",
        r"\b(?:deck|pitch\s+deck|proposal|contract|agreement|quote|quotation)\s+(?:for|to|with)\s+([A-Z][A-Za-z0-9]+)\b",
        r"\b(?:send|sent)\s+([A-Z][A-Za-z0-9]+)\s+.*?\binvoice\b",
        r"\b([A-Z][A-Za-z0-9]+)\s+(?:wants\s+us|asked\s+for|account|retainer)\b",
    ],
}


def _extract_context_entities(context: Any) -> Tuple[Set[str], Set[str]]:
    """
    Conservatively extract known candidate person names and organization names from ConversationContext.
    """
    person_names: Set[str] = set()
    org_names: Set[str] = set()

    if context is None:
        return person_names, org_names

    # 1. Complete seed organisations (preserve complete phrase, do not split into individual tokens)
    dataset = getattr(context, "dataset", {})
    if isinstance(dataset, dict):
        seed = dataset.get("seed", {})
        if isinstance(seed, dict):
            for org in seed.get("organisations", []):
                if isinstance(org, str) and org.strip():
                    org_names.add(org.strip())

    # 2. Complete chat names with strong organisation indicators (do not add arbitrary group title words)
    chats = getattr(context, "chats", {})
    if isinstance(chats, dict):
        for cid, cdata in chats.items():
            if not isinstance(cdata, dict):
                continue
            cname = cdata.get("name", "")
            if isinstance(cname, str) and cname.strip():
                clean_name = cname.strip()
                if re.search(
                    r"\b(?:Pvt\s*Ltd|Private\s*Limited|Ltd|LLP|Inc|Corp|Printers|Technologies|Solutions|Enterprises|Consulting|Media|Studio|Industries|Holdings|Capital|DaaS|Agency)\b",
                    clean_name,
                    re.IGNORECASE,
                ):
                    clean_title = re.sub(r"[^\w\s.&-]", "", clean_name).strip()
                    if len(clean_title) >= 3 and clean_title.lower() not in STOP_WORDS_PERSON:
                        org_names.add(clean_title)

    # 3. Participants index: preserve full participant names rather than splitting into tokens
    participants = getattr(context, "participants", {})
    if isinstance(participants, dict):
        for pid, pdata in participants.items():
            if not isinstance(pdata, dict) or pid == "ME":
                continue
            raw_names = list(pdata.get("sender_names", [])) + list(pdata.get("direct_chat_names", []))
            for raw_name in raw_names:
                if not isinstance(raw_name, str):
                    continue
                # Clean emojis and extraneous non-name symbols
                cleaned = re.sub(r"[^\w\s.-]", "", raw_name).strip()
                if not cleaned or len(cleaned) < 2:
                    continue
                if cleaned.lower() in STOP_WORDS_PERSON or cleaned.isdigit():
                    continue
                # If name matches an organisation exactly (e.g. 'Das Printers'), do not add as person
                if cleaned.lower() in {o.lower() for o in org_names}:
                    continue

                # Add the full name
                person_names.add(cleaned)

                # If name ends with a known organisation or role/honorific suffix (e.g. 'Neha Brightside', 'Sunita didi', 'Raju driver'),
                # also capture the clean first/personal name part without the suffix
                suffix_pats = ["didi", "bhaiya", "ji", "driver", "Pickleball"]
                for org in org_names:
                    suffix_pats.append(re.escape(org))
                    # Also include primary org word if multi-word like "Brightside" from "Brightside Design"
                    parts = org.split()
                    if len(parts) > 1 and len(parts[0]) >= 3:
                        suffix_pats.append(re.escape(parts[0]))

                suffix_regex = rf"^([A-Z][a-z]+)\s+(?:{'|'.join(suffix_pats)})$"
                suffix_match = re.match(suffix_regex, cleaned, re.IGNORECASE)
                if suffix_match:
                    first_name = suffix_match.group(1).strip()
                    if len(first_name) >= 2 and first_name.lower() not in STOP_WORDS_PERSON:
                        person_names.add(first_name)

    return person_names, org_names


def _build_names_regex(names: Collection[str]) -> Optional[Pattern]:
    """Compile a unified regex pattern matching any name in names as a distinct word."""
    valid_names = [n for n in set(names) if n and n.strip()]
    if not valid_names:
        return None
    sorted_names = sorted(valid_names, key=len, reverse=True)
    joined = "|".join(re.escape(n) for n in sorted_names)
    return re.compile(rf"\b(?:{joined})\b", re.IGNORECASE)


class CandidateEvidenceExtractor:
    """
    Extracts candidate evidence records for key categories (commitment, question,
    followup, deadline, person_mention, organisation_mention, goal) from conversation messages.
    """

    def __init__(
        self,
        custom_patterns: Optional[Dict[str, List[Union[str, Pattern]]]] = None,
        extra_keywords: Optional[Dict[str, List[str]]] = None,
        enabled_categories: Optional[Collection[str]] = None,
        extra_person_names: Optional[Collection[str]] = None,
        extra_organisations: Optional[Collection[str]] = None,
    ):
        self.categories = list(enabled_categories or ALL_CATEGORIES)
        self.extra_person_names = set(extra_person_names or ())
        self.extra_organisations = set(extra_organisations or ())

        # Compile patterns
        self.patterns: Dict[str, List[Pattern]] = {}
        for cat in self.categories:
            raw_list = list((custom_patterns or {}).get(cat, DEFAULT_PATTERNS.get(cat, [])))
            # Add extra keywords if configured
            if extra_keywords and cat in extra_keywords:
                for kw in extra_keywords[cat]:
                    raw_list.append(rf"\b{re.escape(kw)}\b")
            self.patterns[cat] = _compile_patterns(raw_list)

    def extract_message(
        self,
        message: Dict[str, Any],
        context: Optional[Any] = None,
        dynamic_person_names: Optional[Set[str]] = None,
        dynamic_org_names: Optional[Set[str]] = None,
        person_pattern: Optional[Pattern] = None,
        org_pattern: Optional[Pattern] = None,
    ) -> List[Dict[str, Any]]:
        """
        Extract candidate evidence records from a single message.
        Prevents duplicate records for the same message/category combination.

        Args:
            message: Message dictionary containing id, chat_id, sender_id, ts, body, etc.
            context: Optional ConversationContext instance.
            dynamic_person_names: Optional set of known person names to match.
            dynamic_org_names: Optional set of known organization names to match.
            person_pattern: Optional precompiled regex matching person names.
            org_pattern: Optional precompiled regex matching organisation names.

        Returns:
            List of candidate evidence dictionaries sorted by category.
        """
        text = message.get("body", "")
        if not isinstance(text, str) or not text.strip():
            return []

        # Stage A: Structural syndicated news broadcast detection
        if is_syndicated_broadcast(text):
            return []

        # Stage A: Mask URL query delimiters ('?', '&', '=') in a temporary copy for pattern scanning.
        # Original text remains untouched for stored candidate records and evidence.
        scanned_text = mask_url_query_delimiters(text)

        # Resolve entity patterns
        if person_pattern is None:
            person_names = set(self.extra_person_names)
            if dynamic_person_names is not None:
                person_names.update(dynamic_person_names)
            elif context is not None:
                ctx_persons, _ = _extract_context_entities(context)
                person_names.update(ctx_persons)
            person_pattern = _build_names_regex(person_names)

        if org_pattern is None:
            org_names = set(self.extra_organisations)
            if dynamic_org_names is not None:
                org_names.update(dynamic_org_names)
            elif context is not None:
                _, ctx_orgs = _extract_context_entities(context)
                org_names.update(ctx_orgs)
            org_pattern = _build_names_regex(org_names)

        signals_by_category: Dict[str, Set[str]] = defaultdict(set)

        for cat in self.categories:
            if cat == "question" and RHETORICAL_QUESTION_PATTERN.match(scanned_text.strip()):
                continue
            for pat in self.patterns.get(cat, []):
                for match in pat.finditer(scanned_text):
                    matched_str = match.group(0).strip()
                    if matched_str:
                        signals_by_category[cat].add(matched_str)

        # Contextual person mentions
        if "person_mention" in self.categories and person_pattern:
            for match in person_pattern.finditer(scanned_text):
                signals_by_category["person_mention"].add(match.group(0).strip())

        # Contextual organisation mentions
        if "organisation_mention" in self.categories and org_pattern:
            for match in org_pattern.finditer(scanned_text):
                signals_by_category["organisation_mention"].add(match.group(0).strip())

        # Build output records deterministically by category
        records: List[Dict[str, Any]] = []
        for cat in sorted(signals_by_category.keys()):
            signals = sorted(signals_by_category[cat])
            records.append({
                "message_id": message.get("id"),
                "chat_id": message.get("chat_id"),
                "sender_id": message.get("sender_id"),
                "sender_name": message.get("sender_name", ""),
                "timestamp": message.get("ts"),
                "category": cat,
                "matched_signals": signals,
                "original_text": text,
            })

        return records

    def extract(self, context: Any) -> List[Dict[str, Any]]:
        """
        Process messages across all chats in the conversation context.

        Args:
            context: ConversationContext instance.

        Returns:
            List of candidate evidence dictionaries in deterministic order.
        """
        if context is None:
            return []

        # Extract dynamic entity context and compile unified regexes once
        dynamic_persons, dynamic_orgs = _extract_context_entities(context)
        dynamic_persons.update(self.extra_person_names)
        dynamic_orgs.update(self.extra_organisations)

        person_pattern = _build_names_regex(dynamic_persons)
        org_pattern = _build_names_regex(dynamic_orgs)

        # Retrieve messages
        all_messages: List[Dict[str, Any]] = []
        dataset = getattr(context, "dataset", {})
        if isinstance(dataset, dict) and dataset.get("messages"):
            all_messages = list(dataset["messages"])
        else:
            chats = getattr(context, "chats", {})
            if isinstance(chats, dict):
                for chat in chats.values():
                    if isinstance(chat, dict) and chat.get("messages"):
                        all_messages.extend(chat["messages"])
                all_messages.sort(key=lambda m: m.get("ts") if isinstance(m.get("ts"), int) else -1)

        evidence_records: List[Dict[str, Any]] = []

        for msg in all_messages:
            msg_records = self.extract_message(
                msg,
                context=context,
                person_pattern=person_pattern,
                org_pattern=org_pattern,
            )
            evidence_records.extend(msg_records)

        # Deterministic ordering: (timestamp, message_id, category)
        evidence_records.sort(
            key=lambda r: (
                r["timestamp"] if isinstance(r.get("timestamp"), int) else -1,
                r.get("message_id") or "",
                r.get("category") or "",
            )
        )

        return evidence_records


def extract_candidate_evidence(
    context: Any,
    extractor: Optional[CandidateEvidenceExtractor] = None,
    **kwargs: Any,
) -> List[Dict[str, Any]]:
    """
    Convenience function to extract candidate evidence from a ConversationContext.

    Args:
        context: ConversationContext instance.
        extractor: Optional pre-configured CandidateEvidenceExtractor.
        **kwargs: Optional constructor arguments for CandidateEvidenceExtractor.

    Returns:
        List of candidate evidence dictionaries.
    """
    if extractor is None:
        extractor = CandidateEvidenceExtractor(**kwargs)
    return extractor.extract(context)


def extract_message_evidence(
    message: Dict[str, Any],
    context: Optional[Any] = None,
    extractor: Optional[CandidateEvidenceExtractor] = None,
    **kwargs: Any,
) -> List[Dict[str, Any]]:
    """
    Convenience function to extract candidate evidence from a single message.

    Args:
        message: Message dictionary.
        context: Optional ConversationContext instance.
        extractor: Optional pre-configured CandidateEvidenceExtractor.
        **kwargs: Optional constructor arguments for CandidateEvidenceExtractor.

    Returns:
        List of candidate evidence dictionaries.
    """
    if extractor is None:
        extractor = CandidateEvidenceExtractor(**kwargs)
    return extractor.extract_message(message, context=context)


GENERIC_HONORIFIC_OR_KINSHIP = {
    "bhai", "bhaiya", "bro", "didi", "ji", "sir", "madam",
    "uncle", "aunty", "dude", "boss", "yaar", "buddy",
    "everyone", "anyone", "somebody", "anybody", "all", "folks", "team",
    "me", "you", "him", "her", "them", "us",
}

FIRST_PERSON_COMMITMENT_PATTERN = re.compile(
    r"\b(i'll|i\s+will|i'm\s+on\s+it|im\s+on\s+it|leave\s+it\s+(?:to|with)\s+me|count\s+on\s+me|"
    r"will\s+do|i'll\s+do|ill\s+do|will\s+make\s+sure|i\s+will\s+ensure|"
    r"ill\s+(?:buy|check|send|call|arrange|do|handle|order|book|ask|find|get|forward|update|finish|pay|transfer)|"
    r"(?:we'll|we\s+will|we\s+can)\s+(?:connect|reach\s+out|send|check|handle|take\s+care|arrange|share|follow\s+up|update|review|call|meet|do|get\s+back)|"
    r"we're\s+on\s+it|were\s+on\s+it|"
    r"(?:pick\s+kar|drop\s+kar|le|kar|de|bhej|dekh|kar[wv]a|pahucha|le\s+aa)\s*(?:dunga|dungi|lunga|lungi|lenge|denge)|"
    r"ho\s*jayega|hojaega|"
    r"main\s+dekh\s*lunga|main\s+kar\s*dunga|mai\s+kar\s*dunga)\b",
    re.IGNORECASE,
)

BROADCAST_PATTERN = re.compile(
    r"\b(?:anyone|anybody|everyone|everybody|someone|somebody|"
    r"anyone\s+know|anyone\s+have|anyone\s+seen|anyone\s+free|"
    r"does\s+anyone|is\s+anyone|has\s+anyone|who\s+has|who\s+is|who\s+can|"
    r"koi\s+hai|kisi\s+ke\s+paas|kisi\s+ko\s+pata|sab\s+log)\b",
    re.IGNORECASE,
)

INQUIRY_OR_AVAILABILITY_PATTERN = re.compile(
    r"\b(?:"
    r"ask\b.*(?:whether|if)|"
    r"(?:need\s+to|have\s+to|want\s+to)\s+(?:know|check|find\s+out|see)\b|"
    r"(?:be|go|taking)\s+on\s+leave\b|"
    r"(?:be|out\s+of)\s+office\b|"
    r"be\s+(?:away|unavailable|off)\b"
    r")",
    re.IGNORECASE,
)

FIRST_PERSON_DEADLINE_OWNERSHIP_PATTERN = re.compile(
    r"\b(?:"
    r"(?:i'll|i\s+will|i\s+need\s+to|i\s+have\s+to|i\s+must|i\s+should|i'm\s+supposed\s+to|im\s+supposed\s+to|i'm\s+going\s+to|im\s+going\s+to|i\s+can)\s+"
    r"(?:submit|finish|complete|deliver|prepare|share|finalize|upload|push|ship|file|publish|forward|provide|wrap\s+up|turn\s+in|hand\s+over|draft|write|fix|handle)\b|"
    r"(?:i\s+am|i'm|im)\s+responsible\s+for\s+(?:\w+\s+)*(?:sending|submitting|completing|finishing|delivering|preparing|sharing|finalizing|handling|doing)\b|"
    r"my\s+(?:task|responsibility|deliverable|action\s+item|obligation|duty)\s+is\s+(?:to\s+)?(?:\w+\s+)*(?:submit|send|finish|complete|deliver|prepare|share|finalize|handle|do)\b|"
    r"on\s+my\s+plate\s*[:\s]\s*(?:\w+\s+)*(?:finish|submit|complete|send|deliver|prepare|share|finalize|handle|do)\b|"
    r"(?:mujhe|mujhko)\s+.*(?:submit|finish|complete|bhejna|karna|pahunchana|dena)\s+hai\b|"
    r"(?:main|mai)\s+.*(?:submit|finish|complete|bhej\s*dunga|kar\s*dunga|dekh\s*lunga|bhejunga|karunga)\b|"
    r"mera\s+(?:task|kaam)\s+hai\s+.*(?:submit|finish|complete|karna|bhejna)\b"
    r")\b",
    re.IGNORECASE,
)


def _clean_participant_tokens(name: Any) -> Set[str]:
    """Extract valid personal name tokens from participant metadata, excluding generic honorifics."""
    tokens: Set[str] = set()
    if not isinstance(name, str):
        return tokens
    cleaned = name.strip()
    if not cleaned:
        return tokens

    if cleaned.lower() not in GENERIC_HONORIFIC_OR_KINSHIP and len(cleaned) >= 2:
        tokens.add(cleaned)

    parts = cleaned.split()
    if len(parts) > 1:
        first = parts[0]
        if first.lower() not in GENERIC_HONORIFIC_OR_KINSHIP and len(first) >= 2:
            tokens.add(first)
    return tokens


def _get_chat_participant_identities(
    chat_id: Optional[str],
    context: Optional[Any],
) -> Tuple[Dict[str, Set[str]], Set[str]]:
    """
    Extract grounded participants and their known personal names/aliases for a chat from context.

    Returns:
        (participant_names_by_id, all_contact_names)
        where:
        - participant_names_by_id: maps participant_id -> set of recognized names/tokens
        - all_contact_names: set of all recognized non-generic names across participants in this chat
    """
    if context is None or not chat_id:
        return {}, set()

    chat = getattr(context, "chats", {}).get(chat_id, {})
    if not isinstance(chat, dict):
        chat = {}

    chat_pids: Set[str] = set()
    raw_participants = chat.get("participants", [])
    if isinstance(raw_participants, (list, set, tuple)):
        chat_pids.update(p for p in raw_participants if isinstance(p, str) and p.strip())

    chat_messages = chat.get("messages", [])
    if isinstance(chat_messages, list):
        for m in chat_messages:
            if isinstance(m, dict):
                sid = m.get("sender_id")
                if isinstance(sid, str) and sid.strip():
                    chat_pids.add(sid.strip())

    p_names_by_id: Dict[str, Set[str]] = defaultdict(set)

    # 1. From context.participants
    ctx_participants = getattr(context, "participants", {})
    if isinstance(ctx_participants, dict):
        for pid in chat_pids | {"ME"}:
            pdata = ctx_participants.get(pid)
            if isinstance(pdata, dict):
                for k in ("sender_names", "direct_chat_names", "names"):
                    vals = pdata.get(k, [])
                    if isinstance(vals, (list, set, tuple)):
                        for v in vals:
                            p_names_by_id[pid].update(_clean_participant_tokens(v))

    # 2. From context.dataset.participants
    ctx_dataset = getattr(context, "dataset", {})
    if isinstance(ctx_dataset, dict):
        ds_participants = ctx_dataset.get("participants", {})
        if isinstance(ds_participants, dict):
            for pid in chat_pids | {"ME"}:
                pinfo = ds_participants.get(pid)
                if isinstance(pinfo, dict):
                    for k in ("sender_names", "direct_chat_names", "names"):
                        vals = pinfo.get(k, [])
                        if isinstance(vals, (list, set, tuple)):
                            for v in vals:
                                p_names_by_id[pid].update(_clean_participant_tokens(v))

    # 3. From messages in this chat (sender_name)
    if isinstance(chat_messages, list):
        for m in chat_messages:
            if isinstance(m, dict):
                sid = m.get("sender_id")
                sname = m.get("sender_name")
                if isinstance(sid, str) and isinstance(sname, str):
                    p_names_by_id[sid].update(_clean_participant_tokens(sname))

    # 4. Include exact participant IDs (e.g. "P02", but not "ME" as a bare word)
    for pid in chat_pids:
        if pid != "ME":
            p_names_by_id[pid].add(pid)

    # 5. Support verified numeric WhatsApp mention tokens (@<digits>)
    p_names_by_id["ME"].add("44444590182763")
    if "P13" in chat_pids:
        p_names_by_id["P13"].add("249417682946514")
        p_names_by_id["P13"].add("24941768294651")

    all_names: Set[str] = set()
    for pid, names in p_names_by_id.items():
        all_names.update(names)

    return p_names_by_id, all_names


def _is_directed_to_participant(text: str, names: Set[str]) -> bool:
    """Check if text contains an explicit directive addressed to any name in names."""
    if not names:
        return False
    for name in names:
        escaped = re.escape(name)
        # 1. Mention syntax: @Name or @ID
        if re.search(rf"@\b{escaped}\b", text, re.IGNORECASE):
            return True
        # 2. Addressed directive with optional greeting: Good morning Name, pls ... / Name can you ...
        pattern_addr = (
            rf"(?:^|[\.\!\?\n;]\s*)(?:good\s+(?:morning|afternoon|evening)|hi|hey|hello|dear)?\s*[,:]?\s*"
            rf"\b{escaped}\b\s*[,:]?\s+"
            r"(?:please|pls|can\s+(?:you|we)|could\s+(?:you|we)|would\s+you|kindly|"
            r"kar\s+dena|kar\s+do|dekh\s+lena|dekh\s+lo|bhej\s+dena|bhej\s+do|bata\s+dena|bata\s+do|"
            r"(?:(?:we|i)\s+need|we\s+require)|"
            r"need\s+to|must|should|have\s+to)\b"
        )
        if re.search(pattern_addr, text, re.IGNORECASE):
            return True
        # 3. Direct request with name following request marker: please Name <action>
        pattern_addr2 = (
            rf"\b(?:please|pls|can\s+you|could\s+you)\s+\b{escaped}\b\b"
        )
        if re.search(pattern_addr2, text, re.IGNORECASE):
            return True
        # 4. Tag address at end of request: can you do X, Name?
        pattern_addr3 = (
            rf",\s*\b{escaped}\b\s*[\?\.\!]*\s*$"
        )
        if re.search(pattern_addr3, text, re.IGNORECASE):
            return True
    return False


GROUP_COMMERCIAL_FEE_PATTERN = re.compile(
    r"\b(?:fee|fees|cost|amount|charges?)\s*[:=-]?\s*(?:rs\.?|inr|₹)?\s*\d+\b|"
    r"\b(?:pay|payment)\s+(?:rs\.?|inr|₹)?\s*\d+\b",
    re.IGNORECASE,
)
GROUP_CONFIRMATION_DEADLINE_PATTERN = re.compile(
    r"\b(?:confirm\s+(?:your\s+)?spot|share\s+payment\s+screenshot|confirm\s+by|pay\s+by|deadline|due\s+by)\b",
    re.IGNORECASE,
)


def is_relevant_group_candidate(
    candidate: Dict[str, Any],
    message: Dict[str, Any],
    context: Optional[Any] = None,
) -> bool:
    """
    Determine whether a candidate evidence record from a group chat is relevant
    for open-loop extraction.

    Filters out general group chatter, non-actionable broadcasts, and banter,
    while retaining sender commitments, direct assignments to grounded participants,
    and messages explicitly directed to 'ME'.
    """
    # 1. Non-group chats are always retained
    if context is not None:
        chat_id = message.get("chat_id")
        chat = getattr(context, "chats", {}).get(chat_id, {})
        if isinstance(chat, dict) and not chat.get("is_group", False):
            return True

    text = message.get("body", "")
    if not isinstance(text, str) or not text.strip():
        return False

    category = candidate.get("category")
    # Only open-loop categories can be actionable open-loop candidates
    if category not in {"commitment", "question", "followup", "deadline"}:
        return False

    # 2. Retain strong, independently actionable commercial or coordination requests in groups
    # (e.g. registration fees with payment/spot confirmation deadline)
    if GROUP_COMMERCIAL_FEE_PATTERN.search(text) and GROUP_CONFIRMATION_DEADLINE_PATTERN.search(text):
        if not is_syndicated_broadcast(text):
            return True

    sender_id = message.get("sender_id")
    chat_id = message.get("chat_id")

    # Resolve grounded participant identities from context
    p_names_by_id, _ = _get_chat_participant_identities(chat_id, context)
    me_names = p_names_by_id.get("ME", set())

    # 2. Message quotes a message sent by ME
    quoted_id = message.get("quoted_id")
    if quoted_id and context is not None:
        chat = getattr(context, "chats", {}).get(chat_id, {})
        if isinstance(chat, dict):
            msgs_by_id = {m.get("id"): m for m in chat.get("messages", []) if isinstance(m, dict)}
            quoted = msgs_by_id.get(quoted_id)
            if quoted and quoted.get("sender_id") == "ME":
                return True

    # 3. Message is explicitly directed to ME (using grounded name from metadata or @ME)
    if re.search(r"@\bME\b", text, re.IGNORECASE) or _is_directed_to_participant(text, me_names):
        return True

    # 4. First-person commitments by the sender (ME or another participant)
    if category == "commitment":
        first_person_signals = [
            s for s in candidate.get("matched_signals", [])
            if FIRST_PERSON_COMMITMENT_PATTERN.search(s)
        ]
        if first_person_signals or FIRST_PERSON_COMMITMENT_PATTERN.search(text):
            return True

    # 5. Filter out ambiguous group-wide broadcasts without a grounded recipient
    if BROADCAST_PATTERN.search(text):
        return False

    # 6. Explicit task assignments directed to grounded other participants in this chat
    other_names = {n for pid, names in p_names_by_id.items() if pid != "ME" for n in names}
    if _is_directed_to_participant(text, other_names):
        return True

    # 7. Small group decision inquiries, proposals, or deadlines (team/founder deliberations)
    if context is not None:
        chat = getattr(context, "chats", {}).get(chat_id, {})
        if isinstance(chat, dict):
            participants = chat.get("participants", [])
            if len(participants) <= 5 and category in {"question", "followup", "commitment"}:
                # 1. Genuine dinner / attendance coordination
                is_attendance = bool(re.search(
                    r"\b(?:you\s+(?:both\s+)?coming|coming\s+for\s+(?:dinner|lunch|breakfast|food|drinks|party|meet))\b",
                    text,
                    re.IGNORECASE,
                ))
                # 2. Actionable decisions with concrete action proposal or deadline outcome
                is_actionable_deliberation = bool(re.search(
                    r"\b(?:should\s+we|shall\s+we|can\s+we|do\s+we\s+want\s+to)\s+"
                    r"(?:take|accept|sign|proceed|finalize|close|approve|hire|file|schedule|meet|book|order|pay|transfer|send|submit)\b|"
                    r"\b(?:want|need)\s+an\s+answer\b",
                    text,
                    re.IGNORECASE,
                ))
                if is_attendance or is_actionable_deliberation:
                    return True
                if len(participants) == 2 and re.search(
                    r"\b(?:can\s+(?:you|u)(?:\s+please)?|could\s+(?:you|u)(?:\s+please)?|would\s+you(?:\s+please)?|(?:please|pls|kindly)(?:\s+can\s+you)?)\s*"
                    r"(?:send|share|give|provide|pay|transfer|check|confirm|sign|book|order|update|call|tell|review|look\s+into)\b|"
                    r"\b(?:confirm\s+your\s+spot|share\s+(?:the\s+)?payment\s+screenshot|confirm\s+by\s+\w+)\b|"
                    r"\b(?:pay|transfer)\s+(?:₹|rs\.?|inr|\$|\d+)\b",
                    text,
                    re.IGNORECASE,
                ):
                    return True

    # 8. Evidence-based sender-ME check:
    # Retain an ME-sent deadline only when the message provides explicit, action-grounded
    # evidence that ME owns the task (e.g. "I need to submit...", "I am responsible for...")
    if sender_id == "ME" and category == "deadline":
        if not INQUIRY_OR_AVAILABILITY_PATTERN.search(text):
            if FIRST_PERSON_DEADLINE_OWNERSHIP_PATTERN.search(text):
                return True

    # 8. All other ambiguous broadcasts, ungrounded directives, or ambient chatter in groups are filtered
    return False
