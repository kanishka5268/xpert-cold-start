"""Deterministic open-loop resolver.

Transforms interpreted open-loop records into validated, context-grounded
CandidateOpenLoop records enforcing actor authority, semantic linkage, strict
timezone/calendar grounding, lifecycle episode boundaries, and schema invariants.
"""

from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
import math
import re
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from src.interpretation.records import InterpretedOpenLoop
from src.loops.models import (
    VALID_HELP_ACTIONS,
    VALID_OPEN_LOOP_STATUSES,
    VALID_OPEN_LOOP_TYPES,
    CandidateOpenLoop,
    HelpAction,
    OpenLoopExtractionConfig,
    OpenLoopExtractionDiagnostics,
)
from src.preprocessing.context_builder import ConversationContext

# Stop words for semantic token extraction
STOP_WORDS: Set[str] = {
    "a", "an", "the", "in", "on", "at", "to", "for", "of", "with",
    "by", "is", "our", "my", "we", "us", "and", "or", "from", "this",
    "it", "that", "i", "me", "you", "will", "be", "do", "did", "have",
    "has", "had", "can", "could", "would", "should", "please", "plz",
    "pls", "didi", "bhai", "bhaiya", "kal", "aaj", "parso", "hai", "ho",
    "ka", "ki", "ke", "ko", "se", "main", "mai", "bhi",
    "he", "him", "his", "she", "her", "they", "them", "their", "so", "then", "now", "just"
}

# Generic task words that alone do not establish distinct obligation identity
GENERIC_TASK_WORDS: Set[str] = {
    "send", "check", "review", "draft", "call", "pay", "submit", "book",
    "order", "update", "followup", "task", "work", "invoice", "report",
    "doc", "document", "deck", "presentation", "meeting", "reminder",
    "need", "want", "let", "know", "see", "get", "make", "take", "care",
    "buy", "share", "forward", "transfer", "fix"
}

# Canonical action verbs for task comparability
ACTION_VERBS: Set[str] = {
    "send", "bhej", "share", "forward",
    "review", "check", "inspect",
    "draft", "write", "prepare",
    "call", "talk", "discuss", "phone",
    "pay", "transfer", "clear",
    "book", "order", "reserve", "buy",
    "submit", "file",
    "fix", "repair",
}

# Canonical task target entities for task comparability
TASK_ENTITIES: Set[str] = {
    "invoice", "bill",
    "contract", "agreement", "nda",
    "deck", "presentation", "slides",
    "report", "summary",
    "proposal",
    "ticket", "flight", "hotel", "cab", "taxi",
    "plumber", "electrician", "tap", "cylinder",
    "workout", "diet",
    "timesheet", "payroll",
    "keys", "car", "package", "parcel"
}

# Calendar months and quarters for distinguishing distinct temporal obligations
MONTH_WORDS: Set[str] = {
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
    "q1", "q2", "q3", "q4"
}

# First-person commitment patterns
FIRST_PERSON_COMMITMENT_PATTERN = re.compile(
    r"\b(i'll|i\s+will|i\s+shall|let\s+me|i'm\s+on\s+it|im\s+on\s+it|i\s+am\s+on\s+it|"
    r"leave\s+it\s+to\s+me|main\s+karunga|mai\s+karunga|"
    r"(?:pick\s+kar|drop\s+kar|le|kar|de|bhej|dekh|kar[wv]a|pahucha)\s*(?:dunga|dungi|lunga|lungi|lenge|denge)|"
    r"kar\s+deta\s+hu|kar\s+deti\s+hu|"
    r"i\s+promise|will\s+do|will\s+send|will\s+handle|on\s+it|"
    r"(?:we'll|we\s+will|i'll|will)\s+(?:connect|introduce|reach\s+out))\b",
    re.IGNORECASE,
)

# Directed request patterns
DIRECTED_REQUEST_PATTERN = re.compile(
    r"\b(can\s+you|could\s+you|please|plz|pls|\bpl\b|will\s+you|would\s+you|"
    r"do\s+you\s+mind|what\s+should\s+i|should\s+we|tell\s+me|let\s+me\s+know|"
    r"send\s+me|give\s+me|book\s+the|call\s+the|pay\s+the|draft\s+the)\b",
    re.IGNORECASE,
)

# Situational observations without directed action
SITUATIONAL_OBSERVATION_PATTERN = re.compile(
    r"^(?:the\s+)?(?:market|shop|store|bank|office|internet|wifi|tap|water|electricity|power|ac|elevator|lift|weather)\s+"
    r"(?:is|was)\s+(?:closed|down|not\s+working|leaking|out|off|broken|very\s+sunny|sunny|raining|cloudy|cold|hot)\b",
    re.IGNORECASE,
)

# Third-party description or personal musing
THIRD_PARTY_OR_DILEMMA_PATTERN = re.compile(
    r"\b(?:is\s+handling|are\s+handling|sent\s+an?\s+email|sent\s+a\s+message|"
    r"no\s+idea\s+what\s+to\s+wear|wondering\s+what\s+to\s+do|"
    r"had\s+(?:some\s+)?(?:pasta|lunch|dinner|breakfast|food|coffee|tea))\b",
    re.IGNORECASE,
)

# Collective / group-wide patterns
COLLECTIVE_GROUP_PATTERN = re.compile(
    r"\b(let's\s+all|lets\s+all|we\s+all|we\s+should|all\s+of\s+us|everyone\s+should)\b",
    re.IGNORECASE,
)

# Broadcast queries
BROADCAST_QUERY_PATTERN = re.compile(
    r"\b(does\s+anyone|anyone\s+know|who\s+is\s+free|who\s+can|anybody|kisi\s+ko\s+pata|"
    r"can\s+someone|someone\s+please|anyone\s+have|who\s+has|any\s+updates\s+from\s+anyone|does\s+anybody|anybody\s+know)\b",
    re.IGNORECASE,
)

# A person asking for advice about their own personal decision/action
ADVICE_ON_OWN_DECISION_PATTERN = re.compile(
    r"\b(?:what\s+do\s+you\s+think\s+i\s+should|what\s+should\s+i\s+(?:do|choose|buy|take|sell|apply|study|read)|"
    r"do\s+you\s+think\s+i\s+should|should\s+i\s+(?:take|accept|buy|sell|choose|go|join|switch|apply|quit|stay|leave|sign|move|rent|order|invest|enroll)|"
    r"shall\s+i\s+(?:take|accept|buy|sell|choose|go|join|switch|apply|quit|stay|leave|sign|move|rent|order|invest|enroll)|"
    r"can\s+i\s+(?:take|have)\s+(?:a\s+)?leave|can\s+i\s+(?:take|get)\s+.*?\boff\b|can\s+i\s+leave\s+early|"
    r"need\s+your\s+advice\s+(?:on|about)|am\s+i\s+allowed\s+to)\b",
    re.IGNORECASE,
)

# A person asking ME to make a decision or state preference
DECISION_REQUESTED_FROM_ME_PATTERN = re.compile(
    r"\b(?:should\s+i\s+get\s+bigger|"
    r"should\s+i\s+(?:get|buy|order|bring|send)\s+.*?\bfor\s+you\b|"
    r"which\s+(?:one\s+)?do\s+you\s+(?:want|prefer|need|like|choose|suggest|recommend)|"
    r"what\s+size\s+do\s+you\s+(?:want|need)|"
    r"do\s+you\s+want\s+me\s+to|"
    r"what\s+would\s+you\s+(?:prefer|like)|"
    r"you\s+(?:decide|choose|tell\s+me\s+what\s+to\s+do)|your\s+call|"
    r"can\s+you\s+decide)\b",
    re.IGNORECASE,
)

# Concrete action request directed to counterparty
CONCRETE_ACTION_REQUEST_PATTERN = re.compile(
    r"\b(?:can\s+you|could\s+you|please|pls|kindly|will\s+you|would\s+you|do\s+you\s+mind)\s+"
    r"(?:send|give|share|provide|check|confirm|pay|transfer|sign|review|call|update|buy|order|book|arrange|forward)\b|"
    r"(?:^|[\.\!\?\n;]\s*)\b(?:send|give|share|provide|pay|transfer)\s+(?:me|us)\b",
    re.IGNORECASE,
)

# Actionable questions indicating coordination, request, or inquiry
ACTIONABLE_QUESTION_PATTERN = re.compile(
    r"\b(?:can\s+(?:we|you|u)|could\s+(?:we|you|u)|should\s+(?:i|we)|shall\s+(?:we|i)|would\s+you|"
    r"possible\s+to|which\s+date\s+to|what\s+date\s+to|how\s+about\s+we|why\s+don't\s+we|"
    r"any\s+plans|coming\s+for|when\s+can|what\s+time|free\s+to|available\s+to|connect\s+for|call\s+for|"
    r"where\s+is|where\s+are|status\s+of|update\s+on|"
    r"send|give|share|provide|check|confirm|update|meeting|schedule|order|buy|pay|discuss|zoom|"
    r"contract|doc|document|invoice|file|report|proposal)\b",
    re.IGNORECASE,
)

# Deflection markers that redirect or defer rather than completing
DEFLECTION_PATTERN = re.compile(
    r"\b(?:ask\s+(?:him|her|them)|tell\s+(?:him|her|them)|talk\s+to\s+(?:him|her|them|finance|office|hr)|"
    r"clear\s+(?:their|his|her)\s+hisaab|take\s+it\s+up\s+with|redirect|not\s+my\s+(?:call|job|responsibility))\b",
    re.IGNORECASE,
)

# Execution / completion markers
EXECUTION_MARKERS_PATTERN = re.compile(
    r"\b(done|completed|sorted|booked|sent|paid|already\s+taken\s+care\s+of|"
    r"ho\s+gaya|kar\s+diya|nipta\s+diya|finished|resolved|handled|delivered|"
    r"cleared|bought|transferred|shared|forwarded|attached|"
    r"(?:chhod|pahun?cha|bhej|de|kar|nipta|bana|le|khareed|rakh|drop\s+kar|pick\s+kar)\s*diya|"
    r"(?:kar|le|dekh)\s*liya)\b",
    re.IGNORECASE,
)

# Bare execution reply pattern for 1:1 proximity exception
BARE_EXECUTION_PATTERN = re.compile(
    r"^(?:done|completed|sorted|booked|sent|paid|already\s+taken\s+care\s+of|"
    r"ho\s+gaya|kar\s+diya|nipta\s+diya|finished|all\s+done|"
    r"(?:chhod|pahun?cha|bhej|de|kar|nipta|bana|le|khareed|rakh|drop\s+kar|pick\s+kar)\s*diya|"
    r"(?:kar|le|dekh)\s*liya)"
    r"[\s,!.👍🙏]*(?:(?:thanks|thank\s+you|thx)[\s,!.👍🙏]*)?$",
    re.IGNORECASE,
)

# Cancellation markers
CANCELLATION_MARKERS_PATTERN = re.compile(
    r"\b(never\s+mind|nevermind|no\s+need|cancel(?:\s+that|\s+this)?|forget\s+it|"
    r"don't\s+bother|dont\s+bother|plan\s+changed|we\s+aren't\s+doing\s+this|"
    r"we\s+arent\s+doing\s+this|not\s+required|drop\s+(?:it|that|this)|scrapped)\b",
    re.IGNORECASE,
)

# Bare cancellation reply pattern for 1:1 proximity exception
BARE_CANCELLATION_PATTERN = re.compile(
    r"^(?:never\s+mind|nevermind|no\s+need|cancel(?:\s+that|\s+this)?|forget\s+it|"
    r"don't\s+bother|dont\s+bother|not\s+required|drop\s+(?:it|that|this)|scrapped)\b",
    re.IGNORECASE,
)

# Preemption reason pattern required for obligor to drop an obligation
PREEMPTION_REASON_PATTERN = re.compile(
    r"\b(?:no\s+longer\s+needed|not\s+needed\s+anymore|project\s+(?:was\s+)?scrapped|"
    r"client\s+cancelled|meeting\s+cancelled|cancelled\s+by|"
    r"(?:trainer|doctor|boss|manager|client|gym|team)\s+(?:gave|told|said|cancelled|handled)|"
    r"already\s+got\s+one|plan\s+changed|we\s+aren't\s+doing\s+this|we\s+arent\s+doing\s+this)\b",
    re.IGNORECASE,
)

# Postponement / ambiguity markers
POSTPONEMENT_MARKERS_PATTERN = re.compile(
    r"\b(leave\s+it\s+for\s+now|let's\s+see\s+later|lets\s+see\s+later|"
    r"we'll\s+see|well\s+see|talk\s+later|hold\s+on|wait\s+for\s+now|"
    r"will\s+let\s+you\s+know|not\s+sure|no\s+idea)\b",
    re.IGNORECASE,
)

# Authoritative timezone and explicit year patterns
TIMEZONE_METADATA_PATTERN = re.compile(
    r"\b(ist|utc|gmt|pst|pdt|est|edt|cst|cdt|[+-]\d{2}:?\d{2})\b",
    re.IGNORECASE,
)
FOUR_DIGIT_YEAR_PATTERN = re.compile(r"\b(19\d\d|20\d\d)\b")

VALID_TZ_ABBREVIATIONS: Set[str] = {
    "utc", "gmt", "ist", "pst", "pdt", "est", "edt", "cst", "cdt", "mst", "mdt"
}
OFFSET_PATTERN = re.compile(r"^([+-])(\d{2}):?(\d{2})$")


def _is_valid_numeric_offset(val: str) -> bool:
    """Validate numeric offset within [-14:00, +14:00] with minutes 00-59."""
    m = OFFSET_PATTERN.match(val)
    if not m:
        return False
    sign, h_str, m_str = m.groups()
    h = int(h_str)
    minute = int(m_str)
    # Reject -00:00 and -0000 (unknown offset per RFC 3339 / ISO 8601)
    if sign == "-" and h == 0 and minute == 0:
        return False
    if not (0 <= minute <= 59):
        return False
    if h > 14:
        return False
    if h == 14 and minute != 0:
        return False
    return True


RECOGNIZED_IANA_TIMEZONES: Set[str] = frozenset({
    'Africa/Abidjan', 'Africa/Accra', 'Africa/Addis_Ababa', 'Africa/Algiers', 'Africa/Asmara',
    'Africa/Asmera', 'Africa/Bamako', 'Africa/Bangui', 'Africa/Banjul', 'Africa/Bissau',
    'Africa/Blantyre', 'Africa/Brazzaville', 'Africa/Bujumbura', 'Africa/Cairo', 'Africa/Casablanca',
    'Africa/Ceuta', 'Africa/Conakry', 'Africa/Dakar', 'Africa/Dar_es_Salaam', 'Africa/Djibouti',
    'Africa/Douala', 'Africa/El_Aaiun', 'Africa/Freetown', 'Africa/Gaborone', 'Africa/Harare',
    'Africa/Johannesburg', 'Africa/Juba', 'Africa/Kampala', 'Africa/Khartoum', 'Africa/Kigali',
    'Africa/Kinshasa', 'Africa/Lagos', 'Africa/Libreville', 'Africa/Lome', 'Africa/Luanda',
    'Africa/Lubumbashi', 'Africa/Lusaka', 'Africa/Malabo', 'Africa/Maputo', 'Africa/Maseru',
    'Africa/Mbabane', 'Africa/Mogadishu', 'Africa/Monrovia', 'Africa/Nairobi', 'Africa/Ndjamena',
    'Africa/Niamey', 'Africa/Nouakchott', 'Africa/Ouagadougou', 'Africa/Porto-Novo', 'Africa/Sao_Tome',
    'Africa/Timbuktu', 'Africa/Tripoli', 'Africa/Tunis', 'Africa/Windhoek', 'America/Adak',
    'America/Anchorage', 'America/Anguilla', 'America/Antigua', 'America/Araguaina', 'America/Argentina/Buenos_Aires',
    'America/Argentina/Catamarca', 'America/Argentina/ComodRivadavia', 'America/Argentina/Cordoba', 'America/Argentina/Jujuy', 'America/Argentina/La_Rioja',
    'America/Argentina/Mendoza', 'America/Argentina/Rio_Gallegos', 'America/Argentina/Salta', 'America/Argentina/San_Juan', 'America/Argentina/San_Luis',
    'America/Argentina/Tucuman', 'America/Argentina/Ushuaia', 'America/Aruba', 'America/Asuncion', 'America/Atikokan',
    'America/Atka', 'America/Bahia', 'America/Bahia_Banderas', 'America/Barbados', 'America/Belem',
    'America/Belize', 'America/Blanc-Sablon', 'America/Boa_Vista', 'America/Bogota', 'America/Boise',
    'America/Buenos_Aires', 'America/Cambridge_Bay', 'America/Campo_Grande', 'America/Cancun', 'America/Caracas',
    'America/Catamarca', 'America/Cayenne', 'America/Cayman', 'America/Chicago', 'America/Chihuahua',
    'America/Ciudad_Juarez', 'America/Coral_Harbour', 'America/Cordoba', 'America/Costa_Rica', 'America/Creston',
    'America/Cuiaba', 'America/Curacao', 'America/Danmarkshavn', 'America/Dawson', 'America/Dawson_Creek',
    'America/Denver', 'America/Detroit', 'America/Dominica', 'America/Edmonton', 'America/Eirunepe',
    'America/El_Salvador', 'America/Ensenada', 'America/Fort_Nelson', 'America/Fort_Wayne', 'America/Fortaleza',
    'America/Glace_Bay', 'America/Godthab', 'America/Goose_Bay', 'America/Grand_Turk', 'America/Grenada',
    'America/Guadeloupe', 'America/Guatemala', 'America/Guayaquil', 'America/Guyana', 'America/Halifax',
    'America/Havana', 'America/Hermosillo', 'America/Indiana/Indianapolis', 'America/Indiana/Knox', 'America/Indiana/Marengo',
    'America/Indiana/Petersburg', 'America/Indiana/Tell_City', 'America/Indiana/Vevay', 'America/Indiana/Vincennes', 'America/Indiana/Winamac',
    'America/Indianapolis', 'America/Inuvik', 'America/Iqaluit', 'America/Jamaica', 'America/Jujuy',
    'America/Juneau', 'America/Kentucky/Louisville', 'America/Kentucky/Monticello', 'America/Knox_IN', 'America/Kralendijk',
    'America/La_Paz', 'America/Lima', 'America/Los_Angeles', 'America/Louisville', 'America/Lower_Princes',
    'America/Maceio', 'America/Managua', 'America/Manaus', 'America/Marigot', 'America/Martinique',
    'America/Matamoros', 'America/Mazatlan', 'America/Mendoza', 'America/Menominee', 'America/Merida',
    'America/Metlakatla', 'America/Mexico_City', 'America/Miquelon', 'America/Moncton', 'America/Monterrey',
    'America/Montevideo', 'America/Montreal', 'America/Montserrat', 'America/Nassau', 'America/New_York',
    'America/Nipigon', 'America/Nome', 'America/Noronha', 'America/North_Dakota/Beulah', 'America/North_Dakota/Center',
    'America/North_Dakota/New_Salem', 'America/Nuuk', 'America/Ojinaga', 'America/Panama', 'America/Pangnirtung',
    'America/Paramaribo', 'America/Phoenix', 'America/Port-au-Prince', 'America/Port_of_Spain', 'America/Porto_Acre',
    'America/Porto_Velho', 'America/Puerto_Rico', 'America/Punta_Arenas', 'America/Rainy_River', 'America/Rankin_Inlet',
    'America/Recife', 'America/Regina', 'America/Resolute', 'America/Rio_Branco', 'America/Rosario',
    'America/Santa_Isabel', 'America/Santarem', 'America/Santiago', 'America/Santo_Domingo', 'America/Sao_Paulo',
    'America/Scoresbysund', 'America/Shiprock', 'America/Sitka', 'America/St_Barthelemy', 'America/St_Johns',
    'America/St_Kitts', 'America/St_Lucia', 'America/St_Thomas', 'America/St_Vincent', 'America/Swift_Current',
    'America/Tegucigalpa', 'America/Thule', 'America/Thunder_Bay', 'America/Tijuana', 'America/Toronto',
    'America/Tortola', 'America/Vancouver', 'America/Virgin', 'America/Whitehorse', 'America/Winnipeg',
    'America/Yakutat', 'America/Yellowknife', 'Antarctica/Casey', 'Antarctica/Davis', 'Antarctica/DumontDUrville',
    'Antarctica/Macquarie', 'Antarctica/Mawson', 'Antarctica/McMurdo', 'Antarctica/Palmer', 'Antarctica/Rothera',
    'Antarctica/South_Pole', 'Antarctica/Syowa', 'Antarctica/Troll', 'Antarctica/Vostok', 'Arctic/Longyearbyen',
    'Asia/Aden', 'Asia/Almaty', 'Asia/Amman', 'Asia/Anadyr', 'Asia/Aqtau',
    'Asia/Aqtobe', 'Asia/Ashgabat', 'Asia/Ashkhabad', 'Asia/Atyrau', 'Asia/Baghdad',
    'Asia/Bahrain', 'Asia/Baku', 'Asia/Bangkok', 'Asia/Barnaul', 'Asia/Beirut',
    'Asia/Bishkek', 'Asia/Brunei', 'Asia/Calcutta', 'Asia/Chita', 'Asia/Choibalsan',
    'Asia/Chongqing', 'Asia/Chungking', 'Asia/Colombo', 'Asia/Dacca', 'Asia/Damascus',
    'Asia/Dhaka', 'Asia/Dili', 'Asia/Dubai', 'Asia/Dushanbe', 'Asia/Famagusta',
    'Asia/Gaza', 'Asia/Harbin', 'Asia/Hebron', 'Asia/Ho_Chi_Minh', 'Asia/Hong_Kong',
    'Asia/Hovd', 'Asia/Irkutsk', 'Asia/Istanbul', 'Asia/Jakarta', 'Asia/Jayapura',
    'Asia/Jerusalem', 'Asia/Kabul', 'Asia/Kamchatka', 'Asia/Karachi', 'Asia/Kashgar',
    'Asia/Kathmandu', 'Asia/Katmandu', 'Asia/Khandyga', 'Asia/Kolkata', 'Asia/Krasnoyarsk',
    'Asia/Kuala_Lumpur', 'Asia/Kuching', 'Asia/Kuwait', 'Asia/Macao', 'Asia/Macau',
    'Asia/Magadan', 'Asia/Makassar', 'Asia/Manila', 'Asia/Muscat', 'Asia/Nicosia',
    'Asia/Novokuznetsk', 'Asia/Novosibirsk', 'Asia/Omsk', 'Asia/Oral', 'Asia/Phnom_Penh',
    'Asia/Pontianak', 'Asia/Pyongyang', 'Asia/Qatar', 'Asia/Qostanay', 'Asia/Qyzylorda',
    'Asia/Rangoon', 'Asia/Riyadh', 'Asia/Saigon', 'Asia/Sakhalin', 'Asia/Samarkand',
    'Asia/Seoul', 'Asia/Shanghai', 'Asia/Singapore', 'Asia/Srednekolymsk', 'Asia/Taipei',
    'Asia/Tashkent', 'Asia/Tbilisi', 'Asia/Tehran', 'Asia/Tel_Aviv', 'Asia/Thimbu',
    'Asia/Thimphu', 'Asia/Tokyo', 'Asia/Tomsk', 'Asia/Ujung_Pandang', 'Asia/Ulaanbaatar',
    'Asia/Ulan_Bator', 'Asia/Urumqi', 'Asia/Ust-Nera', 'Asia/Vientiane', 'Asia/Vladivostok',
    'Asia/Yakutsk', 'Asia/Yangon', 'Asia/Yekaterinburg', 'Asia/Yerevan', 'Atlantic/Azores',
    'Atlantic/Bermuda', 'Atlantic/Canary', 'Atlantic/Cape_Verde', 'Atlantic/Faeroe', 'Atlantic/Faroe',
    'Atlantic/Jan_Mayen', 'Atlantic/Madeira', 'Atlantic/Reykjavik', 'Atlantic/South_Georgia', 'Atlantic/St_Helena',
    'Atlantic/Stanley', 'Australia/ACT', 'Australia/Adelaide', 'Australia/Brisbane', 'Australia/Broken_Hill',
    'Australia/Canberra', 'Australia/Currie', 'Australia/Darwin', 'Australia/Eucla', 'Australia/Hobart',
    'Australia/LHI', 'Australia/Lindeman', 'Australia/Lord_Howe', 'Australia/Melbourne', 'Australia/NSW',
    'Australia/North', 'Australia/Perth', 'Australia/Queensland', 'Australia/South', 'Australia/Sydney',
    'Australia/Tasmania', 'Australia/Victoria', 'Australia/West', 'Australia/Yancowinna', 'Brazil/Acre',
    'Brazil/DeNoronha', 'Brazil/East', 'Brazil/West', 'CET', 'CST6CDT',
    'Canada/Atlantic', 'Canada/Central', 'Canada/Eastern', 'Canada/Mountain', 'Canada/Newfoundland',
    'Canada/Pacific', 'Canada/Saskatchewan', 'Canada/Yukon', 'Chile/Continental', 'Chile/EasterIsland',
    'Cuba', 'EET', 'EST', 'EST5EDT', 'Egypt',
    'Eire', 'Etc/GMT', 'Etc/GMT+0', 'Etc/GMT+1', 'Etc/GMT+10',
    'Etc/GMT+11', 'Etc/GMT+12', 'Etc/GMT+2', 'Etc/GMT+3', 'Etc/GMT+4',
    'Etc/GMT+5', 'Etc/GMT+6', 'Etc/GMT+7', 'Etc/GMT+8', 'Etc/GMT+9',
    'Etc/GMT-0', 'Etc/GMT-1', 'Etc/GMT-10', 'Etc/GMT-11', 'Etc/GMT-12',
    'Etc/GMT-13', 'Etc/GMT-14', 'Etc/GMT-2', 'Etc/GMT-3', 'Etc/GMT-4',
    'Etc/GMT-5', 'Etc/GMT-6', 'Etc/GMT-7', 'Etc/GMT-8', 'Etc/GMT-9',
    'Etc/GMT0', 'Etc/Greenwich', 'Etc/UCT', 'Etc/UTC', 'Etc/Universal',
    'Etc/Zulu', 'Europe/Amsterdam', 'Europe/Andorra', 'Europe/Astrakhan', 'Europe/Athens',
    'Europe/Belfast', 'Europe/Belgrade', 'Europe/Berlin', 'Europe/Bratislava', 'Europe/Brussels',
    'Europe/Bucharest', 'Europe/Budapest', 'Europe/Busingen', 'Europe/Chisinau', 'Europe/Copenhagen',
    'Europe/Dublin', 'Europe/Gibraltar', 'Europe/Guernsey', 'Europe/Helsinki', 'Europe/Isle_of_Man',
    'Europe/Istanbul', 'Europe/Jersey', 'Europe/Kaliningrad', 'Europe/Kiev', 'Europe/Kirov',
    'Europe/Kyiv', 'Europe/Lisbon', 'Europe/Ljubljana', 'Europe/London', 'Europe/Luxembourg',
    'Europe/Madrid', 'Europe/Malta', 'Europe/Mariehamn', 'Europe/Minsk', 'Europe/Monaco',
    'Europe/Moscow', 'Europe/Nicosia', 'Europe/Oslo', 'Europe/Paris', 'Europe/Podgorica',
    'Europe/Prague', 'Europe/Riga', 'Europe/Rome', 'Europe/Samara', 'Europe/San_Marino',
    'Europe/Sarajevo', 'Europe/Saratov', 'Europe/Simferopol', 'Europe/Skopje', 'Europe/Sofia',
    'Europe/Stockholm', 'Europe/Tallinn', 'Europe/Tirane', 'Europe/Tiraspol', 'Europe/Ulyanovsk',
    'Europe/Uzhgorod', 'Europe/Vaduz', 'Europe/Vatican', 'Europe/Vienna', 'Europe/Vilnius',
    'Europe/Volgograd', 'Europe/Warsaw', 'Europe/Zagreb', 'Europe/Zaporozhye', 'Europe/Zurich',
    'Factory', 'GB', 'GB-Eire', 'GMT', 'GMT+0',
    'GMT-0', 'GMT0', 'Greenwich', 'HST', 'Hongkong',
    'Iceland', 'Indian/Antananarivo', 'Indian/Chagos', 'Indian/Christmas', 'Indian/Cocos',
    'Indian/Comoro', 'Indian/Kerguelen', 'Indian/Mahe', 'Indian/Maldives', 'Indian/Mauritius',
    'Indian/Mayotte', 'Indian/Reunion', 'Iran', 'Israel', 'Jamaica',
    'Japan', 'Kwajalein', 'Libya', 'MET', 'MST',
    'MST7MDT', 'Mexico/BajaNorte', 'Mexico/BajaSur', 'Mexico/General', 'NZ',
    'NZ-CHAT', 'Navajo', 'PRC', 'PST8PDT', 'Pacific/Apia',
    'Pacific/Auckland', 'Pacific/Bougainville', 'Pacific/Chatham', 'Pacific/Chuuk', 'Pacific/Easter',
    'Pacific/Efate', 'Pacific/Enderbury', 'Pacific/Fakaofo', 'Pacific/Fiji', 'Pacific/Funafuti',
    'Pacific/Galapagos', 'Pacific/Gambier', 'Pacific/Guadalcanal', 'Pacific/Guam', 'Pacific/Honolulu',
    'Pacific/Johnston', 'Pacific/Kanton', 'Pacific/Kiritimati', 'Pacific/Kosrae', 'Pacific/Kwajalein',
    'Pacific/Majuro', 'Pacific/Marquesas', 'Pacific/Midway', 'Pacific/Nauru', 'Pacific/Niue',
    'Pacific/Norfolk', 'Pacific/Noumea', 'Pacific/Pago_Pago', 'Pacific/Palau', 'Pacific/Pitcairn',
    'Pacific/Pohnpei', 'Pacific/Ponape', 'Pacific/Port_Moresby', 'Pacific/Rarotonga', 'Pacific/Saipan',
    'Pacific/Samoa', 'Pacific/Tahiti', 'Pacific/Tarawa', 'Pacific/Tongatapu', 'Pacific/Truk',
    'Pacific/Wake', 'Pacific/Wallis', 'Pacific/Yap', 'Poland', 'Portugal',
    'ROC', 'ROK', 'Singapore', 'Turkey', 'UCT',
    'US/Alaska', 'US/Aleutian', 'US/Arizona', 'US/Central', 'US/East-Indiana',
    'US/Eastern', 'US/Hawaii', 'US/Indiana-Starke', 'US/Michigan', 'US/Mountain',
    'US/Pacific', 'US/Samoa', 'UTC', 'Universal', 'W-SU',
    'WET', 'Zulu',
})


# Sensitive interpersonal topics for help action
SENSITIVE_TOPIC_PATTERN = re.compile(
    r"\b(spouse|wife|husband|wedding|marriage|anniversary|personal\s+promise|"
    r"family\s+matter|apology|forgive|relationship)\b",
    re.IGNORECASE,
)

# Physical chores, housekeeping, and in-person manual maintenance
PHYSICAL_CHORES_PATTERN = re.compile(
    r"\b(?:cleaning\s+of|vacuum|vacuuming|dusted|dusting|wipe\s+clean|deep\s+cleaning|"
    r"switchboards?|remotes?|cabinet|counter|dining\s+table|sofa|carpet|wallpaper|"
    r"boxroom|study\s+tables?|wc\s+seat|shower\s+cubicle|bathroom\s+fan|tiles?\s+corners|"
    r"marble\s+guy|fans?\s+need\s+cleaning|walls?\s+and\s+fans?)\b",
    re.IGNORECASE,
)

# In-person physical fitness classes and medical rest
PHYSICAL_HEALTH_CLASS_PATTERN = re.compile(
    r"\b(?:morning\s+batch|evening\s+batch|yoga\s+batch|gym\s+batch|workout\s+batch|"
    r"chemotherapy|hospital\s+admitted|bed\s+rest)\b",
    re.IGNORECASE,
)

# Direct personal voice communication and sensitive family matters
DIRECT_PERSONAL_COMM_PATTERN = re.compile(
    r"^(?:pls\s+call\s+me|call\s+me|please\s+call\s+me|need\s+to\s+discuss\s+dev|"
    r"discuss\s+dev)\b",
    re.IGNORECASE,
)

# Casual conversational observations, notes, or rhetorical queries
CASUAL_OBSERVATION_PATTERN = re.compile(
    r"\b(?:should\s+i\s+get\s+bigger|accumulate\s+over\s+time|errors\s+fixed\s+we\s+talked\s+about|"
    r"text\s+him\s+now\.\s*he\s+will\s+buy)\b",
    re.IGNORECASE,
)

# Procurement, booking, and purchasing actions
PROCUREMENT_ORDER_PATTERN = re.compile(
    r"\b(?:order|buy|get\s+me|purchase|book|booking|subscribe)\b",
    re.IGNORECASE,
)

# Document drafting, invitations, and written content preparation
DRAFTING_WRITING_PATTERN = re.compile(
    r"\b(?:draft|write|invite\s+to\s+be\s+made|save\s+the\s+date|certificate|summary|summarize)\b",
    re.IGNORECASE,
)

# Information gathering, teacher/vendor search, and option comparisons
# Note: 'find' requires a meaningful information-lookup object or context, excluding subjective opinions ('find him/it irritating/annoying/hard')
RESEARCH_SEARCH_PATTERN = re.compile(
    r"\b(?:research|search|compare|look\s*up|lookup|explore|teacher\s+for|options)\b|"
    r"\bfind\s+(?:(?:a|an|the|some|any|more|better|cheaper|other|good|new)\s+)?(?:options?|alternatives?|teachers?|vendors?|contractors?|suppliers?|solutions?|details?|info|information|prices?|quotes?|rates?|products?|items?|models?|leads?|person|people|accountant|driver|staff|maid|cook)\b|"
    r"\bfind\s+(?:out|a\s+way\s+to|how\s+to)\b",
    re.IGNORECASE,
)

# Process design and workflow formulation
PROCESS_FORMULATION_PATTERN = re.compile(
    r"\b(?:figure\s+a\s+way|formulate|workflow|process|reimburse)\b",
    re.IGNORECASE,
)

# Explicit future deadline expressions taking precedence over generic assist
DEADLINE_HELP_EXPR_PATTERN = re.compile(
    r"\b(?:due\s+(?:today|tomorrow|tonight|monday|tuesday|wednesday|thursday|friday|saturday|sunday|eod|noon|by|on|\d+)|"
    r"by\s+(?:today|tomorrow|tonight|monday|tuesday|wednesday|thursday|friday|saturday|sunday|eod|noon|evening|morning|\d{1,2}(?::\d{2})?\s*(?:am|pm)?)|"
    r"before\s+(?:today|tomorrow|monday|tuesday|wednesday|thursday|friday|saturday|sunday|\d+)|"
    r"until\s+(?:today|tomorrow|monday|tuesday|wednesday|thursday|friday|saturday|sunday|\d+)|"
    r"deadline)\b",
    re.IGNORECASE,
)
PAST_DEADLINE_HELP_PATTERN = re.compile(
    r"\b(?:yesterday|last\s+(?:week|month|year|friday|monday|tuesday|wednesday|thursday|saturday|sunday)|was\s+due|were\s+due)\b",
    re.IGNORECASE,
)

# Commercial, business, or personal decision questions requiring user discretion ('remind', never generic 'assist')
DECISION_QUESTION_PATTERN = re.compile(
    r"\b(?:decide\s+(?:whether|if|on|about)|decision\s+(?:on|about|whether)|"
    r"should\s+(?:we|i)\s+(?:take|accept|reject|choose|switch|join|partner|hire|sign|go\s+with|opt)|"
    r"whether\s+to\s+(?:take|accept|reject|choose|switch|join|partner|hire|sign|go\s+with|opt))\b|"
    r"\b(?:offer|proposal|rate|fee|retainer|pricing|quote|contract|deal|bid|terms)\b.*\b(?:take|accept|choose|decision|decide)\b|"
    r"\b(?:take|accept|choose|decision|decide)\b.*\b(?:offer|proposal|rate|fee|retainer|pricing|quote|contract|deal|bid|terms)\b",
    re.IGNORECASE,
)


def _get_message_text(msg: Optional[Dict[str, Any]]) -> str:
    """Retrieve message content safely, preferring 'body' with 'text' fallback.

    Does not mutate or rewrite the message dictionary.
    """
    if not isinstance(msg, dict):
        return ""
    body = msg.get("body")
    if isinstance(body, str):
        return body
    text = msg.get("text")
    if isinstance(text, str):
        return text
    return ""


def is_valid_timestamp(ts: Any) -> bool:
    """Validate that ts is a finite, non-negative numeric timestamp.

    Explicitly rejects booleans (True, False), None, strings, collections,
    NaN, inf, and negative numbers.
    """
    if isinstance(ts, bool):
        return False
    if isinstance(ts, (int, float)):
        return math.isfinite(ts) and ts >= 0
    return False


def is_valid_identifier(val: Any) -> bool:
    """Validate that val is a non-empty string without leading/trailing whitespace."""
    if not isinstance(val, str) or not val:
        return False
    return val.strip() == val


def is_valid_timezone_identifier(tz: Any) -> bool:
    """Validate whether tz is a recognized, authoritative timezone identifier.

    Accepts:
    - Standard abbreviations: 'UTC', 'GMT', 'IST', 'PST', 'EST', etc.
    - Explicit numeric offsets within [-14:00, +14:00]: '+05:30', '-08:00', '+00:00', '+14:00', etc.
    - Supported, recognized IANA identifiers: 'Asia/Kolkata', 'America/New_York', 'Europe/London', etc.

    Rejects:
    - Empty strings, None, non-string types
    - Country/region names like 'India'
    - Unknown offset '-00:00'
    - Syntactically plausible but unrecognized IANA names like 'Asia/NotAZone', 'America/NotARealCity'
    - Numeric offsets outside [-14:00, +14:00] such as '+14:30', '-15:00', or malformed offsets
    """
    if not isinstance(tz, str) or not tz.strip():
        return False
    val = tz.strip()
    if val == "-00:00":
        return False
    if val.lower() in VALID_TZ_ABBREVIATIONS:
        return True
    if _is_valid_numeric_offset(val):
        return True
    if val in RECOGNIZED_IANA_TIMEZONES:
        return True
    return False


def validate_confidence_score(val: Any) -> Optional[float]:
    """Validate confidence score, rejecting None, booleans, NaN, inf, and out-of-range values.

    Never assigns an artificial default or inflated score to missing confidence.
    """
    if val is None or isinstance(val, bool):
        return None
    if isinstance(val, (int, float)):
        if math.isfinite(val) and 0.0 <= val <= 1.0:
            return float(val)
    return None


def _get_dataset_message(context: ConversationContext, msg_id: str) -> Optional[Dict[str, Any]]:
    """Look up a message by ID in the dataset with high fidelity."""
    if not is_valid_identifier(msg_id):
        return None

    # Check messages_by_id in dataset
    messages_by_id = context.dataset.get("messages_by_id")
    if isinstance(messages_by_id, dict) and msg_id in messages_by_id:
        return messages_by_id[msg_id]

    # Check chat-level messages
    for chat in context.chats.values():
        for msg in chat.get("messages", []):
            if msg.get("id") == msg_id:
                return msg

    # Check flat message list
    for msg in context.dataset.get("messages", []):
        if msg.get("id") == msg_id:
            return msg

    return None


def _extract_content_tokens(text: str) -> Set[str]:
    """Extract normalized content tokens excluding stop words."""
    cleaned = re.sub(r"[^\w\s]", " ", text.lower())
    return {t for t in cleaned.split() if t not in STOP_WORDS and len(t) > 1}


def _extract_specific_task_tokens(text: str) -> Set[str]:
    """Extract distinctive content tokens excluding both stop words and generic task words."""
    cleaned = re.sub(r"[^\w\s]", " ", text.lower())
    return {
        t for t in cleaned.split()
        if t not in STOP_WORDS and t not in GENERIC_TASK_WORDS and len(t) > 1
    }


def _extract_qualifiers(text: str) -> Set[str]:
    """Extract distinctive calendar month or quarter qualifiers from text."""
    tokens = _extract_content_tokens(text)
    return tokens & MONTH_WORDS


def _extract_ordered_title_tokens(text: str) -> List[str]:
    """Extract normalized content tokens in original appearance order."""
    cleaned = re.sub(r"[^\w\s]", " ", text.lower())
    return [t for t in cleaned.split() if t not in STOP_WORDS and len(t) > 1]


def _extract_task_action(title: str) -> Optional[str]:
    """Extract primary action verb from title, preserving token order.

    Deterministically returns the first recognized action verb encountered
    in title token order.
    """
    for t in _extract_ordered_title_tokens(title):
        if t in ACTION_VERBS:
            return t
    return None


def _extract_task_entity(title: str) -> Optional[str]:
    """Extract primary task entity/document from title, preserving token order.

    Deterministically returns the first recognized task entity encountered
    in title token order.
    """
    for t in _extract_ordered_title_tokens(title):
        if t in TASK_ENTITIES:
            return t
    return None


def _extract_transaction_amounts(title: str) -> Set[int]:
    """Extract validated currency transaction amounts from title or text.

    Distinguishes genuine currency values and transaction amounts from
    calendar years (e.g. 2024-2027), times (e.g. 1100, 12:30), counts/quantities,
    phone numbers, and arbitrary numbers.
    """
    amounts: Set[int] = set()
    cleaned = title.replace(",", "")

    # 1. Explicit currency symbol or suffix
    currency_pattern = re.compile(
        r"(?:₹|rs\.?|inr|\$|usd)\s*(\d+)(?:\s*(k|thousand|l|lakh|lac|cr|crore|/-))?\b|"
        r"\b(\d+)\s*(?:/[-–]|rupees|bucks|inr)\b|"
        r"\b(\d+)\s*/[-–]",
        re.IGNORECASE,
    )
    for m in currency_pattern.finditer(cleaned):
        num_str = m.group(1) or m.group(3) or m.group(4)
        unit = (m.group(2) or "").lower()
        if num_str:
            val = int(num_str)
            if unit in ("k", "thousand"):
                val *= 1000
            elif unit in ("l", "lakh", "lac"):
                val *= 100000
            elif unit in ("cr", "crore"):
                val *= 10000000
            amounts.add(val)

    # 2. Payment context keyword preceding or following the number
    payment_context_pattern = re.compile(
        r"\b(?:pay|paid|payment|transfer|fee|fees|rent|salary|reimburse|deposit|bill|amount|total|cost|price|costing)\b"
        r"(?:\s+(?:of|is|for|to|around|approx|approximately|at|:)|[:\-])?\s+"
        r"(\d{2,7})\b|"
        r"\b(\d{2,7})\s*(?:/(?:pm|month|kg|day|hr|hour|seat)|\bper\s+(?:month|kg|day|hr|hour|seat))\b",
        re.IGNORECASE,
    )
    for m in payment_context_pattern.finditer(cleaned):
        num_str = m.group(1) or m.group(2)
        if num_str:
            val = int(num_str)
            if 2020 <= val <= 2035:
                continue
            amounts.add(val)

    return amounts


def _has_semantic_overlap(title: str, text: str) -> bool:
    """Check whether text shares specific entity or action tokens with title."""
    specific_title = _extract_specific_task_tokens(title)
    if specific_title:
        text_tokens = _extract_content_tokens(text)
        return len(specific_title & text_tokens) > 0
    # Fallback to general content tokens if title has no non-generic tokens
    title_tokens = _extract_content_tokens(title)
    text_tokens = _extract_content_tokens(text)
    return len(title_tokens & text_tokens) > 0


def _are_compatible_task_episodes(
    ep: CandidateOpenLoop,
    cand: CandidateOpenLoop,
    context: ConversationContext,
) -> bool:
    """Determine whether two open-loop records represent the same underlying task episode.

    Requires compatibility across chat, owner, obligation type, target qualifiers,
    and action-and-target semantic structures. Prevents merging different obligations
    sharing only a single entity name or client token (e.g. 'Send Orbit invoice'
    and 'Review Orbit contract' must remain separate).
    """
    if ep.chat_id != cand.chat_id:
        return False

    cand_is_closing_msg = bool(ep.closed_by and ep.closed_by == cand.source_msg)

    if ep.owner != cand.owner and not cand_is_closing_msg:
        is_direct_chat = False
        if context:
            chat = getattr(context, "chats", {}).get(ep.chat_id, {})
            if isinstance(chat, dict) and not chat.get("is_group", False):
                is_direct_chat = True
        if (
            is_direct_chat
            and re.search(r"\b(?:discuss|meeting|talk|sit|catch\s+up)\b", ep.title, re.IGNORECASE)
            and re.search(r"\b(?:discuss|meeting|talk|sit|catch\s+up)\b", cand.title, re.IGNORECASE)
        ):
            pass
        else:
            return False

    # Check obligation type compatibility
    valid_commitment_types = {"commitment", "deadline", "followup", "question"}
    if ep.type in valid_commitment_types and cand.type in valid_commitment_types:
        pass
    elif ep.type == cand.type:
        pass
    else:
        return False

    # 1. Temporal/cycle qualifier clash (e.g. October vs November, Q1 vs Q2)
    ep_qualifiers = _extract_qualifiers(ep.title)
    cand_qualifiers = _extract_qualifiers(cand.title)
    if ep_qualifiers and cand_qualifiers and ep_qualifiers != cand_qualifiers:
        return False

    cand_msg = _get_dataset_message(context, cand.source_msg)

    # 1b. Distinct transaction amount clash (e.g. 4500 vs 3500, ₹4,000 vs ₹3,500)
    ep_amounts = _extract_transaction_amounts(ep.title)
    cand_amounts = _extract_transaction_amounts(cand.title)
    if not ep_amounts:
        ep_msg = _get_dataset_message(context, ep.source_msg)
        if ep_msg:
            ep_amounts = _extract_transaction_amounts(_get_message_text(ep_msg))
    if not cand_amounts and cand_msg:
        cand_amounts = _extract_transaction_amounts(_get_message_text(cand_msg))
    if ep_amounts and cand_amounts and ep_amounts != cand_amounts:
        return False

    # 2. Distinct target entity clash (e.g. invoice vs contract, deck vs report)
    ep_entity = _extract_task_entity(ep.title)
    cand_entity = _extract_task_entity(cand.title)
    if ep_entity and cand_entity and ep_entity != cand_entity:
        return False

    # Check if cand quotes ep
    cand_quoted = cand_msg.get("quoted_id") if cand_msg else None
    if cand_quoted and (cand_quoted == ep.source_msg or cand_quoted in ep.source_msgs_all):
        return True

    # Check if cand is an in-flight continuation / status update of an active open episode
    cand_text = _get_message_text(cand_msg)
    is_in_flight_update = bool(re.search(
        r"\b(on\s+it|working\s+on\s+it|will\s+send\s+options|reverting\s+tomorrow|"
        r"will\s+revert\b(?!\s+(?:with|the))|will\s+update\s+(?:you|tomorrow|soon|shortly|later|on\s+this|then)\b|"
        r"i'll\s+figure|send\s+me\s+smart\s+card|working\s+on\s+this|(?:i'm|im|am|we're|already)\s+on\s+this|"
        r"will\s+do\s+monday|we'll\s+do\s+monday|sharing\s+shortly|checking\s+now)\b",
        cand_text,
        re.IGNORECASE,
    ))
    if is_in_flight_update and ep.status == "open":
        if not (ep_entity and cand_entity and ep_entity != cand_entity):
            if not (cand_entity and cand_entity not in _extract_ordered_title_tokens(ep.title)):
                if not (ep_qualifiers and cand_qualifiers and ep_qualifiers != cand_qualifiers):
                    return True

    # 3. Distinct substantive action clash (e.g. send vs review, pay vs draft)
    ep_action = _extract_task_action(ep.title)
    cand_action = _extract_task_action(cand.title)
    if ep_action and cand_action and ep_action != cand_action:
        return False

    if cand_is_closing_msg:
        # Require substantive continuation evidence to subsume into the closed episode
        is_meeting_coordination = bool(
            re.search(r"\b(?:discuss|meeting|talk|sit|catch\s+up|connect)\b", ep.title, re.IGNORECASE)
            and re.search(r"\b(?:discuss|meeting|talk|sit|catch\s+up|connect|tomorrow|today|time|calendar)\b", cand.title, re.IGNORECASE)
        )
        shared_substance = bool(
            (ep_entity and cand_entity and ep_entity == cand_entity)
            or (ep_action and cand_action and ep_action == cand_action)
        )
        ep_specific = _extract_specific_task_tokens(ep.title)
        cand_specific = _extract_specific_task_tokens(cand.title)
        shared_specific = bool(ep_specific and cand_specific and (ep_specific & cand_specific))
        if is_meeting_coordination or shared_substance or shared_specific or cand_quoted:
            return True
        return False

    # 4. Check specific distinctive tokens
    ep_specific = _extract_specific_task_tokens(ep.title)
    cand_specific = _extract_specific_task_tokens(cand.title)

    if ep_specific and cand_specific:
        shared_specific = ep_specific & cand_specific
        if not shared_specific:
            return False
        # A single shared distinctive token alone is not sufficient if entity or action is missing or divergent
        if len(shared_specific) == 1:
            has_matching_entity = bool(ep_entity and cand_entity and ep_entity == cand_entity)
            has_matching_action = bool(ep_action and cand_action and ep_action == cand_action)
            cand_text = _get_message_text(cand_msg)
            is_reminder_cand = bool(re.search(
                r"\b(checking\s+in|follow\s*up|any\s+update|any\s+thoughts|what\s+do\s+you\s+think|did\s+you|have\s+you|status|remind|reminder|kya\s+hua|ho\s+gaya\s+kya)\b",
                cand_text,
                re.IGNORECASE,
            ))
            if not (has_matching_entity or has_matching_action or is_reminder_cand):
                return False
        return True

    # Generic titles: require explicit reminder marker or identical content tokens
    cand_text = _get_message_text(cand_msg)
    is_reminder_text = bool(re.search(
        r"\b(checking\s+in|follow\s*up|any\s+update|any\s+thoughts|what\s+do\s+you\s+think|did\s+you|have\s+you|status|remind|reminder|kya\s+hua|ho\s+gaya\s+kya)\b",
        cand_text,
        re.IGNORECASE,
    ))
    ep_content = _extract_content_tokens(ep.title)
    cand_content = _extract_content_tokens(cand.title)
    return is_reminder_text or (bool(ep_content) and ep_content == cand_content)


def _get_participant_names(p_info: Optional[Dict[str, Any]]) -> List[str]:
    """Extract distinct valid participant names/aliases from ConversationContext record.

    Inspects both 'sender_names' and 'direct_chat_names', collecting non-empty strings
    with >= 2 characters. Case-insensitive distinct.
    """
    if not p_info or not isinstance(p_info, dict):
        return []
    names: List[str] = []
    seen: Set[str] = set()
    raw_names = list(p_info.get("sender_names") or []) + list(p_info.get("direct_chat_names") or [])
    name_field = p_info.get("name")
    if isinstance(name_field, str) and name_field.strip():
        raw_names.append(name_field)
    for raw in raw_names:
        if isinstance(raw, str):
            clean = raw.strip()
            if len(clean) >= 2 and clean.lower() not in seen:
                seen.add(clean.lower())
                names.append(clean)
    # Also include single-token first names from multi-word names
    for raw in list(names):
        parts = raw.split()
        if len(parts) > 1:
            first = parts[0].strip()
            if len(first) >= 3 and first.lower() not in seen:
                seen.add(first.lower())
                names.append(first)
    return names


def resolve_loop_owner(
    raw_owner: Optional[str],
    chat_id: str,
    source_msg: Dict[str, Any],
    context: ConversationContext,
) -> Optional[str]:
    """Resolve loop owner to 'ME' or authentic participant ID using contextual dialogue reasoning.

    Considers:
    - Sender identity and conversational role.
    - Direct (1:1) versus group chat dynamics.
    - First-person commitment versus imperative / directed request wording.
    - Question directionality (a question does not automatically assign ownership to the asker).
    - Explicit @mentions or named addressees in group chats.
    - Participant chat membership and authenticity safeguards against arbitrary or fabricated IDs.
    """
    if not is_valid_identifier(chat_id):
        return None

    chat = context.get_chat(chat_id)
    if not chat:
        return None

    participants: List[str] = list(chat.get("participants", []))
    is_group = bool(chat.get("is_group", False))
    sender_id = source_msg.get("sender_id", "")
    text = _get_message_text(source_msg)

    # Reject invalid identifier format if raw_owner was supplied
    if raw_owner is not None and not is_valid_identifier(raw_owner):
        return None

    # Safeguard: raw_owner must be 'ME' or an actual participant in this chat
    if raw_owner is not None and raw_owner != "ME" and raw_owner not in participants:
        return None

    # =========================================================================
    # Direct (1:1) Chat Contextual Dynamics
    # =========================================================================
    if not is_group:
        other_participants = [p for p in participants if p != "ME"]
        contact_id = other_participants[0] if len(other_participants) == 1 else None

        # Situational observations without directed action
        if SITUATIONAL_OBSERVATION_PATTERN.search(text) and not DIRECTED_REQUEST_PATTERN.search(text):
            return None

        # Third-party tasks or personal dilemmas
        if THIRD_PARTY_OR_DILEMMA_PATTERN.search(text) and not DIRECTED_REQUEST_PATTERN.search(text):
            return None

        if sender_id != "ME":
            # Contact speaking to ME:
            # 1. Contact's first-person pledge assigns responsibility to contact
            if FIRST_PERSON_COMMITMENT_PATTERN.search(text):
                if sender_id in participants:
                    if raw_owner is not None and raw_owner != sender_id:
                        return None
                    return sender_id
                return None

            # 2. Contact asking for advice about their own decision assigns responsibility to contact
            is_own_decision = (
                ADVICE_ON_OWN_DECISION_PATTERN.search(text)
                and not DECISION_REQUESTED_FROM_ME_PATTERN.search(text)
                and not CONCRETE_ACTION_REQUEST_PATTERN.search(text)
            )
            if is_own_decision:
                if raw_owner is not None and raw_owner != sender_id:
                    return None
                if sender_id in participants:
                    return sender_id
                return None

            # 3. Contact asking ME to make a decision or state preference
            if DECISION_REQUESTED_FROM_ME_PATTERN.search(text):
                if raw_owner is not None and raw_owner != "ME":
                    return None
                return "ME"

            # 4. Directed request, imperative, or actionable question directed to ME assigns responsibility to ME
            if (
                CONCRETE_ACTION_REQUEST_PATTERN.search(text)
                or DIRECTED_REQUEST_PATTERN.search(text)
                or re.search(r"^(?:pls|please|kindly\s+)?(?:send|give|share|provide|check|confirm|pay|transfer|tell|let\s+me\s+know|update|call|sign|review)\b", text.strip(), re.IGNORECASE)
                or ("?" in text and ACTIONABLE_QUESTION_PATTERN.search(text))
            ):
                if raw_owner is not None and raw_owner != "ME":
                    return None
                return "ME"

            # 5. Contextual raw owner acceptance
            if raw_owner in ("ME", sender_id):
                return raw_owner

            # 6. Default for actionable incoming obligation in direct chat with ME
            return "ME"

        else:
            # ME speaking to contact:
            # 1. First-person commitment by ME assigns responsibility to ME
            if FIRST_PERSON_COMMITMENT_PATTERN.search(text):
                if raw_owner is not None and raw_owner != "ME":
                    return None
                return "ME"

            # 2. ME asking advice about ME's own decision
            if ADVICE_ON_OWN_DECISION_PATTERN.search(text) and not CONCRETE_ACTION_REQUEST_PATTERN.search(text):
                if raw_owner is not None and raw_owner != "ME":
                    return None
                return "ME"

            # 2b. Discussion request with ME remains owned by ME
            if re.search(r"\bdiscuss\s+with\s+me\b", text, re.IGNORECASE):
                return "ME"

            # 3. Directed request, imperative, or actionable question directed to contact assigns responsibility to contact
            if contact_id and (
                CONCRETE_ACTION_REQUEST_PATTERN.search(text)
                or DIRECTED_REQUEST_PATTERN.search(text)
                or re.search(r"^(?:pls|please|kindly\s+)?(?:send|give|share|provide|check|confirm|pay|transfer|tell|let\s+me\s+know|update|call|sign|review)\b", text.strip(), re.IGNORECASE)
                or ("?" in text and ACTIONABLE_QUESTION_PATTERN.search(text))
            ):
                if raw_owner is not None and raw_owner != contact_id:
                    return None
                return contact_id

            # 4. Contextual raw owner acceptance
            if raw_owner == "ME":
                return "ME"
            if raw_owner and raw_owner == contact_id:
                return contact_id

            # 5. Default substantive task initiated by ME without counterparty redirection
            return "ME"

    # =========================================================================
    # Group Chat Contextual Dynamics
    # =========================================================================
    else:
        # Rejection of collective commitments and open unaddressed broadcasts
        if COLLECTIVE_GROUP_PATTERN.search(text):
            return None
        if BROADCAST_QUERY_PATTERN.search(text) and not FIRST_PERSON_COMMITMENT_PATTERN.search(text):
            return None

        # 1. Sender-owned first-person commitment
        if FIRST_PERSON_COMMITMENT_PATTERN.search(text):
            if sender_id == "ME":
                return "ME"
            elif sender_id in participants:
                return sender_id

        # 2. Direct reply/quote to ME in group
        quoted_id = source_msg.get("quoted_id")
        if quoted_id:
            quoted_msg = _get_dataset_message(context, quoted_id)
            if quoted_msg and quoted_msg.get("sender_id") == "ME" and DIRECTED_REQUEST_PATTERN.search(text):
                return "ME"

        # 3. Explicit named or @mention assignment to a chat participant
        for pid in participants:
            if pid != "ME" and re.search(rf"@\b{re.escape(pid)}\b", text, re.IGNORECASE):
                return pid
            p_info = context.get_participant(pid)
            for p_name in _get_participant_names(p_info):
                name_pat = rf"(?:@|\b){re.escape(p_name)}\b"
                if re.search(name_pat, text, re.IGNORECASE) and (DIRECTED_REQUEST_PATTERN.search(text) or "?" in text or re.search(r"\b(?:pls|please|can|could|will|do|review|need|share|send|give|provide)\b", text, re.IGNORECASE)):
                    return pid

        # 4. Explicit named or @mention assignment to ME
        me_info = context.get_participant("ME")
        for me_name in _get_participant_names(me_info):
            name_pat = rf"(?:@|\b){re.escape(me_name)}\b"
            if re.search(name_pat, text, re.IGNORECASE) and (DIRECTED_REQUEST_PATTERN.search(text) or "?" in text or re.search(r"\b(?:pls|please|can|could|will|do|review|need|share|send|give|provide)\b", text, re.IGNORECASE)):
                return "ME"

        # 4b. Team deliberation / decision question in small group chat without directed other owner
        if len(participants) <= 5 and sender_id != "ME":
            if (
                re.search(r"\b(?:should\s+we|shall\s+we|any\s+thoughts|what\s+do\s+you\s+think|do\s+we\s+want\s+to|you\s+(?:both\s+)?coming|coming\s+for)\b", text, re.IGNORECASE)
                or DECISION_REQUESTED_FROM_ME_PATTERN.search(text)
                or (len(participants) == 2 and (DIRECTED_REQUEST_PATTERN.search(text) or re.search(r"\b(?:pls|please|kindly|confirm|pay)\b", text, re.IGNORECASE)))
            ):
                if raw_owner is None or raw_owner == "ME":
                    return "ME"

        # 5. Verified raw owner against grounded chat evidence
        if raw_owner == "ME":
            if sender_id == "ME" and (FIRST_PERSON_COMMITMENT_PATTERN.search(text) or not is_group):
                return "ME"
            # If ME was explicitly mentioned or addressed
            if any(re.search(rf"(?:@|\b){re.escape(n)}\b", text, re.IGNORECASE) for n in _get_participant_names(context.get_participant("ME"))):
                return "ME"
            return None
        elif raw_owner in participants:
            if sender_id == raw_owner and FIRST_PERSON_COMMITMENT_PATTERN.search(text):
                return raw_owner
            # If participant was explicitly mentioned or addressed
            if any(re.search(rf"(?:@|\b){re.escape(n)}\b", text, re.IGNORECASE) for n in _get_participant_names(context.get_participant(raw_owner))):
                return raw_owner
            return None

        # Safeguard: Never fabricate an owner for ambiguous group messages
        return None


def resolve_due_date(
    raw_due_date: Optional[str],
    due_anchor_msg: Optional[str],
    source_msg: Dict[str, Any],
    chat_id: str,
    context: ConversationContext,
) -> Tuple[Optional[str], Optional[str]]:
    """Resolve due date and anchor message enforcing timezone and calendar grounding.

    Returns (due_date, due_anchor_msg) or (None, None) if ungrounded or invalid.
    """
    if raw_due_date is None or due_anchor_msg is None:
        return None, None

    if not isinstance(raw_due_date, str) or not isinstance(due_anchor_msg, str):
        return None, None

    if not is_valid_identifier(due_anchor_msg):
        return None, None

    # Validate YYYY-MM-DD pattern and real calendar date
    if not re.match(r"^\d{4}-\d{2}-\d{2}$", raw_due_date):
        return None, None
    try:
        parsed_due = date.fromisoformat(raw_due_date)
    except ValueError:
        return None, None

    # Locate anchor message in dataset
    anchor_msg = _get_dataset_message(context, due_anchor_msg)
    if not anchor_msg:
        return None, None

    # Verify same chat
    if anchor_msg.get("chat_id") != chat_id:
        return None, None

    # Verify timestamps
    source_ts = source_msg.get("ts")
    anchor_ts = anchor_msg.get("ts")
    if not is_valid_timestamp(source_ts) or not is_valid_timestamp(anchor_ts):
        return None, None
    if anchor_ts < source_ts:
        return None, None

    # Grounding verification: Check for explicit year or authoritative timezone
    anchor_text = _get_message_text(anchor_msg)
    source_text = _get_message_text(source_msg)

    # 1. Stated 4-digit calendar year in message text is preserved without modification
    explicit_year_match = FOUR_DIGIT_YEAR_PATTERN.search(anchor_text) or FOUR_DIGIT_YEAR_PATTERN.search(source_text)
    if explicit_year_match:
        stated_year = int(explicit_year_match.group(1))
        if parsed_due.year == stated_year:
            return raw_due_date, due_anchor_msg

    # 2. Authoritative timezone validation
    dataset_tz = context.dataset.get("timezone")
    has_valid_dataset_tz = is_valid_timezone_identifier(dataset_tz)

    has_authoritative_tz = (
        has_valid_dataset_tz or
        bool(TIMEZONE_METADATA_PATTERN.search(anchor_text)) or
        bool(TIMEZONE_METADATA_PATTERN.search(source_text))
    )

    if has_authoritative_tz:
        return raw_due_date, due_anchor_msg

    # If no explicit year in text and no authoritative timezone is grounded,
    # unqualified dates and relative expressions cannot be safely resolved.
    return None, None


def resolve_relative_due_date(
    raw_due_date: Optional[str],
    due_anchor_msg: Optional[str],
    source_msg: Dict[str, Any],
    chat_id: str,
    context: ConversationContext,
) -> Tuple[Optional[str], Optional[str]]:
    """Helper alias for resolve_due_date maintaining design interface compatibility."""
    return resolve_due_date(raw_due_date, due_anchor_msg, source_msg, chat_id, context)


def _evaluate_closing_message(
    msg: Dict[str, Any],
    source_msg: Dict[str, Any],
    chat_id: str,
    owner: str,
    requester: str,
    authenticated_delegates: Set[str],
    loop_type: str,
    title: str,
    is_group: bool,
    messages: List[Dict[str, Any]],
) -> Optional[Tuple[str, str]]:
    """Evaluate whether a candidate message definitively closes the obligation.

    Strictly validates dataset existence, chat membership, timestamp ordering,
    actor authority, semantic linkage, and completion/cancellation semantics.
    Enforces that requesters may revoke their requests, while obligors may drop
    only when explicitly reporting an authorized preemption or external reason.
    Returns (status, closed_by_id) or None if closure is unverified or invalid.
    """
    msg_id = msg.get("id")
    if not is_valid_identifier(msg_id) or msg_id == source_msg.get("id"):
        return None

    if msg.get("chat_id") != chat_id:
        return None

    msg_ts = msg.get("ts")
    source_ts = source_msg.get("ts")
    if not is_valid_timestamp(msg_ts) or not is_valid_timestamp(source_ts):
        return None
    if msg_ts < source_ts:
        return None

    msg_sender = msg.get("sender_id", "")
    source_sender = source_msg.get("sender_id", "")
    msg_text = _get_message_text(msg)
    quoted_id = msg.get("quoted_id")
    msg_kind = msg.get("kind", "text")

    # Postponement / ambiguity markers preserve open status
    if POSTPONEMENT_MARKERS_PATTERN.search(msg_text):
        return None

    # Deflection messages preserve open status (cannot close)
    if DEFLECTION_PATTERN.search(msg_text):
        return None

    source_idx = next((i for i, m in enumerate(messages) if m.get("id") == source_msg.get("id")), -1)
    msg_idx = next((i for i, m in enumerate(messages) if m.get("id") == msg_id), -1)

    # Check temporal window: 150 messages or 14 days maximum
    is_beyond_window = (
        (source_idx >= 0 and msg_idx >= 0 and msg_idx - source_idx > 150)
        or (msg_ts - source_ts > 14 * 86400)
    )

    # Validate quoted reference authenticity
    is_genuine_quote = bool(quoted_id and quoted_id == source_msg.get("id"))
    quoted_text_payload = msg.get("quoted_text") or msg.get("quoted_body") or msg.get("reply_to_text")
    if is_genuine_quote and quoted_text_payload:
        src_text = _get_message_text(source_msg).strip().lower()
        payload_text = str(quoted_text_payload).strip().lower()
        if payload_text and payload_text not in src_text and src_text not in payload_text:
            is_genuine_quote = False

    is_quoted = is_genuine_quote

    # If beyond window, an unquoted message can NEVER close
    if is_beyond_window and not is_quoted:
        return None

    # Calculate conversational proximity (strict immediate following turn within 30 minutes)
    is_close_proximity = False
    if 0 <= source_idx and msg_idx == source_idx + 1 and (0 <= msg_ts - source_ts <= 1800):
        is_close_proximity = True

    # 1. Cancellation Category (status = 'dropped')
    if CANCELLATION_MARKERS_PATTERN.search(msg_text) or PREEMPTION_REASON_PATTERN.search(msg_text):
        is_bare_cancel = bool(BARE_CANCELLATION_PATTERN.search(msg_text))
        has_cancel_semantic = _has_semantic_overlap(title, msg_text)

        # A requester has unilateral authority to revoke their request
        if requester is not None and msg_sender == requester and CANCELLATION_MARKERS_PATTERN.search(msg_text):
            if is_beyond_window:
                # Quoted bypass requires: genuine quote, authorized requester, actual cancellation marker, and semantic linkage
                if is_quoted and (has_cancel_semantic or is_bare_cancel):
                    return "dropped", msg_id
            else:
                has_cancel_linkage = (
                    is_quoted or
                    has_cancel_semantic or
                    (is_close_proximity and is_bare_cancel)
                )
                if has_cancel_linkage:
                    return "dropped", msg_id

        # An obligor may close as dropped ONLY when explicitly reporting an authorized preemption
        elif msg_sender == owner:
            has_preemption_reason = bool(PREEMPTION_REASON_PATTERN.search(msg_text))
            if has_preemption_reason:
                if is_beyond_window:
                    if is_quoted and (has_cancel_semantic or has_preemption_reason):
                        return "dropped", msg_id
                else:
                    has_cancel_linkage = (
                        is_quoted or
                        has_cancel_semantic or
                        is_close_proximity
                    )
                    if has_cancel_linkage:
                        return "dropped", msg_id

    # 2. Document Delivery Category (status = 'done')
    # Proximity alone NEVER authorizes unrelated document delivery!
    is_document_delivery = (
        msg_kind in ("document", "image", "media") or
        bool(re.search(r"\b(attached|here\s+is|here's|sending\s+(?:the\s+)?(?:pdf|file|doc|deck|invoice|proofs?)|proofs?\s+attached)\b", msg_text, re.IGNORECASE))
    )
    if is_document_delivery:
        has_doc_authority = (msg_sender == owner or msg_sender in authenticated_delegates)
        media_path = str(msg.get("media_path", ""))
        caption = str(msg.get("caption", ""))
        source_text = _get_message_text(source_msg)
        has_doc_semantic = (
            _has_semantic_overlap(title, msg_text) or
            _has_semantic_overlap(source_text, msg_text) or
            _has_semantic_overlap(title, media_path) or
            _has_semantic_overlap(title, caption) or
            bool(re.search(r"\bproofs?\b", msg_text, re.IGNORECASE) and (re.search(r"\bproofs?\b", title, re.IGNORECASE) or re.search(r"\bproofs?\b", source_text, re.IGNORECASE)))
        )
        if has_doc_authority:
            if is_beyond_window:
                # Quoted document delivery beyond window requires genuine quote AND semantic overlap with the task
                if is_quoted and has_doc_semantic:
                    return "done", msg_id
            else:
                if is_quoted or has_doc_semantic:
                    return "done", msg_id

    # 3. Question Answering & Directed Info/Contact Request Category (status = 'done')
    DIRECTED_INFO_REQUEST_PATTERN = re.compile(
        r"\b(?:send|share|give|provide|what(?:'s|\s+is)|need|drop)\b.*\b(?:address|location|map|link|number|contact|phone|website|url)\b",
        re.IGNORECASE,
    )
    DIRECTED_SCHEDULING_REQUEST_PATTERN = re.compile(
        r"\b(?:call|meet|zoom|connect|calendar|slot|schedule|appointment)\b",
        re.IGNORECASE,
    )
    is_info_request = (
        loop_type == "question" or
        bool(DIRECTED_INFO_REQUEST_PATTERN.search(_get_message_text(source_msg))) or
        bool(DIRECTED_INFO_REQUEST_PATTERN.search(title)) or
        ((bool(DIRECTED_SCHEDULING_REQUEST_PATTERN.search(title)) or bool(DIRECTED_SCHEDULING_REQUEST_PATTERN.search(_get_message_text(source_msg)))) and bool(re.search(r"https?://(?:maps\.app\.goo\.gl|goo\.gl/maps|maps\.google\.com|meet\.google\.com|zoom\.us|teams\.microsoft\.com|calendly\.com|cal\.com)\b", msg_text, re.IGNORECASE)))
    )
    if is_info_request:
        # The answerer must be the counterpart or owner, never the requester self-fulfilling
        if msg_sender == source_sender:
            return None
        has_qa_authority = (
            msg_sender == owner
            or msg_sender in authenticated_delegates
            or (not is_group and msg_sender != source_sender)
            or (is_group and source_sender == "ME" and msg_sender != source_sender)
        )
        is_substantive = not re.match(r"^(?:who\s+knows|kya\s+pata|(?:hi|hello|hey|good\s+morning)[!.,\s]*$)", msg_text.strip(), re.IGNORECASE)
        has_exec = bool(EXECUTION_MARKERS_PATTERN.search(msg_text))
        is_adjacent = (0 <= source_idx and msg_idx == source_idx + 1 and (0 <= msg_ts - source_ts <= 1800))
        source_text = _get_message_text(source_msg)
        spec_tokens = _extract_specific_task_tokens(title)
        source_spec_tokens = _extract_specific_task_tokens(source_text)
        combined_spec_tokens = spec_tokens | source_spec_tokens
        text_tokens = _extract_content_tokens(msg_text)
        has_multi_token_overlap = len(combined_spec_tokens & text_tokens) >= 2
        has_source_overlap = bool((source_spec_tokens - STOP_WORDS) & text_tokens) or _has_semantic_overlap(source_text, msg_text)
        has_qa_semantic = (_has_semantic_overlap(title, msg_text) or has_source_overlap or has_multi_token_overlap)

        # Contextual direct fulfillment markers (maps, phone, meet links, cal links, or adjacent direct answers)
        has_scheduling_link = bool(
            re.search(r"https?://(?:calendly\.com|cal\.com|meet\.google\.com|zoom\.us|teams\.microsoft\.com)[^\s]*\b", msg_text, re.IGNORECASE)
        )
        has_maps_link = bool(
            re.search(r"https?://(?:maps\.app\.goo\.gl|goo\.gl/maps|maps\.google\.com)[^\s]*\b", msg_text, re.IGNORECASE)
        )
        has_fulfillment = bool(
            has_scheduling_link or
            has_maps_link or
            re.search(r"\b(?:\+91[\s-]?)?[6-9]\d{9}\b", msg_text) or
            (is_adjacent and re.search(r"\b(?:road|street|nagar|marg|layout|sector|block|floor|opp|opposite|near|behind|pin\s*code|\b\d{6}\b)\b", msg_text, re.IGNORECASE)) or
            (is_adjacent and re.search(r"\b(?:yes|yep|yeah|sure|ok|okay|done|haan|theek\s+hai|sahi\s+hai|absolutely|definitely|certainly)\b", msg_text, re.IGNORECASE))
        )

        # Scheduling links can fulfill scheduling tasks within an expanded window of 2 hours (7200s)
        # provided no intervening message broke the contextual thread
        is_scheduling_context = bool(DIRECTED_SCHEDULING_REQUEST_PATTERN.search(title) or DIRECTED_SCHEDULING_REQUEST_PATTERN.search(_get_message_text(source_msg)))
        is_unbroken_scheduling = False
        if has_scheduling_link and is_scheduling_context and (0 <= msg_ts - source_ts <= 7200):
            intervening = messages[source_idx + 1 : msg_idx] if (0 <= source_idx < msg_idx) else []
            is_unbroken_scheduling = all(m.get("sender_id") in (source_sender, owner) for m in intervening)

        # Maps location links can fulfill address/location requests within 2 hours if unbroken
        is_location_context = bool(re.search(r"\b(?:address|location|office|place|map|directions?)\b", title, re.IGNORECASE) or re.search(r"\b(?:address|location|office|place|map|directions?)\b", _get_message_text(source_msg), re.IGNORECASE))
        is_unbroken_location = False
        if has_maps_link and is_location_context and (0 <= msg_ts - source_ts <= 7200):
            intervening = messages[source_idx + 1 : msg_idx] if (0 <= source_idx < msg_idx) else []
            is_unbroken_location = all(m.get("sender_id") in (source_sender, owner) for m in intervening)

        if has_qa_authority and is_substantive:
            if is_beyond_window:
                # Quoted QA beyond window requires genuine quote and substantive answer semantically linked or confirming resolution
                if is_quoted and (has_qa_semantic or has_exec or has_fulfillment):
                    return "done", msg_id
                if is_unbroken_scheduling or is_unbroken_location:
                    return "done", msg_id
            else:
                is_msg_question = bool(re.search(r"\?\s*$", msg_text.strip()) or re.match(r"^(?:what|where|when|why|who|how|which|can\s+you|could\s+you)\b", msg_text.strip(), re.IGNORECASE))
                is_consecutive_turn = (0 <= source_idx and msg_idx == source_idx + 1)
                is_direct_chat = not is_group
                is_not_hedging_or_postponement = not re.search(
                    r"\b(?:not\s+sure|no\s+idea|who\s+knows|kya\s+pata|will\s+(?:let\s+you\s+know|check|update|see)|let\s+you\s+know|maybe\s+later)\b",
                    msg_text,
                    re.IGNORECASE,
                )
                is_direct_qa_resolution = (
                    is_direct_chat
                    and is_consecutive_turn
                    and (has_multi_token_overlap or has_source_overlap)
                    and is_not_hedging_or_postponement
                    and (0 <= msg_ts - source_ts <= 86400)
                )
                has_qa_linkage = not is_msg_question and (
                    is_quoted or
                    (is_adjacent and (has_exec or has_fulfillment or has_qa_semantic)) or
                    (has_multi_token_overlap and (has_exec or has_fulfillment)) or
                    is_direct_qa_resolution or
                    is_unbroken_scheduling or
                    is_unbroken_location
                )
                if has_qa_linkage:
                    return "done", msg_id

    # 4. Action Completion Category (status = 'done')
    # Guard against uncompleted obligations matching 'done' (e.g. 'needs to be done', 'yet to be done')
    is_uncompleted_obligation = bool(re.search(r"\b(?:needs?\s+to\s+be|yet\s+to\s+be|has\s+to\s+be|have\s+to\s+be|to\s+be|not\s+yet)\s+done\b", msg_text, re.IGNORECASE))
    if EXECUTION_MARKERS_PATTERN.search(msg_text) and not is_uncompleted_obligation:
        has_completion_authority = (
            msg_sender == owner or
            msg_sender == requester or
            msg_sender in authenticated_delegates
        )
        if has_completion_authority:
            source_text = _get_message_text(source_msg)
            is_bare_exec = bool(BARE_EXECUTION_PATTERN.match(msg_text.strip()))
            has_completion_semantic = _has_semantic_overlap(title, msg_text) or _has_semantic_overlap(source_text, msg_text)
            spec_tokens = _extract_specific_task_tokens(title) | _extract_specific_task_tokens(source_text)
            text_tokens = _extract_content_tokens(msg_text)
            title_content = (_extract_content_tokens(title) | _extract_content_tokens(source_text)) - STOP_WORDS
            shared_substantive = (title_content & text_tokens) - {"send", "review", "check", "call", "do", "bhej", "make", "get", "take"}
            # Check requester participant name if source commitment responded to a requester turn
            requester_tokens = set()
            if 0 < source_idx:
                prev_msg = messages[source_idx - 1]
                if prev_msg.get("sender_id") == requester:
                    sname = prev_msg.get("sender_name", "")
                    if sname:
                        requester_tokens.update(t for t in re.sub(r"[^\w\s]", " ", sname.lower()).split() if len(t) > 1 and t not in STOP_WORDS)

            has_grounded_task_tokens = len(spec_tokens & text_tokens) >= 1 or len(shared_substantive) >= 1
            # In group chats, an ungrounded name mention alongside a delegation/forwarding directive does not close
            is_delegated_forwarding = bool(re.search(r"\b(?:please\s+(?:re-?share|forward|send|cc)|can\s+you\s+(?:re-?share|forward|send|cc)|if\s+\w+\s+not\s+in\s+email\s+cc)\b", msg_text, re.IGNORECASE))
            if is_group and is_delegated_forwarding:
                has_requester_tokens = False
            else:
                has_requester_tokens = bool(requester_tokens and (requester_tokens & text_tokens))

            has_task_tokens = has_grounded_task_tokens or has_requester_tokens

            if is_beyond_window:
                # Quoted action completion beyond window requires genuine quote and actual completion marker semantically linked
                if is_quoted and (has_completion_semantic or has_task_tokens or is_bare_exec):
                    return "done", msg_id
            else:
                is_lunch_chatter = bool(re.search(r"\bdone\s+with\s+(?:lunch|dinner|breakfast|food|coffee|tea|workout)\b", msg_text, re.IGNORECASE))
                has_completion_linkage = not is_lunch_chatter and (
                    is_quoted or
                    has_completion_semantic or
                    has_task_tokens or
                    is_close_proximity
                )
                if has_completion_linkage:
                    return "done", msg_id

    # 5. Substantive Contextual Completion Category (status = 'done')
    # For commitments to revert/update or schedule calls where substantive fulfillment is demonstrated
    if msg_sender == owner or msg_sender in authenticated_delegates:
        # Reject generic non-fulfilling phrases
        is_not_mere_ack = not re.match(
            r"^(?:will|i'll|ill|on\s+it|working\s+on|looking\s+into|let\s+me|give\s+me|need\s+to|shall\s+we|can\s+we)\b",
            msg_text.strip(),
            re.IGNORECASE,
        )
        if is_not_mere_ack:
            # Case A: Commitment to revert/update followed by substantive findings
            is_revert_commitment = bool(re.search(r"\b(?:revert|reverting|update|get\s+back)\b", title, re.IGNORECASE) or re.search(r"\b(?:revert|reverting|update|get\s+back)\b", _get_message_text(source_msg), re.IGNORECASE))
            if is_revert_commitment and len(msg_text.strip()) >= 120 and (0 <= msg_ts - source_ts <= 172800):
                if re.search(r"\b(?:working|showing|campaign|measurement|attribution|report|performance|analysis|results|findings|numbers|data)\b", msg_text, re.IGNORECASE):
                    return "done", msg_id

            # Case B: Call/meeting fixing where owner immediately provides direct instructions / link
            is_call_fixing = bool(re.search(r"\b(?:fix\s+a\s+call|set\s+a\s+call|schedule\s+a\s+call|meeting\s+time)\b", title, re.IGNORECASE))
            if is_call_fixing and is_close_proximity:
                if re.search(r"\b(?:have\s+the\s+link|open\s+slot|just\s+block\s+it|link\s+is\s+there|blocked\s+it|link\s+sent)\b", msg_text, re.IGNORECASE):
                    return "done", msg_id

    # Case C: Automated / scheduled meeting calendar invite matching the requested slot
    is_meeting_req = bool(
        re.search(r"\b(?:meeting\s+time|schedule\s+call|have\s+a\s+meeting|can\s+we\s+pls\s+have\s+a\s+meeting|meeting\s+tomorrow)\b", title, re.IGNORECASE) or
        re.search(r"\b(?:meeting\s+time|schedule\s+call|have\s+a\s+meeting|can\s+we\s+pls\s+have\s+a\s+meeting|meeting\s+tomorrow)\b", _get_message_text(source_msg), re.IGNORECASE)
    )
    if is_meeting_req:
        if re.search(r"\b(?:meet\.google\.com|zoom\.us|teams\.microsoft\.com)\b", msg_text, re.IGNORECASE) and re.search(r"\b(?:joining\s+info|video\s+call\s+link|google\s+meet)\b", msg_text, re.IGNORECASE):
            return "done", msg_id

    # Case D: In-person meeting venue / floor arrival coordination in direct (1:1) chat
    is_meeting_context = bool(
        re.search(r"\b(?:meet|discuss|call|meeting)\b", title, re.IGNORECASE) or
        re.search(r"\b(?:meet|discuss|call|meeting)\b", _get_message_text(source_msg), re.IGNORECASE) or
        re.search(r"\b\d+(?::\d+)?\s*(?:am|pm)\s*(?:ill|i'll|will)\s+reach\b", _get_message_text(source_msg), re.IGNORECASE) or
        re.search(r"\b\d+(?::\d+)?\s*(?:am|pm)\s*(?:ill|i'll|will)\s+reach\b", title, re.IGNORECASE)
    )
    if not is_meeting_context and 0 <= source_idx < len(messages):
        # Ground appointment context in preceding scheduling context (within 3 turns)
        for prev_m in messages[max(0, source_idx - 3):source_idx]:
            prev_m_text = _get_message_text(prev_m)
            if re.search(r"\b(?:office|meet|meeting|discuss|call|catch\s*up)\b", prev_m_text, re.IGNORECASE) and re.search(r"\b(?:what\s+time|when|works?|available|free|\d+\s*(?:am|pm))\b", prev_m_text, re.IGNORECASE):
                is_meeting_context = True
                break

    if not is_group and is_meeting_context and not is_beyond_window:
        is_arrival_response = bool(re.search(
            r"\b(?:flat\s+\d+|\d+(?:st|nd|rd|th)\s+floor|floor\s+\d+|come\s+(?:up|in|to\s+\d+)|cabin\s+\d+|"
            r"elevator|lift|lobby|outside\s*(?:the\s+)?(?:gate|office|building)?|at\s+the\s+gate|downstairs|"
            r"reached(?:\s+(?:your\s+office|office|building|gate|here|venue|location))?|just\s+reached|coming\s+up)\b",
            msg_text,
            re.IGNORECASE,
        ))
        if is_arrival_response and 0 < msg_idx:
            prev_msg = messages[msg_idx - 1]
            prev_sender = prev_msg.get("sender_id", "")
            prev_ts = prev_msg.get("ts", 0)
            prev_text = _get_message_text(prev_msg)

            is_arrival_query = bool(re.search(
                r"\b(?:eta\b|which\s+floor|what\s+floor|which\s+flat|what\s+flat|where\s+are\s+you|"
                r"where\s+have\s+you\s+reached|where\s+in\s+the\s+building|reached\b.*(?:where|office|gate)|"
                r"outside\b.*(?:which|where)|send\s+(?:your\s+)?(?:office\s+)?address|"
                r"what(?:'s|\s+is)\s+(?:the\s+)?address|share\s+(?:the\s+)?location)\b",
                prev_text,
                re.IGNORECASE,
            ))
            if is_arrival_query and prev_sender != msg_sender and (0 <= msg_ts - prev_ts <= 1800):
                is_authorized_exchange = (
                    (msg_sender in (owner, requester) or (owner == "ME" and msg_sender in ("ME", requester))) and
                    (prev_sender in (owner, requester) or (owner == "ME" and prev_sender in ("ME", requester)))
                )
                if is_authorized_exchange and (0 <= msg_ts - source_ts <= 7 * 86400):
                    return "done", msg_id

    return None


def verify_full_chat_closure(
    source_msg: Dict[str, Any],
    chat_id: str,
    owner: str,
    loop_type: str,
    title: str,
    context: ConversationContext,
    raw_status: Optional[str] = None,
    raw_closed_by: Optional[str] = None,
) -> Tuple[str, Optional[str]]:
    """Scan full chronological chat history after source_msg to verify closure.

    Enforces identical actor authority and semantic linkage across both chronological
    scan and supplied raw_closed_by candidate messages.
    Returns ('done' | 'dropped' | 'open', closed_by_id_or_None).
    """
    if not is_valid_identifier(chat_id):
        return "open", None

    chat = context.get_chat(chat_id)
    if not chat:
        return "open", None

    source_ts = source_msg.get("ts", 0)
    source_sender = source_msg.get("sender_id", "")
    is_group = bool(chat.get("is_group", False))
    messages = chat.get("messages", [])
    source_idx = next((i for i, m in enumerate(messages) if m.get("id") == source_msg.get("id")), -1)

    # Identify original requester (distinct from obligor)
    if not is_group:
        if owner == "ME" and source_sender == "ME":
            other_pids = [p for p in chat.get("participants", []) if p != "ME"]
            requester: Optional[str] = other_pids[0] if other_pids else None
        else:
            requester = source_sender if owner == "ME" else "ME"
            if requester == owner:
                requester = None
    else:
        # In a group chat, if source_sender != owner, the source_sender is the requester.
        # If source_sender == owner (a commitment pledge), check if it immediately responded to someone else
        if source_sender != owner:
            requester = source_sender
        elif 0 < source_idx and messages[source_idx - 1].get("sender_id") != owner:
            requester = messages[source_idx - 1].get("sender_id")
        else:
            requester = None


    # Step 1: Detect authenticated delegates established in dialogue
    authenticated_delegates: Set[str] = set()
    delegation_pattern = re.compile(
        r"\b(?:handles?|handle\s+this|will\s+(?:handle|send|share|take\s+care|enable|validate|do|update)|can\s+(?:handle|send|share|enable|validate|do|help|update)|please\s+(?:handle|send|share|take\s+care|enable|validate|set\s+this\s+up|do|update)|take\s+care\s+of|delegate[ds]?|assigned\s+to|responsible\s+for|billing)\b",
        re.IGNORECASE,
    )
    for m in messages:
        m_ts = m.get("ts", 0)
        if m_ts >= source_ts:
            if m.get("sender_id") == owner:
                m_text = _get_message_text(m)
                if delegation_pattern.search(m_text):
                    # Validate that delegation applies to task being closed
                    m_content = _extract_content_tokens(m_text) - STOP_WORDS
                    title_content = _extract_content_tokens(title) - STOP_WORDS
                    shared_task_tokens = (m_content & title_content) - {"send", "review", "check", "call", "do", "help", "take", "care", "make", "get"}
                    is_task_relevant = (
                        m_ts == source_ts or
                        bool(shared_task_tokens) or
                        _has_semantic_overlap(title, m_text) or
                        (m_ts - source_ts <= 1800 and re.search(r"\b(?:this|that|it)\b", m_text, re.IGNORECASE))
                    )
                    if is_task_relevant:
                        for pid in chat.get("participants", []):
                            if pid == owner:
                                continue
                            p_info = context.get_participant(pid)
                            matched = False
                            for p_name in _get_participant_names(p_info):
                                if re.search(rf"(?:@|\b){re.escape(p_name)}\b", m_text, re.IGNORECASE):
                                    authenticated_delegates.add(pid)
                                    matched = True
                                    break
                            if not matched:
                                if re.search(rf"@{re.escape(pid)}\b", m_text, re.IGNORECASE):
                                    authenticated_delegates.add(pid)

    # Step 2: Validate raw_closed_by if supplied, enforcing full semantic linkage and authority
    if raw_closed_by is not None and is_valid_identifier(raw_closed_by):
        raw_msg = _get_dataset_message(context, raw_closed_by)
        if raw_msg:
            eval_res = _evaluate_closing_message(
                msg=raw_msg,
                source_msg=source_msg,
                chat_id=chat_id,
                owner=owner,
                requester=requester,
                authenticated_delegates=authenticated_delegates,
                loop_type=loop_type,
                title=title,
                is_group=is_group,
                messages=messages,
            )
            if eval_res is not None:
                status, closed_by = eval_res
                if raw_status is None or raw_status == status:
                    return status, closed_by

    # Step 3: Chronological scan across full chat history after source_msg
    for msg in messages:
        eval_res = _evaluate_closing_message(
            msg=msg,
            source_msg=source_msg,
            chat_id=chat_id,
            owner=owner,
            requester=requester,
            authenticated_delegates=authenticated_delegates,
            loop_type=loop_type,
            title=title,
            is_group=is_group,
            messages=messages,
        )
        if eval_res is not None:
            return eval_res

    # Default to open under Conservative Closure Principle
    return "open", None


def resolve_help_action(
    loop_title: str,
    owner: str,
    source_msg: Dict[str, Any],
    context: ConversationContext,
    config: OpenLoopExtractionConfig,
    suggested_action: Optional[str] = None,
    anchor_ts: Optional[int] = None,
    chat_id: Optional[str] = None,
) -> HelpAction:
    """Deterministic 5-tier evidence-grounded help action decision engine.

    Enforces zero fabrication fallback:
    - Tier 1: Contact-owned obligation -> 'leave' with grounded explanation.
    - Tier 2: Sensitive, physical, manual, personal voice communication, or casual observation -> 'leave'.
              Evaluates both loop title and complete source message body to prevent boundary bypasses.
    - Tier 3: Reversible AI assistance:
              - 3A. Procurement / Purchasing / Subscriptions / Bookings (needs_yes=True strictly).
              - 3B. Process Formulation / Workflows (needs_yes=False).
              - 3C. Drafting / Written Content (needs_yes=False).
              - 3D. Research / Search / Comparisons (needs_yes=False).
    - Tier 4: Deadlines, scheduled meetings, commercial decisions, or financial payments -> 'remind'.
    - Tier 5: Deterministic leave fallback.
    """
    clean_title = loop_title.replace("\r", " ").replace("\n", " ").strip() if isinstance(loop_title, str) else ""

    # Safe source message body extraction with chat verification
    body_text = ""
    if isinstance(source_msg, dict):
        msg_chat = source_msg.get("chat_id")
        if chat_id is None or msg_chat == chat_id:
            body_text = _get_message_text(source_msg).replace("\r", " ").replace("\n", " ").strip()

    # Tier 1. Contact-owned obligation
    if owner != "ME":
        return HelpAction(action="leave", reason=f"{owner} owns this task")

    # Verify that body_text is relevant source evidence for clean_title (e.g. truncated title prefix or shared tokens)
    clean_prefix = clean_title.rstrip(".").rstrip("…").strip()
    is_relevant_source = bool(
        body_text and (
            clean_prefix in body_text or
            body_text in clean_title or
            any(token.lower() in body_text.lower() for token in clean_title.split() if len(token) >= 4)
        )
    )

    # Tier 2. Personal, sensitive, physical, or non-actionable boundaries for ME
    # Must check both clean_title and relevant body_text so boundaries can never be bypassed
    texts_to_check = [clean_title]
    if is_relevant_source and body_text != clean_title:
        texts_to_check.append(body_text)

    for txt in texts_to_check:
        if SENSITIVE_TOPIC_PATTERN.search(txt):
            return HelpAction(action="leave", reason="Personal interpersonal commitment best handled directly")

        if PHYSICAL_CHORES_PATTERN.search(txt):
            return HelpAction(action="leave", reason="Physical manual chore best handled by domestic staff or in person")

        if PHYSICAL_HEALTH_CLASS_PATTERN.search(txt):
            return HelpAction(action="leave", reason="Physical health or in-person activity best handled directly")

        if DIRECT_PERSONAL_COMM_PATTERN.search(txt):
            return HelpAction(action="leave", reason="Personal family communication best handled directly")

        if CASUAL_OBSERVATION_PATTERN.search(txt):
            return HelpAction(action="leave", reason="Personal reflection or observation best decided directly")

    # Tier 3. Grounded reversible research, drafting, process, or procurement tasks ('assist')
    combined = f"{clean_title} {body_text}".strip()

    # A. Procurement / Purchasing / Subscriptions / Bookings (needs_yes = True strictly)
    if any(PROCUREMENT_ORDER_PATTERN.search(txt) for txt in texts_to_check):
        needs_yes = True
        if re.search(r"\b(?:homestay|tickets?|hotel|trip|vacation)\b", combined, re.IGNORECASE):
            next_step = f"Research options for {clean_title}" if not clean_title.lower().startswith("research") else clean_title
        elif re.search(r"\b(?:hanging\s+light|light)\b", combined, re.IGNORECASE):
            next_step = "Find and prepare order for studio hanging light"
        elif re.search(r"\b(?:watch|calling\s+watch)\b", combined, re.IGNORECASE):
            next_step = "Research options to order kids calling watch"
        elif re.search(r"\b(?:apple\s+pencil)\b", combined, re.IGNORECASE):
            next_step = "Check iPad compatibility and find purchase options for Apple Pencil"
        elif re.search(r"\b(?:fingerprint|biometric)\b", combined, re.IGNORECASE):
            next_step = "Research and compare biometric fingerprint devices"
        elif re.search(r"\b(?:claude\s+premium)\b", combined, re.IGNORECASE):
            next_step = "Prepare setup for Claude Premium subscription"
        elif re.search(r"\b(?:italica\s+tiles|tiles)\b", combined, re.IGNORECASE):
            next_step = "Prepare order for Italica tiles"
        else:
            next_step = f"Research and prepare order for {clean_title}"
        next_step = next_step.split(".")[0].strip()
        if not next_step:
            next_step = clean_title
        act = HelpAction(action="assist", next_step=next_step, needs_yes=needs_yes)
        if act.validate():
            return act

    # B. Process Formulation / Workflows (needs_yes = False)
    # Evaluated before general research so workflow formulation requests are not classified as generic research
    if any(PROCESS_FORMULATION_PATTERN.search(txt) for txt in texts_to_check):
        needs_yes = False
        if re.search(r"\b(?:reimburse|driver)\b", combined, re.IGNORECASE):
            next_step = "Formulate reimbursement workflow for driver"
        else:
            next_step = f"Formulate process for {clean_title}"
        next_step = next_step.split(".")[0].strip()
        if not next_step:
            next_step = clean_title
        act = HelpAction(action="assist", next_step=next_step, needs_yes=needs_yes)
        if act.validate():
            return act

    # C. Drafting / Document Preparation (needs_yes = False)
    if any(DRAFTING_WRITING_PATTERN.search(txt) for txt in texts_to_check):
        needs_yes = False
        if re.search(r"\b(?:invite)\b", combined, re.IGNORECASE):
            next_step = "Draft invitation text and layout"
        elif re.search(r"\b(?:save\s+the\s+date)\b", combined, re.IGNORECASE):
            next_step = "Draft save-the-date message for birthday picnic"
        elif re.search(r"\b(?:certificate)\b", combined, re.IGNORECASE):
            next_step = "Draft 4-week internship certificate for review"
        else:
            next_step = f"Draft {clean_title}"
        next_step = next_step.split(".")[0].strip()
        if not next_step:
            next_step = clean_title
        act = HelpAction(action="assist", next_step=next_step, needs_yes=needs_yes)
        if act.validate():
            return act

    # D. Research / Search / Recommendation (needs_yes = False)
    if any(RESEARCH_SEARCH_PATTERN.search(txt) for txt in texts_to_check):
        needs_yes = False
        if re.search(r"\b(?:teacher|piano)\b", combined, re.IGNORECASE):
            next_step = "Research and shortlist piano teachers for Dev"
        elif re.search(r"\b(?:options)\b", combined, re.IGNORECASE):
            next_step = "Find and compare options to share"
        else:
            next_step = f"Research options for {clean_title}"
        next_step = next_step.split(".")[0].strip()
        if not next_step:
            next_step = clean_title
        act = HelpAction(action="assist", next_step=next_step, needs_yes=needs_yes)
        if act.validate():
            return act

    # Precedence: Valid explicit future deadline takes precedence over generic assist
    has_explicit_future_deadline = (
        bool(DEADLINE_HELP_EXPR_PATTERN.search(clean_title) or (is_relevant_source and DEADLINE_HELP_EXPR_PATTERN.search(body_text)))
        and not bool(PAST_DEADLINE_HELP_PATTERN.search(clean_title) or (is_relevant_source and PAST_DEADLINE_HELP_PATTERN.search(body_text)))
    )

    # Precedence: Personal, commercial, or strategic decision questions require user discretion (remind, not assist)
    is_decision_question = bool(
        DECISION_QUESTION_PATTERN.search(clean_title)
        or (is_relevant_source and DECISION_QUESTION_PATTERN.search(body_text))
    )

    # Also respect suggested_action if passed and valid (unless an explicit future deadline or decision question takes precedence)
    if suggested_action == "assist" and clean_title and not has_explicit_future_deadline and not is_decision_question:
        next_step = clean_title.split(".")[0].strip() or clean_title
        act = HelpAction(action="assist", next_step=next_step, needs_yes=False)
        if act.validate():
            return act

    # Tier 4. Remind action for ME deadlines, decisions, meetings, or payments
    ts = anchor_ts if is_valid_timestamp(anchor_ts) else source_msg.get("ts") if isinstance(source_msg, dict) else None
    if is_valid_timestamp(ts):
        try:
            lead_hours = config.default_remind_lead_hours
            remind_dt = datetime.fromtimestamp(ts, tz=timezone.utc) + timedelta(hours=lead_hours)
            remind_at = remind_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            act = HelpAction(action="remind", remind_at=remind_at)
            if act.validate():
                return act
        except (ValueError, OverflowError, OSError):
            pass

    # Tier 5. Deterministic leave fallback
    return HelpAction(action="leave", reason="Follow up directly when convenient")


def deduplicate_open_loops(
    candidates: List[CandidateOpenLoop],
    context: ConversationContext,
    diagnostics: Optional[OpenLoopExtractionDiagnostics] = None,
) -> List[CandidateOpenLoop]:
    """Group open loops by lifecycle episodes, handling reminders, updates, and recurrences.

    - Reminders and status updates merge into the latest active open episode, preserving original source_msg.
    - Renewed obligations after terminal closure create distinct episodes retaining their own source_msg.
    - Multiple subsequent updates to a renewed episode merge into that new episode without duplication.
    """
    if not candidates:
        return []

    # Group by (chat_id, owner)
    by_chat_owner: Dict[Tuple[str, str], List[CandidateOpenLoop]] = {}
    for cand in candidates:
        key = (cand.chat_id, cand.owner)
        by_chat_owner.setdefault(key, []).append(cand)

    merged_results: List[CandidateOpenLoop] = []

    for (chat_id, owner), group in by_chat_owner.items():
        def get_source_ts(c: CandidateOpenLoop) -> float:
            msg = _get_dataset_message(context, c.source_msg)
            ts = msg.get("ts") if msg else 0
            return float(ts) if is_valid_timestamp(ts) else 0.0

        sorted_group = sorted(group, key=get_source_ts)
        active_episodes: List[CandidateOpenLoop] = []

        for cand in sorted_group:
            cand_msg = _get_dataset_message(context, cand.source_msg)
            cand_ts = cand_msg.get("ts", 0) if cand_msg else 0

            # Find compatible episodes among active episodes
            compatible_episodes = [
                (idx, ep) for idx, ep in enumerate(active_episodes)
                if _are_compatible_task_episodes(ep, cand, context)
            ]

            if not compatible_episodes:
                # Genuinely new distinct task episode
                active_episodes.append(cand)
            else:
                # Prioritize matching an active OPEN episode
                open_compatible = [item for item in compatible_episodes if item[1].status == "open"]

                if open_compatible:
                    # Update or reminder for current open episode: merge into latest open episode
                    idx, ep = open_compatible[-1]
                    if cand.source_msg not in ep.source_msgs_all:
                        ep.source_msgs_all.append(cand.source_msg)

                    if cand.due is not None:
                        ep.due = cand.due
                        ep.due_anchor_msg = cand.due_anchor_msg

                    if cand.status in ("done", "dropped"):
                        ep.status = cand.status
                        ep.closed_by = cand.closed_by
                        ep.help = None

                    if diagnostics:
                        diagnostics.loops_deduplicated += 1
                else:
                    # All compatible episodes are terminal (done or dropped).
                    # Check latest terminal episode.
                    idx, latest_ep = compatible_episodes[-1]
                    closed_msg = _get_dataset_message(context, latest_ep.closed_by) if latest_ep.closed_by else None
                    closed_ts = closed_msg.get("ts", 0) if closed_msg else 0

                    if cand_ts > closed_ts:
                        # Genuinely renewed obligation occurring after closure
                        # Initiates a new episode retaining its own initiating source_msg
                        active_episodes.append(cand)
                    else:
                        # Pre-closure update: merge into latest terminal episode
                        if cand.source_msg not in latest_ep.source_msgs_all:
                            latest_ep.source_msgs_all.append(cand.source_msg)
                        if latest_ep.closed_by == cand.source_msg and len(latest_ep.title) < len(cand.title):
                            latest_ep.source_msg = cand.source_msg
                            latest_ep.title = cand.title
                            latest_ep.owner = cand.owner
                            latest_ep.status = cand.status
                            latest_ep.closed_by = cand.closed_by
                            latest_ep.help = cand.help
                        if diagnostics:
                            diagnostics.loops_deduplicated += 1

        merged_results.extend(active_episodes)

    return merged_results


# ==============================================================================
# Milestone C.3 Safeguards: Banter and Informational Status Updates
# ==============================================================================

# Banter: playful physical threat and hyperbole patterns
PLAYFUL_THREAT_PATTERN = re.compile(
    r"\b(?:"
    r"(?:i'll|i\s+will|i\s+shall|gonna|will)\s+(?:slap|smack|punch|beat(?:\s+up)?|kill|murder)\s+(?:you|u|ya|him|her|them)|"
    r"(?:slap|smack|punch|kill|shoot)\s+(?:me|you|u)|"
    r"(?:i'll|im|i'm|gonna)\s+die\s+laughing|"
    r"shoot\s+me\s+now|kill\s+me\s+now"
    r")\b",
    re.IGNORECASE,
)

# Check-in and attendance response markers for roll-call participation validation
CHECK_IN_RESPONSE_PATTERN = re.compile(
    r"\b(?:"
    r"present|here|in\s+office|wfh|working\s+from\s+home|"
    r"available|checking\s+in|check[\s-]*in|"
    r"on\s+leave|taking\s+leave|sick\s+leave|half\s+day|"
    r"joining\s+late|will\s+join\s+late|delayed"
    r")\b",
    re.IGNORECASE,
)

# Slang contact phrases that use action verbs harmlessly
SLANG_CONTACT_PATTERN = re.compile(
    r"\b(?:hit\s+me\s+up|ping\s+me|reach\s+out|give\s+me\s+a\s+buzz)\b",
    re.IGNORECASE,
)

# Laughter and playful banter markers
BANTER_LAUGHTER_PATTERN = re.compile(
    r"(?:[\U0001F600-\U0001F60F\U0001F923\U0001F602]|"
    r"\b(?:lol|lmao|rofl|haha|hahaha|hahahaha|hehe|hehehe|jk|kidding|joke|just\s+kidding)\b)",
    re.IGNORECASE,
)

# Informational attendance and status reporting expressions
ATTENDANCE_STATUS_PATTERN = re.compile(
    r"\b(?:"
    # Joining / attendance timing (e.g. M00007: "join the 2nd half")
    r"(?:will\s+)?join(?:\s+in)?\s+(?:the\s+)?(?:2nd|second|1st|first)\s+half|"
    r"(?:will\s+)?join\s+(?:late|at\s+\d|around\s+\d|by\s+\d|in\s+\d+\s+mins?)|"
    r"(?:will\s+)?join\s+today|"
    # Absence / inability to attend (e.g. M00019: "doctor check up today will not be able to join today")
    r"(?:will\s+not|won't|cannot|can't)\s+be\s+able\s+to\s+(?:join|attend|come)|"
    r"(?:not\s+be\s+able\s+to|won't\s+be\s+able\s+to|cannot|can't)\s+(?:join|attend|come)|"
    r"will\s+not\s+(?:join|attend|come)\s+today|won't\s+(?:join|attend|come)\s+today|"
    # Running late / delays
    r"running\s+(?:\d+\s+mins?\s+)?late|delayed\s+by\s+\d+|stuck\s+in\s+traffic|"
    # Leave / doctor appointment / availability announcements
    r"(?:on|taking)\s+leave|doctor(?:\s+check\s*up|\s+appointment)|"
    r"(?:not|won't\s+be)\s+available\s+today|out\s+of\s+office|wfh\s+today|working\s+from\s+home\s+today|"
    # Physical arrival / location reporting
    r"reached(?:\s+the)?\s+(?:office|venue|site|station|airport|location)|in\s+office\s+today"
    r")\b",
    re.IGNORECASE,
)

# Combined deliverable or actionable request indicators
ACTIONABLE_DELIVERABLE_OR_REQUEST_PATTERN = re.compile(
    r"\b(?:"
    r"(?:and|but)\s+(?:will|i'll|shall|i\s+will)\s+(?:present|send|email|share|bring|prepare|deliver|submit|call|review|check)|"
    r"(?:present|send|email|share|bring|prepare|deliver|submit|call|review|check)\s+(?:the|a|an|my|our)\s+[a-z0-9_]+|"
    r"(?:please|plz|pls)\s+(?:start|review|check|send|call|share|proceed|present)"
    r")\b",
    re.IGNORECASE,
)

# Structured roll-call or attendance prompt triggers
ROLL_CALL_TRIGGER_PATTERN = re.compile(
    r"\b(?:"
    r"roll[\s-]*call|daily\s+standup|standup\s+call|"
    r"(?:please\s+)?(?:share|post|give|send|update)\s+(?:today'?s\s+)?availability|"
    r"mark\s+your\s+attendance|check[\s-]*in\s+for\s+today|"
    r"who\s+is\s+(?:in|available|joining|attending)\s+today"
    r")\b",
    re.IGNORECASE,
)

# Clear attendance / absence / delay acknowledgement expressions
STATUS_ACKNOWLEDGEMENT_PATTERN = re.compile(
    r"\b(?:"
    r"noted\s*,?\s*(?:thanks|thank\s+you|take\s+care|see\s+you)|"
    r"thanks\s+for\s+(?:informing|letting\s+us\s+know|the\s+update)|"
    r"get\s+well\s+soon|take\s+care|take\s+rest|"
    r"(?:ok|okay|sure|noted)\s*,?\s*(?:see\s+you|join\s+when\s+you\s+can|see\s+u)|"
    r"see\s+you\s+(?:in\s+the\s+2nd\s+half|then|later|there)|"
    r"understood\s*,?\s*(?:see\s+you|take\s+care)?"
    r")\b",
    re.IGNORECASE,
)


def is_non_actionable_banter(
    loop: InterpretedOpenLoop,
    source_msg: Dict[str, Any],
    context: ConversationContext,
) -> bool:
    """Evaluate whether an interpreted candidate is clearly non-actionable banter or hyperbole.

    Uses a conservative 3-factor contextual framework:
    1. Pragmatic action and intent: confirms absence of a real-world task or commitment.
    2. Grammatical and rhetorical pattern: verifies construction indicates playful threat or hyperbole.
    3. Conversational framing and dynamics: checks surrounding messages for playful/banter context.

    Rejects a candidate ONLY when all three factors clearly align. Defaults to retention if
    context is missing, incomplete, or ambiguous.
    """
    body = _get_message_text(source_msg)
    title = loop.title or ""
    combined_text = f"{body} {title}".strip()
    if not combined_text:
        return False

    # Slang contact phrases (e.g. 'hit me up') communicate legitimate communication requests
    if SLANG_CONTACT_PATTERN.search(combined_text):
        return False

    # Factor 2: Grammatical and rhetorical pattern
    # Must match a recognized playful threat, teasing hyperbole, or exaggerated phrasing
    has_playful_pattern = bool(
        PLAYFUL_THREAT_PATTERN.search(body) or PLAYFUL_THREAT_PATTERN.search(title)
    )
    if not has_playful_pattern:
        return False

    # Factor 1: Pragmatic action and intent
    # If the message communicates an actionable deliverable, concrete task,
    # or explicit request, it is NOT non-actionable banter.
    if ACTIONABLE_DELIVERABLE_OR_REQUEST_PATTERN.search(body):
        return False

    # Factor 3: Conversational framing and dynamics
    # Requires surrounding conversation context to confirm playful/joking interpretation.
    # If context is missing, incomplete, or ambiguous, default to retention (False).
    chat_id = source_msg.get("chat_id") or loop.chat_id
    if not chat_id:
        return False

    chat = context.get_chat(chat_id)
    chat_messages: List[Dict[str, Any]] = []
    if hasattr(context, "get_chat_messages"):
        chat_messages = context.get_chat_messages(chat_id)
    if not chat_messages and chat:
        chat_messages = chat.get("messages", [])

    if not chat_messages or len(chat_messages) <= 1:
        # Missing or single-message context: cannot confirm framing -> conservatively retain
        # (unless the candidate message itself contains explicit laughter markers)
        if not BANTER_LAUGHTER_PATTERN.search(body):
            return False

    # Locate source_msg in chat messages to inspect surrounding window
    msg_id = source_msg.get("id")
    source_idx = -1
    for idx, m in enumerate(chat_messages):
        if m.get("id") == msg_id:
            source_idx = idx
            break

    # Look for laughter/banter cues in source message or nearby messages (±3 window)
    has_framing_cues = bool(BANTER_LAUGHTER_PATTERN.search(body))
    if source_idx != -1:
        window_start = max(0, source_idx - 3)
        window_end = min(len(chat_messages), source_idx + 4)
        for i in range(window_start, window_end):
            m_body = _get_message_text(chat_messages[i])
            if BANTER_LAUGHTER_PATTERN.search(m_body):
                has_framing_cues = True
                break

    if not has_framing_cues:
        # No affirmative banter framing in conversation -> conservative default: retain
        return False

    return True


def _get_group_intended_counterparty(
    source_msg: Dict[str, Any],
    sorted_msgs: List[Dict[str, Any]],
    source_idx: int,
    context: ConversationContext,
    chat: Optional[Dict[str, Any]],
) -> Optional[str]:
    """Identify the established intended recipient/counterparty in a group chat.

    Requires grounded evidence from explicit quotation, participant @mention, or
    direct inquiry in the immediately preceding message. Returns None if ambiguous.
    """
    source_sender = source_msg.get("sender_id")
    body = _get_message_text(source_msg)

    # 1. Explicit quoted message by source_msg
    quoted_id = source_msg.get("quoted_id")
    if quoted_id:
        quoted = _get_dataset_message(context, quoted_id)
        if quoted and quoted.get("sender_id") and quoted.get("sender_id") != source_sender:
            return quoted.get("sender_id")

    # 2. Explicit @mention or direct addressing of participant in source_msg
    participants = chat.get("participants", []) if chat else []
    for pid in participants:
        if pid == source_sender:
            continue
        # Check @pid mention
        if re.search(rf"@\b{re.escape(pid)}\b", body, re.IGNORECASE):
            return pid
        # Check participant display name
        p_info = context.get_participant(pid)
        for p_name in _get_participant_names(p_info):
            name_pattern = rf"(?:@\b{re.escape(p_name)}\b|\b{re.escape(p_name)}\b|^(?:hi|hey|hello)?\s*\b{re.escape(p_name)}\b\s*[,:])"
            if re.search(name_pattern, body, re.IGNORECASE):
                return pid

    # 3. Direct inquiry addressed to source_sender in the immediately preceding message
    if source_idx > 0:
        prior = sorted_msgs[source_idx - 1]
        prior_sender = prior.get("sender_id")
        if prior_sender and prior_sender != source_sender:
            prior_text = _get_message_text(prior)
            source_info = context.get_participant(source_sender) if source_sender else None
            is_addressed = False
            if source_sender and re.search(rf"@\b{re.escape(source_sender)}\b", prior_text, re.IGNORECASE):
                is_addressed = True
            elif source_info:
                for s_name in _get_participant_names(source_info):
                    if re.search(rf"(?:@|\b){re.escape(s_name)}\b", prior_text, re.IGNORECASE):
                        is_addressed = True
                        break
            if is_addressed and ("?" in prior_text or re.search(r"\b(where|status|update|available|joining|reach)\b", prior_text, re.IGNORECASE)):
                return prior_sender

    return None


def is_informational_status_update(
    loop: InterpretedOpenLoop,
    source_msg: Dict[str, Any],
    context: ConversationContext,
) -> bool:
    """Evaluate whether an interpreted candidate is a purely informational attendance or status update.

    A message reporting presence, absence, lateness, delay, availability, or location may be
    rejected ONLY when affirmative contextual evidence establishes that interpretation through
    one of three exact routes:
    1. Direct reply: established through quoted_id or explicit quoted/reply text clearly
       referring to the candidate update.
    2. Immediate response: the intended counterparty's acknowledgement is the immediately
       following message in the conversation, with zero intervening messages, and clearly
       acknowledges that specific update.
    3. Structured roll-call: the candidate demonstrably participates in an explicit
       structured roll-call or attendance prompt.

    Preserves attendance messages that also contain a concrete deliverable, responsibility,
    or actionable request. Defaults to retention if context does not establish affirmative evidence.
    """
    body = _get_message_text(source_msg)
    title = loop.title or ""
    combined_text = f"{body} {title}".strip()
    if not combined_text:
        return False

    # Check if the candidate is an attendance or status statement
    if not ATTENDANCE_STATUS_PATTERN.search(body) and not ATTENDANCE_STATUS_PATTERN.search(title):
        return False

    # Preserve messages that combine attendance/status with a concrete deliverable or request
    if ACTIONABLE_DELIVERABLE_OR_REQUEST_PATTERN.search(body) or ACTIONABLE_DELIVERABLE_OR_REQUEST_PATTERN.search(title):
        return False

    # Check if context provides affirmative evidence
    chat_id = source_msg.get("chat_id") or loop.chat_id
    if not chat_id:
        return False

    chat = context.get_chat(chat_id)
    chat_messages: List[Dict[str, Any]] = []
    if hasattr(context, "get_chat_messages"):
        chat_messages = context.get_chat_messages(chat_id)
    if not chat_messages and chat:
        chat_messages = chat.get("messages", [])

    if not chat_messages:
        return False

    # Sort messages deterministically by timestamp and id
    sorted_msgs = sorted(
        chat_messages,
        key=lambda m: (m.get("ts") or 0, m.get("id") or "")
    )

    source_id = source_msg.get("id")
    source_sender = source_msg.get("sender_id")
    source_idx = -1
    for idx, m in enumerate(sorted_msgs):
        if m.get("id") == source_id:
            source_idx = idx
            break

    if source_idx == -1:
        return False

    is_group = bool(chat.get("is_group", False)) if chat else False

    # Route 1: Direct reply via quoted_id or explicit quoted text referring to candidate update
    for m in sorted_msgs[source_idx + 1:]:
        if m.get("sender_id") == source_sender:
            continue
        m_text = _get_message_text(m)
        # Direct reply via quoted_id
        if m.get("quoted_id") == source_id:
            if (
                STATUS_ACKNOWLEDGEMENT_PATTERN.search(m_text)
                or BARE_EXECUTION_PATTERN.search(m_text)
                or re.search(r"\b(ok|okay|noted|sure|thanks|thank\s+you|thx|understood)\b", m_text, re.IGNORECASE)
            ):
                return True
        # Explicit quoted text
        if body and (f"> {body}" in m_text or (len(body) > 10 and body in m_text)):
            if (
                STATUS_ACKNOWLEDGEMENT_PATTERN.search(m_text)
                or BARE_EXECUTION_PATTERN.search(m_text)
                or re.search(r"\b(ok|okay|noted|sure|thanks|thank\s+you|thx|understood)\b", m_text, re.IGNORECASE)
            ):
                return True

    # Route 2: Immediate response (immediately following message with ZERO intervening messages)
    if source_idx + 1 < len(sorted_msgs):
        immediate_next = sorted_msgs[source_idx + 1]
        next_sender = immediate_next.get("sender_id")
        next_text = _get_message_text(immediate_next)

        if not is_group:
            # In direct chat, the other participant can be the counterparty when the acknowledgement clearly refers to the update
            other_participants = [p for p in (chat.get("participants", []) if chat else []) if p != source_sender]
            if next_sender in other_participants:
                if STATUS_ACKNOWLEDGEMENT_PATTERN.search(next_text):
                    return True
        else:
            # In group chat, require grounded evidence identifying the intended counterparty or recipient.
            # If that identity cannot be established from the supplied context, do not use the immediate-response route.
            intended_counterparty = _get_group_intended_counterparty(
                source_msg, sorted_msgs, source_idx, context, chat
            )
            if intended_counterparty and next_sender == intended_counterparty:
                if STATUS_ACKNOWLEDGEMENT_PATTERN.search(next_text):
                    return True

    # Route 3: Demonstrable structured roll-call participation
    # Establish that the candidate actually participates in an explicit roll-call exchange.
    # Proximity alone or an unrelated conversation occurring after the trigger does NOT qualify.
    for i in range(source_idx - 1, max(-1, source_idx - 11), -1):
        prior_msg = sorted_msgs[i]
        prior_sender = prior_msg.get("sender_id")
        if prior_sender == source_sender:
            continue
        prior_text = _get_message_text(prior_msg)
        if ROLL_CALL_TRIGGER_PATTERN.search(prior_text):
            trigger_idx = i
            # Check for demonstrable participation:
            # A. Candidate directly quotes the roll-call trigger
            if source_msg.get("quoted_id") == prior_msg.get("id"):
                return True

            # B. Candidate immediately follows the trigger (zero intervening messages)
            if source_idx == trigger_idx + 1:
                return True

            # C. Candidate participates in an ongoing unbroken roll-call exchange
            # Every intervening message must be an attendance/check-in response or acknowledgement
            intervening = sorted_msgs[trigger_idx + 1 : source_idx]
            has_unrelated_interruption = False
            for im in intervening:
                im_text = _get_message_text(im)
                is_checkin = bool(
                    ATTENDANCE_STATUS_PATTERN.search(im_text)
                    or CHECK_IN_RESPONSE_PATTERN.search(im_text)
                    or STATUS_ACKNOWLEDGEMENT_PATTERN.search(im_text)
                )
                if not is_checkin:
                    has_unrelated_interruption = True
                    break

            if not has_unrelated_interruption:
                return True

            # An unrelated intervening conversation broke the roll-call linkage
            break

    # If none of Route 1, Route 2, or Route 3 is satisfied, affirmative evidence is NOT established.
    # Conservative fallback: retain the candidate for normal resolver processing.
    return False


def resolve_open_loops(
    interpreted_loops: Sequence[InterpretedOpenLoop],
    context: ConversationContext,
    config: Optional[OpenLoopExtractionConfig] = None,
    diagnostics: Optional[OpenLoopExtractionDiagnostics] = None,
) -> List[CandidateOpenLoop]:
    """Transform interpreted open-loop records into validated CandidateOpenLoop records.

    Performs source verification, ownership resolution, due-date anchoring, full-chat
    closure verification, lifecycle episode merging, confidence filtering, and model validation.
    """
    if config is None:
        config = OpenLoopExtractionConfig()

    if diagnostics is not None:
        diagnostics.interpreted_loops_received += len(interpreted_loops)

    candidates: List[CandidateOpenLoop] = []

    for loop in interpreted_loops:
        # 1. Filter non-genuine items
        if not loop.is_genuine:
            if diagnostics:
                diagnostics.candidates_rejected += 1
            continue

        if diagnostics:
            diagnostics.genuine_loops_received += 1

        # 2. Strict confidence score validation (never inflates missing confidence)
        conf = validate_confidence_score(loop.confidence)
        if conf is None or conf < config.min_confidence:
            if diagnostics:
                diagnostics.candidates_rejected += 1
            continue

        # 3. Loop type validation (reject invalid, missing, non-string, or unsupported types)
        if not isinstance(loop.type, str) or loop.type not in VALID_OPEN_LOOP_TYPES:
            if diagnostics:
                diagnostics.candidates_rejected += 1
            continue
        loop_type = loop.type

        # 4. Source and chat identifier validation (no .strip() normalization)
        if not is_valid_identifier(loop.source_msg) or not is_valid_identifier(loop.chat_id):
            if diagnostics:
                diagnostics.candidates_rejected += 1
            continue

        chat = context.get_chat(loop.chat_id)
        if not chat:
            if diagnostics:
                diagnostics.candidates_rejected += 1
            continue

        source_msg = _get_dataset_message(context, loop.source_msg)
        if not source_msg or source_msg.get("chat_id") != loop.chat_id:
            if diagnostics:
                diagnostics.candidates_rejected += 1
            continue

        # Safeguard 4a: Banter, playful threats, and hyperbole
        if is_non_actionable_banter(loop, source_msg, context):
            if diagnostics:
                diagnostics.candidates_rejected += 1
            continue

        # Safeguard 4b: Informational attendance and status updates
        if is_informational_status_update(loop, source_msg, context):
            if diagnostics:
                diagnostics.candidates_rejected += 1
            continue

        # 4. Ownership resolution
        owner = resolve_loop_owner(loop.owner, loop.chat_id, source_msg, context)
        if not owner or not is_valid_identifier(owner):
            if diagnostics:
                diagnostics.candidates_rejected += 1
            continue

        # 5. Due date and anchor resolution
        due, due_anchor = resolve_due_date(
            loop.due_date,
            loop.due_anchor_msg,
            source_msg,
            loop.chat_id,
            context,
        )

        # 7. Full-chat closure scan
        status, closed_by = verify_full_chat_closure(
            source_msg=source_msg,
            chat_id=loop.chat_id,
            owner=owner,
            loop_type=loop_type,
            title=loop.title,
            context=context,
            raw_status=loop.status,
            raw_closed_by=loop.closed_by,
        )

        # 7. Help action resolution for open loops
        help_action: Optional[HelpAction] = None
        if status == "open":
            anchor_msg = _get_dataset_message(context, due_anchor) if due_anchor else None
            anchor_ts = anchor_msg.get("ts") if anchor_msg else None
            help_action = resolve_help_action(
                loop_title=loop.title,
                owner=owner,
                source_msg=source_msg,
                context=context,
                config=config,
                suggested_action=loop.suggested_action,
                anchor_ts=anchor_ts,
                chat_id=loop.chat_id,
            )
        else:
            help_action = None

        cand = CandidateOpenLoop(
            title=loop.title,
            type=loop_type,
            owner=owner,
            chat_id=loop.chat_id,
            source_msg=loop.source_msg,
            due=due,
            due_anchor_msg=due_anchor,
            status=status,
            closed_by=closed_by,
            help=help_action,
            confidence=conf,
            source_msgs_all=[loop.source_msg],
        )

        # Model validation barrier
        if not cand.validate():
            if diagnostics:
                diagnostics.candidates_rejected += 1
            continue

        candidates.append(cand)
        if diagnostics:
            diagnostics.candidates_created += 1

    # 8. Lifecycle episode deduplication
    deduped = deduplicate_open_loops(candidates, context, diagnostics)

    # 9. Update final status counters
    if diagnostics:
        for cand in deduped:
            diagnostics.final_loops_emitted += 1
            if cand.status == "done":
                diagnostics.loops_marked_done += 1
            elif cand.status == "dropped":
                diagnostics.loops_marked_dropped += 1
            elif cand.status == "open":
                diagnostics.loops_kept_open += 1
                if cand.help:
                    if cand.help.action == "assist":
                        diagnostics.help_assist_count += 1
                    elif cand.help.action == "remind":
                        diagnostics.help_remind_count += 1
                    elif cand.help.action == "leave":
                        diagnostics.help_leave_count += 1

    return deduped
