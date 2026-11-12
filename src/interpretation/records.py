"""Intermediate interpretation records for WhatsApp conversation evidence analysis.

These dataclasses decouple semantic interpretation from final schema generation,
capturing extracted intent, conversational relationships, and observational metadata.
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class InterpretedOpenLoop:
    """
    Represents an actionable task, promise, question, or followup evaluated within its conversational context.

    Attributes:
        source_msg: Candidate message ID where the loop originated.
        chat_id: Chat ID containing the loop.
        type: Classification of the open loop ('commitment', 'question', 'followup', 'deadline').
        is_genuine: True if real actionable item; False if pleasantry, banter, or rhetorical.
        title: Concise, actionable summary in English (e.g. 'Send report to Neha').
        owner: 'ME' or participant ID of person who owes the action; None if unresolvable.
        target_person: Participant ID to whom the commitment/question was addressed; None if general/unclear.
        status: Current state ('open', 'done', 'dropped').
        closed_by: Message ID demonstrating resolution/dropping; None if open.
        due_date: Due date formatted as 'YYYY-MM-DD', or None if ambiguous/unspecified.
        suggested_action: Preliminary assistance hint ('assist', 'remind', 'leave', None).
        confidence: Optional debugging score (0.0 to 1.0).
        validation_flags: Optional validation warnings or flags recorded during verification.
        due_anchor_msg: Exact message ID supplying the date expression for due_date; None if unspecified/ambiguous.
    """

    source_msg: str
    chat_id: str
    type: str
    is_genuine: bool
    title: str
    owner: Optional[str]
    target_person: Optional[str]
    status: str
    closed_by: Optional[str]
    due_date: Optional[str]
    suggested_action: Optional[str]
    confidence: Optional[float] = None
    validation_flags: Optional[List[str]] = None
    due_anchor_msg: Optional[str] = None


@dataclass
class InterpretedGoal:
    """
    Represents an overarching personal or professional ambition.

    Attributes:
        title: Concise title of the goal (e.g. 'Run the half marathon in January').
        area: Life area ('self', 'spouse', 'kids', 'parents', 'work', 'friends', 'experiences', 'finances', 'health', 'home').
        type: Goal nature ('habit', 'milestone', 'experience').
        horizon: Time horizon ('3mo', '1yr', '3yr', '5yr').
        evidence_msgs: List of exact message IDs supporting this goal.
        is_genuine: True if genuine long-term ambition; False if tactical/immediate desire.
        confidence: Optional debugging score (0.0 to 1.0).
    """

    title: str
    area: str
    type: str
    horizon: str
    evidence_msgs: List[str]
    is_genuine: bool
    confidence: Optional[float] = None


@dataclass
class InterpretedPerson:
    """
    Represents conversational relationship inference for a specific contact.

    Attributes:
        person_id: Exact participant ID (e.g. 'P01').
        circle: Inferred relationship circle (e.g. 'colleague', 'friend', 'staff', 'spouse'), or None if unknown.
        looks_after: List of participant IDs this person looks after (e.g. staff looking after kids).
        associated_org_names: Observed organisation names linked to this person.
        org_link_type: Relationship to associated organisation ('runs', 'supports', None).
        evidence_msgs: List of exact message IDs supporting this relationship.
        confidence: Optional debugging score (0.0 to 1.0).
    """

    person_id: str
    circle: Optional[str]
    looks_after: List[str]
    associated_org_names: List[str]
    org_link_type: Optional[str]
    evidence_msgs: List[str]
    confidence: Optional[float] = None


@dataclass
class InterpretedOrganisation:
    """
    Represents an observed organisation, its domain, and conversational role.

    Attributes:
        name: Observed organisation name.
        org_type: Organisation classification ('company', 'residence', 'vacation home', 'workspace', 'society', 'school', 'other'), or None.
        side: Domain side ('home', 'work', None).
        role: Primary user's relationship ('owner', 'partner', 'helper', 'resident', 'parent', 'member', 'other', None).
        evidence_msgs: List of exact message IDs referencing this organisation.
        associated_chats: List of chat IDs tied to this organisation.
        confidence: Optional debugging score (0.0 to 1.0).
    """

    name: str
    org_type: Optional[str]
    side: Optional[str]
    role: Optional[str]
    evidence_msgs: List[str]
    associated_chats: List[str]
    confidence: Optional[float] = None

