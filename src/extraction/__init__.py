"""Candidate evidence extraction module."""

from src.extraction.evidence_extractor import (
    CandidateEvidenceExtractor,
    extract_candidate_evidence,
    extract_message_evidence,
    is_relevant_group_candidate,
)

__all__ = [
    "CandidateEvidenceExtractor",
    "extract_candidate_evidence",
    "extract_message_evidence",
    "is_relevant_group_candidate",
]

