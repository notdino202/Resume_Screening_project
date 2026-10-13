"""Education extraction, degree leveling (1-5), and gap scoring module."""

from __future__ import annotations
from dataclasses import dataclass, field
import re
from typing import Optional


DEGREE_LEVELS = {
    "doctorate": 5,
    "master": 4,
    "bachelor": 3,
    "diploma": 2,
    "high_school": 1,
}

# Regex patterns mapped to degree level and canonical name
DEGREE_PATTERNS = [
    # Level 5: Doctorate
    (
        5,
        "Doctorate",
        re.compile(
            r"\b(?:Ph\.?D\.?|Doctor\s+of\s+Philosophy|Doctorate|D\.?Phil|Postdoctoral|Postdoc)\b",
            re.IGNORECASE,
        ),
    ),
    # Level 4: Master's
    (
        4,
        "Master's",
        re.compile(
            r"\b(?:Master\s+of\s+Science|Master\s+of\s+Arts|Master\s+of\s+Engineering|Master\s+of\s+Technology|Master\s+of\s+Business\s+Administration|M\.?S\.?(?:\s+in|\b)|M\.?A\.?(?:\s+in|\b)|M\.?Tech|M\.?E\.?|MBA|Master(?:'s)?(?:\s+degree)?)\b",
            re.IGNORECASE,
        ),
    ),
    # Level 3: Bachelor's
    (
        3,
        "Bachelor's",
        re.compile(
            r"\b(?:Bachelor\s+of\s+Science|Bachelor\s+of\s+Arts|Bachelor\s+of\s+Technology|Bachelor\s+of\s+Engineering|Bachelor\s+of\s+Business|B\.?S\.?(?:\s+in|\b)|B\.?A\.?(?:\s+in|\b)|B\.?Tech|B\.?E\.?|BBA|BCA|Bachelor(?:'s)?(?:\s+degree)?|Undergraduate(?:\s+degree)?)\b",
            re.IGNORECASE,
        ),
    ),
    # Level 2: Diploma / Associate
    (
        2,
        "Diploma/Associate",
        re.compile(
            r"\b(?:Associate(?:\s+of\s+Science|\s+of\s+Arts)?(?:\s+Degree)?|(?<!high school )diploma|Associate's(?:\s+Degree)?|Foundation\s+Degree)\b",
            re.IGNORECASE,
        ),
    ),
    # Level 1: High School
    (
        1,
        "High School",
        re.compile(
            r"\b(?:High\s+School(?:\s+Diploma)?|Secondary\s+School|GED|12th\s+Grade|Higher\s+Secondary)\b",
            re.IGNORECASE,
        ),
    ),
]


@dataclass
class EducationEntry:
    """Individual extracted education credential."""
    degree_name: str
    level: int
    matched_text: str
    raw_context: str


@dataclass
class CandidateEducation:
    """Aggregated education profile for a candidate."""
    entries: list[EducationEntry] = field(default_factory=list)
    highest_level: int = 0
    highest_degree_name: str = "None"


class EducationParser:
    """Extracts degree credentials, assigns 1-5 levels, and scores gap against requirements."""

    def __init__(self):
        self.degree_patterns = DEGREE_PATTERNS

    def parse_education(self, text: str) -> CandidateEducation:
        """Parses education text or full resume text to extract degree entries.

        Args:
            text: Text to scan (typically education section or full resume).

        Returns:
            CandidateEducation containing detected entries and highest degree level.
        """
        if not text or not text.strip():
            return CandidateEducation()

        entries: list[EducationEntry] = []
        lines = text.split("\n")

        for line in lines:
            line_str = line.strip()
            if not line_str:
                continue

            for level, canonical_name, pattern in self.degree_patterns:
                match = pattern.search(line_str)
                if match:
                    entries.append(
                        EducationEntry(
                            degree_name=canonical_name,
                            level=level,
                            matched_text=match.group(0),
                            raw_context=line_str,
                        )
                    )
                    break  # One degree per line match

        if not entries:
            # Fallback scan across entire text block if line-by-line missed multiline degree phrases
            for level, canonical_name, pattern in self.degree_patterns:
                for match in pattern.finditer(text):
                    entries.append(
                        EducationEntry(
                            degree_name=canonical_name,
                            level=level,
                            matched_text=match.group(0),
                            raw_context=match.group(0),
                        )
                    )

        highest_level = max([e.level for e in entries], default=0)
        highest_name = "None"
        for e in entries:
            if e.level == highest_level:
                highest_name = e.degree_name
                break

        return CandidateEducation(
            entries=entries,
            highest_level=highest_level,
            highest_degree_name=highest_name,
        )

    @staticmethod
    def score_education(candidate_level: int, required_level: int = 3) -> tuple[float, int, list[str]]:
        """Computes education score and advisory flags based on level gap.

        SPEC rule:
        - gap = candidate_level - required_level
        - gap >= 0 scores 1.0
        - gap == -1 scores 0.5
        - gap < -1 (otherwise) scores 0.0

        Args:
            candidate_level: Candidate's highest degree level (1-5, or 0 if unextracted).
            required_level: Required degree level from JD (1-5).

        Returns:
            Tuple of (score, gap, advisory_flags).
        """
        gap = candidate_level - required_level
        flags: list[str] = []

        if gap >= 0:
            score = 1.0
        elif gap == -1:
            score = 0.5
            flags.append("education below requirement")
        else:
            score = 0.0
            flags.append("education below requirement")

        return score, gap, flags
