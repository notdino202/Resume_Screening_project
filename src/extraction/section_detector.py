"""Resume section detection and segmentation module."""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
import re
from typing import Optional


class SectionType(str, Enum):
    SUMMARY = "SUMMARY"
    EXPERIENCE = "EXPERIENCE"
    EDUCATION = "EDUCATION"
    SKILLS = "SKILLS"
    PROJECTS = "PROJECTS"
    CERTIFICATIONS = "CERTIFICATIONS"
    OTHER = "OTHER"


SECTION_HEADER_PATTERNS = [
    (SectionType.EXPERIENCE, re.compile(r"^\s*(?:PROFESSIONAL\s+EXPERIENCE|WORK\s+EXPERIENCE|WORK\s+HISTORY|EMPLOYMENT\s+HISTORY|EXPERIENCE|RESEARCH\s+&\s+EXPERIENCE)\s*[:\-]?\s*$", re.IGNORECASE)),
    (SectionType.EDUCATION, re.compile(r"^\s*(?:EDUCATION|ACADEMIC\s+BACKGROUND|ACADEMIC\s+HISTORY|QUALIFICATIONS|DEGREES)\s*[:\-]?\s*$", re.IGNORECASE)),
    (SectionType.SKILLS, re.compile(r"^\s*(?:TECHNICAL\s+SKILLS|CORE\s+COMPETENCIES|SKILLS\s*&?\s*TOOLS|SKILLS|TECHNOLOGIES|AREAS\s+OF\s+EXPERTISE|COMPETENCIES)\s*[:\-]?\s*$", re.IGNORECASE)),
    (SectionType.PROJECTS, re.compile(r"^\s*(?:PROJECTS|KEY\s+PROJECTS|PERSONAL\s+PROJECTS|ACADEMIC\s+PROJECTS|NOTABLE\s+PROJECTS)\s*[:\-]?\s*$", re.IGNORECASE)),
    (SectionType.CERTIFICATIONS, re.compile(r"^\s*(?:CERTIFICATIONS|CERTIFICATES|LICENSES|ACCREDITATIONS|PROFESSIONAL\s+CERTIFICATIONS)\s*[:\-]?\s*$", re.IGNORECASE)),
    (SectionType.SUMMARY, re.compile(r"^\s*(?:PROFESSIONAL\s+SUMMARY|SUMMARY|PROFILE|EXECUTIVE\s+SUMMARY|PROFESSIONAL\s+PROFILE|OBJECTIVE|CAREER\s+OBJECTIVE|ABOUT\s+ME)\s*[:\-]?\s*$", re.IGNORECASE)),
]


@dataclass
class ResumeSections:
    """Stores text segmented into standard resume sections."""
    raw_text: str
    section_texts: dict[SectionType, str] = field(default_factory=dict)
    detected_headers: list[tuple[SectionType, str, int]] = field(default_factory=list)

    def get_text(self, section_type: SectionType) -> str:
        """Returns the concatenated text for the given section type."""
        return self.section_texts.get(section_type, "")

    @property
    def experience_text(self) -> str:
        return self.get_text(SectionType.EXPERIENCE)

    @property
    def education_text(self) -> str:
        return self.get_text(SectionType.EDUCATION)

    @property
    def skills_text(self) -> str:
        return self.get_text(SectionType.SKILLS)

    @property
    def projects_text(self) -> str:
        return self.get_text(SectionType.PROJECTS)

    @property
    def certifications_text(self) -> str:
        return self.get_text(SectionType.CERTIFICATIONS)

    @property
    def summary_text(self) -> str:
        return self.get_text(SectionType.SUMMARY)


class SectionDetector:
    """Parses clean resume text into structured sections using heading heuristics."""

    def __init__(self):
        self.patterns = SECTION_HEADER_PATTERNS

    def detect_sections(self, text: str) -> ResumeSections:
        """Segments text into sections based on detected heading boundaries.

        Args:
            text: Cleaned resume text string.

        Returns:
            ResumeSections instance containing categorized section texts.
        """
        lines = text.split("\n")
        section_accumulator: dict[SectionType, list[str]] = {st: [] for st in SectionType}
        detected_headers: list[tuple[SectionType, str, int]] = []

        current_section = SectionType.SUMMARY  # Top of resume before first heading default

        for idx, line in enumerate(lines):
            clean_line = line.strip()
            if not clean_line:
                continue

            matched_section = self._match_header(clean_line)
            if matched_section is not None:
                current_section = matched_section
                detected_headers.append((matched_section, clean_line, idx))
                continue

            section_accumulator[current_section].append(clean_line)

        # Merge line lists into coherent section text blocks
        section_texts = {
            sec: "\n".join(lines).strip()
            for sec, lines in section_accumulator.items()
            if lines
        }

        return ResumeSections(
            raw_text=text,
            section_texts=section_texts,
            detected_headers=detected_headers,
        )

    def _match_header(self, line: str) -> Optional[SectionType]:
        """Checks if a single line matches any standard section heading."""
        if len(line) > 50:
            return None

        # Clean punctuation and check against patterns
        candidate = re.sub(r"^[#*\-_\s]+|[#*\-_\s]+$", "", line).strip()
        for sec_type, pattern in self.patterns:
            if pattern.match(candidate):
                return sec_type

        return None
