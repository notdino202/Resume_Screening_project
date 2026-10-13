"""Job Description parser extracting must-have/nice-to-have skills, experience, education, and semantic requirements."""

from __future__ import annotations
from pathlib import Path
import re
from typing import Optional

from src.extraction.education_parser import EducationParser
from src.extraction.skills_matcher import SkillsMatcher
from .models import JobDescription


EXP_REQ_REGEX = re.compile(
    r"\b(?:(?:minimum|at\s+least|min\.?)\s+)?(\d+(?:\.\d+)?)\s*(?:\+|-\s*\d+)?\s*(?:years?|yrs?)(?:\s+of)?(?:\s+(?:relevant|industry|professional|work|hands-on|direct|practical))*\s+experience\b",
    re.IGNORECASE,
)

MUST_HAVE_HEADERS = re.compile(
    r"^\s*(?:MINIMUM\s+QUALIFICATIONS|BASIC\s+QUALIFICATIONS|REQUIRED\s+SKILLS|REQUIREMENTS|MUST\s+HAVE|WHAT\s+YOU\s+NEED|REQUIRED\s+EXPERIENCE)\s*[:\-]?\s*$",
    re.IGNORECASE,
)

NICE_TO_HAVE_HEADERS = re.compile(
    r"^\s*(?:PREFERRED\s+QUALIFICATIONS|DESIRED\s+SKILLS|NICE\s+TO\s+HAVE|BONUS\s+POINTS|GOOD\s+TO\s+HAVE|PREFERRED\s+SKILLS|OPTIONAL\s+QUALIFICATIONS)\s*[:\-]?\s*$",
    re.IGNORECASE,
)


class JobDescriptionParser:
    """Parses raw JD text into an editable JobDescription instance."""

    def __init__(
        self,
        skills_matcher: Optional[SkillsMatcher] = None,
        education_parser: Optional[EducationParser] = None,
    ):
        self.skills_matcher = skills_matcher or SkillsMatcher()
        self.education_parser = education_parser or EducationParser()

    def parse(self, text: str, title: Optional[str] = None) -> JobDescription:
        """Parses job description text into structured components.

        Args:
            text: Raw job description text.
            title: Optional manual title override.

        Returns:
            JobDescription instance with extracted attributes.
        """
        clean_text = text.strip()
        lines = [l.strip() for l in clean_text.split("\n") if l.strip()]

        # 1. Title Extraction
        job_title = title or self._extract_title(lines)

        # 2. Required Experience Extraction (in years)
        required_years = self._extract_required_years(clean_text)

        # 4. Must-have vs Nice-to-have Skills and Requirements Extraction
        must_skills, nice_skills, requirements, must_text = self._split_skills_and_requirements(lines, clean_text)

        # 3. Required Education Level Extraction (prioritizes Must-Have qualifications)
        edu_level, edu_name = self._extract_education_requirement(must_text, clean_text)

        return JobDescription(
            title=job_title,
            raw_text=clean_text,
            must_have_skills=must_skills,
            nice_to_have_skills=nice_skills,
            required_years=required_years,
            required_education_level=edu_level,
            required_education_name=edu_name,
            requirements=requirements,
        )

    def _extract_title(self, lines: list[str]) -> str:
        """Identifies job title from top lines."""
        for line in lines[:5]:
            match = re.search(r"^(?:Job\s+Title|Position|Role)\s*[:\-]\s*(.+)$", line, re.IGNORECASE)
            if match:
                return match.group(1).strip()

        # Fallback to the first line if short and plausible
        if lines and len(lines[0]) < 60:
            return lines[0].strip()
        return "Software Professional"

    def _extract_required_years(self, text: str) -> Optional[float]:
        """Extracts required experience years (e.g. '5+ years' -> 5.0)."""
        match = EXP_REQ_REGEX.search(text)
        if match:
            try:
                return float(match.group(1))
            except (ValueError, TypeError):
                pass
        return None

    def _extract_education_requirement(self, must_text: str, full_text: str) -> tuple[int, str]:
        """Determines required education level (1-5), prioritizing Must-Have qualifications."""
        # Check minimum qualifications section first
        if must_text.strip():
            must_edu = self.education_parser.parse_education(must_text)
            if must_edu.highest_level > 0:
                return must_edu.highest_level, must_edu.highest_degree_name

        # Fallback to full text
        full_edu = self.education_parser.parse_education(full_text)
        if full_edu.highest_level > 0:
            return full_edu.highest_level, full_edu.highest_degree_name
        return 3, "Bachelor's"  # Standard industry default

    def _split_skills_and_requirements(
        self, lines: list[str], full_text: str
    ) -> tuple[list[str], list[str], list[str]]:
        """Segments lines into must-have vs nice-to-have sections and individual requirement chunks."""
        must_text_blocks: list[str] = []
        nice_text_blocks: list[str] = []
        all_req_chunks: list[str] = []

        current_zone = "MUST"  # Default assumption

        for line in lines:
            if MUST_HAVE_HEADERS.match(line):
                current_zone = "MUST"
                continue
            elif NICE_TO_HAVE_HEADERS.match(line):
                current_zone = "NICE"
                continue

            # Check if this line looks like a requirement bullet
            if line.startswith(("-", "*", "•", "▪")) or re.match(r"^\d+\.", line):
                clean_bullet = re.sub(r"^[-*•▪\d.]+\s*", "", line).strip()
                if len(clean_bullet) > 15:
                    all_req_chunks.append(clean_bullet)

            if current_zone == "NICE":
                nice_text_blocks.append(line)
            else:
                must_text_blocks.append(line)

        must_text = "\n".join(must_text_blocks)
        nice_text = "\n".join(nice_text_blocks)

        # Extract skills using matcher
        must_extracted = list(self.skills_matcher.extract_skills_from_text(must_text).keys())
        nice_extracted = list(self.skills_matcher.extract_skills_from_text(nice_text).keys())

        # If nice_to_have section was empty, check for lines mentioning 'preferred' or 'bonus'
        if not nice_extracted:
            new_must = []
            new_nice = []
            for skill in must_extracted:
                # Check if the context around the skill mentions preferred / nice to have
                pattern = rf"(?:preferred|nice\s+to\s+have|plus|bonus|optional).{{0,40}}{re.escape(skill)}"
                if re.search(pattern, full_text, re.IGNORECASE):
                    new_nice.append(skill)
                else:
                    new_must.append(skill)
            must_skills = new_must
            nice_skills = new_nice
        else:
            # Deduplicate nice-to-have against must-have
            must_skills = must_extracted
            nice_skills = [s for s in nice_extracted if s not in must_skills]

        # If no bullet requirements found, use sentences of sufficient length
        if not all_req_chunks:
            sentences = re.split(r"(?<=[.!?])\s+", full_text)
            all_req_chunks = [s.strip() for s in sentences if len(s.strip()) > 25 and not s.isupper()]

        return sorted(list(set(must_skills))), sorted(list(set(nice_skills))), all_req_chunks, must_text
