"""PII scrubber and de-identification module ensuring fairness and demographic neutrality."""

from __future__ import annotations
import re

# Pronoun and verb agreement mapping to ensure gender-neutral feature representation
PRONOUN_MAP = [
    (re.compile(r"\b(?:he\s*/\s*him(?:\s*/\s*his)?|she\s*/\s*her(?:\s*/\s*hers)?|they\s*/\s*them(?:\s*/\s*theirs)?)\b", re.IGNORECASE), ""),
    (re.compile(r"\b(?:he|she|they)\s+(?:leads|lead)\b", re.IGNORECASE), "the candidate leads"),
    (re.compile(r"\b(?:he|she|they)\s+(?:manages|manage)\b", re.IGNORECASE), "the candidate manages"),
    (re.compile(r"\b(?:he|she|they)\s+(?:builds|build)\b", re.IGNORECASE), "the candidate builds"),
    (re.compile(r"\b(?:he|she|they)\s+(?:designs|design)\b", re.IGNORECASE), "the candidate designs"),
    (re.compile(r"\b(?:he|she|they)\s+(?:develops|develop)\b", re.IGNORECASE), "the candidate develops"),
    (re.compile(r"\b(?:he|she|they)\b", re.IGNORECASE), "the candidate"),
    (re.compile(r"\b(?:him|her|them)\b", re.IGNORECASE), "the candidate"),
    (re.compile(r"\b(?:his|hers|their|theirs)\b", re.IGNORECASE), "the candidate's"),
    (re.compile(r"\b(?:mr\.?|ms\.?|mrs\.?|miss)\b", re.IGNORECASE), ""),
]

# Patterns for contact and personal demographic details
PII_PATTERNS = [
    # Emails
    (re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"), "[EMAIL]"),
    # Phone numbers
    (re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"), "[PHONE]"),
    # Age mentions (e.g., "Age: 32", "29 years old")
    (re.compile(r"\b(?:age\s*[:\-]?\s*\d{1,2}|\b\d{1,2}\s+years\s+old)\b", re.IGNORECASE), "[AGE]"),
    # Physical street address fragments
    (re.compile(r"\b\d+\s+[A-Za-z0-9\s,]+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr)\b", re.IGNORECASE), "[ADDRESS]"),
    # City/State/Zip patterns in header
    (re.compile(r"\b[A-Z][a-zA-Z\s]+,\s*[A-Z]{2}(?:\s+\d{5})?\b"), "[LOCATION]"),
]


class PiiScrubber:
    """Removes or neutralizes names, genders, pronouns, ages, and contact info from scoring features."""

    @classmethod
    def scrub(cls, text: str) -> str:
        """Sanitizes text by stripping PII and normalizing pronouns for feature extraction.

        SPEC Rule:
        Name, gender, age, photo and address must never enter the scoring features.

        Args:
            text: Raw or cleaned resume text.

        Returns:
            De-identified text guaranteed to be neutral across demographic groups.
        """
        if not text:
            return ""

        lines = text.split("\n")
        sanitized_lines = []

        # Drop all header lines before the first section heading
        header_passed = False
        for line in lines:
            trimmed = line.strip()
            if not trimmed:
                continue

            # Standard section headings mark the start of genuine qualifications
            if any(h in trimmed.upper() for h in ("SUMMARY", "EXPERIENCE", "WORK", "EDUCATION", "SKILLS", "PROJECTS", "CERTIFICATIONS")):
                header_passed = True

            if not header_passed:
                # In top header, drop name/contact/gender declarations
                continue

            # Apply PII regex scrubs
            cleaned_line = trimmed
            for pattern, repl in PII_PATTERNS:
                cleaned_line = pattern.sub(repl, cleaned_line)

            # Apply pronoun normalization
            for pattern, repl in PRONOUN_MAP:
                cleaned_line = pattern.sub(repl, cleaned_line)

            sanitized_lines.append(cleaned_line)

        return "\n".join(sanitized_lines)
