"""Text cleaning, normalization, and quality validation module."""

from __future__ import annotations
from dataclasses import dataclass, field
import re
import unicodedata


@dataclass
class CleanedTextResult:
    """Represents cleaned document text along with quality diagnostics."""
    raw_text: str
    cleaned_text: str
    char_count: int
    word_count: int
    alphanumeric_ratio: float
    quality_score: float
    is_low_quality: bool
    quality_flags: list[str] = field(default_factory=list)


class TextCleaner:
    """Standardizes document text and computes extraction quality metrics."""

    def __init__(self, min_text_length: int = 100, low_quality_threshold: float = 0.60):
        self.min_text_length = min_text_length
        self.low_quality_threshold = low_quality_threshold

    def clean(self, raw_text: str) -> CleanedTextResult:
        """Cleans raw extracted text and assesses extraction quality.

        Args:
            raw_text: Raw string extracted from document.

        Returns:
            CleanedTextResult containing normalized text and quality metrics.
        """
        flags: list[str] = []
        if not raw_text or not raw_text.strip():
            return CleanedTextResult(
                raw_text=raw_text or "",
                cleaned_text="",
                char_count=0,
                word_count=0,
                alphanumeric_ratio=0.0,
                quality_score=0.0,
                is_low_quality=True,
                quality_flags=["EMPTY_TEXT", "low-quality extraction"],
            )

        # Unicode normalization (NFKC)
        normalized = unicodedata.normalize("NFKC", raw_text)

        # Normalize common unicode punctuation
        normalized = normalized.replace("“", '"').replace("”", '"')
        normalized = normalized.replace("‘", "'").replace("’", "'")
        normalized = normalized.replace("—", "-").replace("–", "-")
        normalized = re.sub(r"[\u2022\u25cf\u25aa\u25b6\u2023\u25e6\u2043\u2219]", " - ", normalized)

        # Remove null bytes and non-printable control characters except \n, \r, \t
        cleaned_chars = [
            ch for ch in normalized
            if ch in ("\n", "\r", "\t") or (ord(ch) >= 32 and ord(ch) != 127)
        ]
        cleaned = "".join(cleaned_chars)

        # Standardize line breaks
        cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")

        # Normalize horizontal whitespace per line
        cleaned_lines = []
        for line in cleaned.split("\n"):
            stripped_line = re.sub(r"[ \t]+", " ", line).strip()
            cleaned_lines.append(stripped_line)

        # Collapse 3+ newlines into 2 (preserving section breaks)
        cleaned = "\n".join(cleaned_lines)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()

        # Compute quality metrics
        char_count = len(cleaned)
        words = re.findall(r"\b\w+\b", cleaned)
        word_count = len(words)

        if char_count == 0:
            alnum_ratio = 0.0
            quality_score = 0.0
            flags.extend(["EMPTY_AFTER_CLEANING", "low-quality extraction"])
        else:
            alnum_count = sum(1 for c in cleaned if c.isalnum())
            alnum_ratio = alnum_count / char_count

            # Quality scoring heuristic: based on length, alphanumeric ratio, word count
            len_score = min(char_count / 300.0, 1.0)
            alnum_score = min(alnum_ratio / 0.70, 1.0)
            quality_score = round(0.5 * len_score + 0.5 * alnum_score, 3)

            if char_count < self.min_text_length:
                flags.append("TEXT_TOO_SHORT")
            if alnum_ratio < 0.40:
                flags.append("LOW_ALPHANUMERIC_DENSITY")
            if quality_score < self.low_quality_threshold:
                flags.append("low-quality extraction")

        is_low_quality = len(flags) > 0

        return CleanedTextResult(
            raw_text=raw_text,
            cleaned_text=cleaned,
            char_count=char_count,
            word_count=word_count,
            alphanumeric_ratio=round(alnum_ratio, 3),
            quality_score=quality_score,
            is_low_quality=is_low_quality,
            quality_flags=flags,
        )
