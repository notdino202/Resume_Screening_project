"""Experience extraction, date parsing, interval merging, and experience scoring module."""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
import re
from typing import Optional, Sequence

MONTH_NAME_TO_NUM = {
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}

# Regex to match date ranges like:
# "Jan 2021 - Present", "03/2019 - 05/2021", "2018 - 2020", "Mar 2020 to Current"
DATE_RANGE_REGEX = re.compile(
    r"\b((?:(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+)?(?:19|20)\d{2}|\d{1,2}/\d{4}|\d{1,2}/(?:19|20)\d{2})\s*(?:[-–—]|to)\s*((?:(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+)?(?:19|20)\d{2}|\d{1,2}/\d{4}|\d{1,2}/(?:19|20)\d{2}|present|current|now)\b",
    re.IGNORECASE,
)


@dataclass(order=True)
class DateInterval:
    """Represents a discrete employment interval."""
    start_date: datetime
    end_date: datetime
    raw_text: str = field(compare=False, default="")
    is_relevant: bool = field(compare=False, default=True)

    @property
    def duration_years(self) -> float:
        """Computes duration in decimal years."""
        if self.end_date <= self.start_date:
            return 0.0
        return (self.end_date - self.start_date).days / 365.25


@dataclass
class CandidateExperience:
    """Summary of parsed candidate work history and years."""
    raw_intervals: list[DateInterval] = field(default_factory=list)
    merged_intervals: list[DateInterval] = field(default_factory=list)
    total_years: float = 0.0
    relevant_years: float = 0.0


class ExperienceParser:
    """Extracts job dates, merges overlapping intervals, and calculates experience scores."""

    def __init__(self, reference_date: Optional[datetime] = None):
        """Initializes parser with optional fixed reference date for determinism."""
        self.reference_date = reference_date or datetime.now()

    def parse_date_token(self, token: str, is_end: bool = False) -> datetime:
        """Parses a single date token into a datetime instance."""
        token_str = token.strip().lower()

        if token_str in ("present", "current", "now"):
            return self.reference_date

        # Format MM/YYYY
        if "/" in token_str:
            parts = token_str.split("/")
            if len(parts) == 2:
                try:
                    month = int(parts[0])
                    year = int(parts[1])
                    day = 28 if is_end else 1
                    return datetime(year, month, day)
                except (ValueError, IndexError):
                    pass

        # Format Month Year, e.g. "Jan 2021", "August 2019"
        words = token_str.split()
        if len(words) == 2:
            m_str, y_str = words[0], words[1]
            if m_str in MONTH_NAME_TO_NUM and y_str.isdigit():
                month = MONTH_NAME_TO_NUM[m_str]
                year = int(y_str)
                day = 28 if is_end else 1
                return datetime(year, month, day)

        # Format Year only, e.g. "2020"
        if token_str.isdigit() and len(token_str) == 4:
            year = int(token_str)
            if is_end:
                return datetime(year, 12, 31)
            return datetime(year, 1, 1)

        # Fallback to reference date if unrecognized
        return self.reference_date

    def extract_intervals(self, text: str) -> list[DateInterval]:
        """Finds all employment date ranges in the given text."""
        intervals: list[DateInterval] = []
        if not text:
            return intervals

        for match in DATE_RANGE_REGEX.finditer(text):
            start_str, end_str = match.group(1), match.group(2)
            try:
                start_dt = self.parse_date_token(start_str, is_end=False)
                end_dt = self.parse_date_token(end_str, is_end=True)
                if end_dt >= start_dt:
                    intervals.append(
                        DateInterval(
                            start_date=start_dt,
                            end_date=end_dt,
                            raw_text=match.group(0),
                            is_relevant=True,
                        )
                    )
            except Exception:
                continue

        return intervals

    @staticmethod
    def merge_overlapping_intervals(intervals: Sequence[DateInterval]) -> list[DateInterval]:
        """Merges overlapping or contiguous date intervals to avoid double-counting.

        Args:
            intervals: Sequence of DateIntervals.

        Returns:
            Sorted, non-overlapping list of merged DateIntervals.
        """
        if not intervals:
            return []

        sorted_intervals = sorted(intervals, key=lambda i: (i.start_date, i.end_date))
        merged: list[DateInterval] = []

        curr_start = sorted_intervals[0].start_date
        curr_end = sorted_intervals[0].end_date
        raw_parts = [sorted_intervals[0].raw_text]
        is_rel = sorted_intervals[0].is_relevant

        for item in sorted_intervals[1:]:
            # If item overlaps or is contiguous with current interval
            if item.start_date <= curr_end:
                curr_end = max(curr_end, item.end_date)
                raw_parts.append(item.raw_text)
                is_rel = is_rel or item.is_relevant
            else:
                merged.append(
                    DateInterval(
                        start_date=curr_start,
                        end_date=curr_end,
                        raw_text="; ".join(raw_parts),
                        is_relevant=is_rel,
                    )
                )
                curr_start = item.start_date
                curr_end = item.end_date
                raw_parts = [item.raw_text]
                is_rel = item.is_relevant

        merged.append(
            DateInterval(
                start_date=curr_start,
                end_date=curr_end,
                raw_text="; ".join(raw_parts),
                is_relevant=is_rel,
            )
        )

        return merged

    def parse_experience(self, text: str, relevant_keywords: Optional[list[str]] = None) -> CandidateExperience:
        """Parses experience text into raw and merged intervals, computing total and relevant years.

        Args:
            text: Text to scan (typically Experience section).
            relevant_keywords: Optional list of keywords to determine job relevance.

        Returns:
            CandidateExperience dataclass.
        """
        raw_intervals = self.extract_intervals(text)

        # Merge all intervals to calculate total deduplicated experience
        merged_all = self.merge_overlapping_intervals(raw_intervals)
        total_years = sum(interval.duration_years for interval in merged_all)

        # Calculate relevant years (if keywords provided, filter; otherwise all are considered relevant)
        if relevant_keywords:
            rel_intervals = [
                i for i in raw_intervals
                if any(kw.lower() in i.raw_text.lower() for kw in relevant_keywords)
            ]
            merged_rel = self.merge_overlapping_intervals(rel_intervals)
            relevant_years = sum(interval.duration_years for interval in merged_rel)
        else:
            relevant_years = total_years

        return CandidateExperience(
            raw_intervals=raw_intervals,
            merged_intervals=merged_all,
            total_years=round(total_years, 2),
            relevant_years=round(relevant_years, 2),
        )

    @staticmethod
    def score_experience(
        candidate_years: float,
        required_years: Optional[float] = None,
        relevant_years: Optional[float] = None,
        relevant_weight: float = 0.70,
        total_weight: float = 0.30,
    ) -> tuple[float, list[str]]:
        """Computes experience score and advisory flags according to SPEC.

        SPEC rule:
        - min(effective_years / required_years, 1.0)
        - effective_years = 0.70 * relevant_years + 0.30 * total_years
        - If no requirement stated, score 1.0.

        Args:
            candidate_years: Total years extracted.
            required_years: Minimum years required from JD.
            relevant_years: Relevant years extracted (defaults to candidate_years if None).
            relevant_weight: Weight for relevant experience (default 0.70).
            total_weight: Weight for total experience (default 0.30).

        Returns:
            Tuple of (score in [0.0, 1.0], advisory_flags).
        """
        if required_years is None or required_years <= 0:
            return 1.0, []

        rel_yrs = candidate_years if relevant_years is None else relevant_years
        effective_years = (relevant_weight * rel_yrs) + (total_weight * candidate_years)

        raw_score = effective_years / required_years
        score = min(max(raw_score, 0.0), 1.0)

        flags: list[str] = []
        if effective_years < (required_years * 0.5):
            flags.append("low experience")

        return round(score, 4), flags
