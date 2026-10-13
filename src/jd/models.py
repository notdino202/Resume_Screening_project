"""Data structures and validation for parsed and editable Job Descriptions."""

from __future__ import annotations
from dataclasses import asdict, dataclass, field
import json
from typing import Optional


@dataclass
class JobDescription:
    """Represents a structured, recruiter-editable Job Description."""
    title: str
    raw_text: str
    must_have_skills: list[str] = field(default_factory=list)
    nice_to_have_skills: list[str] = field(default_factory=list)
    required_years: Optional[float] = None
    required_education_level: int = 3  # Default: Bachelor's (Level 3)
    required_education_name: str = "Bachelor's"
    requirements: list[str] = field(default_factory=list)
    weights_override: Optional[dict[str, float]] = None

    def validate_weights(self) -> bool:
        """Validates that weights (override or default) sum to 1.0 within tolerance."""
        if not self.weights_override:
            return True
        total = sum(self.weights_override.values())
        if not (0.999 <= total <= 1.001):
            raise ValueError(f"Weights must sum to 1.0, got {total:.4f}: {self.weights_override}")
        return True

    def to_dict(self) -> dict:
        """Serializes to a plain Python dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> JobDescription:
        """Constructs an instance from a dictionary."""
        return cls(
            title=data.get("title", "Untitled Role"),
            raw_text=data.get("raw_text", ""),
            must_have_skills=list(data.get("must_have_skills", [])),
            nice_to_have_skills=list(data.get("nice_to_have_skills", [])),
            required_years=data.get("required_years"),
            required_education_level=int(data.get("required_education_level", 3)),
            required_education_name=data.get("required_education_name", "Bachelor's"),
            requirements=list(data.get("requirements", [])),
            weights_override=data.get("weights_override"),
        )

    def to_json(self, indent: int = 2) -> str:
        """Serializes to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> JobDescription:
        """Constructs an instance from JSON string."""
        return cls.from_dict(json.loads(json_str))
