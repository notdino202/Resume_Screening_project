"""Job description parsing, validation, and modeling package."""

from .models import JobDescription
from .parser import JobDescriptionParser

__all__ = ["JobDescription", "JobDescriptionParser"]
