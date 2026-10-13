"""Unit tests for Phase 2: Section detection and skill extraction."""

import pytest
from src.extraction.section_detector import SectionDetector, SectionType
from src.extraction.skills_matcher import SkillsMatcher


@pytest.fixture
def detector():
    return SectionDetector()


@pytest.fixture
def matcher():
    return SkillsMatcher()


def test_section_detector(detector):
    text = """JOHN DOE
SUMMARY
Experienced software engineer.

WORK EXPERIENCE
Built microservices using Python and Docker.

EDUCATION
BS in Computer Science.

TECHNICAL SKILLS
Python, Java, Docker, AWS.

CERTIFICATIONS
AWS Certified Solutions Architect
"""
    sections = detector.detect_sections(text)
    assert "software engineer" in sections.summary_text
    assert "microservices" in sections.experience_text
    assert "Computer Science" in sections.education_text
    assert "Docker" in sections.skills_text
    assert "AWS Certified" in sections.certifications_text


def test_token_boundaries_no_false_positive(matcher):
    # SPEC: "Java" is not found in "JavaScript"
    text = "We specialize in modern JavaScript development and Node.js applications."
    skills = matcher.extract_skills_from_text(text)
    assert "JavaScript" in skills
    assert "Java" not in skills, "False positive! 'Java' must not match inside 'JavaScript'"


def test_alias_normalization(matcher):
    # SPEC: aliases normalize ("JS" to "JavaScript", "K8s" to "Kubernetes")
    text = "Built web apps using JS, TS, K8s, Postgres, and TF."
    skills = matcher.extract_skills_from_text(text)
    assert "JavaScript" in skills
    assert "TypeScript" in skills
    assert "Kubernetes" in skills
    assert "PostgreSQL" in skills
    assert "TensorFlow" in skills


def test_evidence_weighting_levels(detector, matcher):
    # SPEC:
    # Experience/Projects = 1.0
    # Certifications = 0.9
    # skills list only = 0.7
    # elsewhere = 0.5
    resume = """JANE DOE
SUMMARY
Proficient in Python and Go.

EXPERIENCE
Architected cloud backend in Go and Docker.

CERTIFICATIONS
AWS Certified Solutions Architect

TECHNICAL SKILLS
Python, Redis, PostgreSQL
"""
    sections = detector.detect_sections(resume)
    extracted = matcher.extract_skills_from_sections(sections)

    # Go is in Experience and Summary -> max weight should be 1.0
    assert extracted["Go"].evidence_factor == 1.0
    assert "EXPERIENCE" in extracted["Go"].sections_found

    # Docker is in Experience -> 1.0
    assert extracted["Docker"].evidence_factor == 1.0

    # AWS is in Certifications -> 0.9
    assert extracted["AWS"].evidence_factor == 0.9

    # Redis is in Skills list only -> 0.7
    assert extracted["Redis"].evidence_factor == 0.7
    assert extracted["Redis"].sections_found == ["SKILLS"]


def test_c_and_cpp_differentiation(matcher):
    text = "Core competencies in C++ and C programming."
    skills = matcher.extract_skills_from_text(text)
    assert "C++" in skills
    assert "C" in skills
