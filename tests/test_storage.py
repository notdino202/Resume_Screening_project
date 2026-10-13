"""Unit tests for SQLite storage layer and profile caching."""

import pytest
from src.jd.models import JobDescription
from src.storage.db import ScreeningDatabase
from src.scoring.scorer import CandidateScoreResult


@pytest.fixture
def db(tmp_path):
    db_file = tmp_path / "test_screening.sqlite"
    return ScreeningDatabase(db_path=db_file)


def test_job_description_crud(db):
    jd = JobDescription(
        title="ML Engineer",
        raw_text="Job description raw text",
        must_have_skills=["Python", "PyTorch"],
        nice_to_have_skills=["Docker"],
        required_years=4.0,
        required_education_level=3,
        required_education_name="Bachelor's",
        requirements=["Build neural networks"],
        weights_override={"skill": 0.4, "semantic": 0.3, "experience": 0.2, "education": 0.1},
    )

    db.save_job_description("job_123", jd)
    retrieved = db.get_job_description("job_123")

    assert retrieved is not None
    assert retrieved.title == jd.title
    assert retrieved.must_have_skills == jd.must_have_skills
    assert retrieved.required_years == 4.0
    assert retrieved.weights_override["skill"] == 0.4


def test_resume_hash_caching(db):
    file_hash = "abc123hash"
    db.cache_parsed_resume(
        file_hash=file_hash,
        file_name="resume.pdf",
        file_type="PDF",
        clean_text="Clean resume content",
        sections_dict={"EXPERIENCE": "5 years at TechCorp"},
        skills_dict={"Python": {"evidence": 1.0}},
        education_dict={"level": 4},
        experience_dict={"years": 5.0},
        quality_flags=[],
    )

    cached = db.get_cached_resume(file_hash)
    assert cached is not None
    assert cached["file_name"] == "resume.pdf"
    assert cached["skills"]["Python"]["evidence"] == 1.0


def test_save_and_retrieve_screening_results(db):
    jd = JobDescription(title="Dev", raw_text="...", must_have_skills=["Python"])
    db.save_job_description("job_test", jd)

    cand = CandidateScoreResult(
        candidate_id="c1",
        file_name="c1.pdf",
        final_score=85.5,
        tier="Strong",
        rank=1,
        skill_score=0.90,
        semantic_score=0.80,
        experience_score=1.0,
        education_score=1.0,
        must_have_coverage=1.0,
        nice_to_have_coverage=0.8,
        advisory_flags=[],
        explanation="Candidate is a strong fit.",
    )

    db.save_screening_results("job_test", [cand], {"c1.pdf": "hash_1"})
    results = db.get_screening_results_for_job("job_test")

    assert len(results) == 1
    assert results[0]["file_name"] == "c1.pdf"
    assert results[0]["final_score"] == 85.5
    assert results[0]["tier"] == "Strong"
