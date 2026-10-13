"""Unit tests for Phase 5 baseline ranking models."""

import pytest
from src.matching.baselines import TfIdfMatcher, SkillOverlapMatcher
from src.jd.models import JobDescription


@pytest.fixture
def tfidf():
    return TfIdfMatcher()


@pytest.fixture
def skill_matcher():
    return SkillOverlapMatcher()


def test_tfidf_scoring(tfidf):
    jd_text = "Senior Python developer with PyTorch and Machine Learning experience."
    resume_good = "Experienced ML Engineer working in Python and PyTorch building neural nets."
    resume_irrelevant = "Frontend developer skilled in HTML, CSS, JavaScript, and Figma."

    score_good = tfidf.score_pair(jd_text, resume_good)
    score_irrel = tfidf.score_pair(jd_text, resume_irrelevant)

    assert 0.0 <= score_good <= 1.0
    assert 0.0 <= score_irrel <= 1.0
    assert score_good > score_irrel, "Relevant ML resume must score higher on TF-IDF than unrelated frontend resume"


def test_skill_overlap_scoring(skill_matcher):
    candidate_skills = {"Python", "PyTorch", "Docker"}
    must_have = ["Python", "PyTorch", "FastAPI"]
    nice_to_have = ["Docker", "AWS"]

    # must_have coverage: 2/3 = 0.6667
    # nice_to_have coverage: 1/2 = 0.5000
    # score = 0.75 * (2/3) + 0.25 * (1/2) = 0.50 + 0.125 = 0.625
    score, matched_must, matched_nice = skill_matcher.score_skills(
        candidate_skills, must_have, nice_to_have
    )

    assert matched_must == {"Python", "PyTorch"}
    assert matched_nice == {"Docker"}
    assert 0.62 <= score <= 0.63


def test_baseline_ranking_order(tfidf, skill_matcher):
    jd = JobDescription(
        title="ML Engineer",
        raw_text="Machine Learning Python PyTorch Docker",
        must_have_skills=["Python", "PyTorch"],
        nice_to_have_skills=["Docker"],
    )

    candidates_text = [
        ("c1", "ml_lead.txt", "Expert Machine Learning Python PyTorch Docker engineer"),
        ("c2", "qa_junior.txt", "Manual QA tester Postman Jira"),
    ]

    tfidf_ranks = tfidf.rank_candidates(jd.raw_text, candidates_text)
    assert tfidf_ranks[0].candidate_id == "c1"
    assert tfidf_ranks[0].rank == 1
    assert tfidf_ranks[0].score > tfidf_ranks[1].score

    candidates_skills = [
        ("c1", "ml_lead.txt", {"Python": None, "PyTorch": None, "Docker": None}),
        ("c2", "qa_junior.txt", {"Postman": None, "JIRA": None}),
    ]

    skill_ranks = skill_matcher.rank_candidates(jd, candidates_skills)
    assert skill_ranks[0].candidate_id == "c1"
    assert skill_ranks[0].score == 1.0
    assert skill_ranks[1].score == 0.0
