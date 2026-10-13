"""Unit and integration tests for Phase 6: Hybrid scoring engine, determinism, and explanations."""

import pytest
from src.jd.models import JobDescription
from src.extraction.education_parser import CandidateEducation, EducationEntry
from src.extraction.experience_parser import CandidateExperience
from src.extraction.skills_matcher import ExtractedSkill
from src.scoring.scorer import HybridScorer
from src.scoring.explainer import ExplanationGenerator


@pytest.fixture
def scorer():
    return HybridScorer()


@pytest.fixture
def sample_jd():
    return JobDescription(
        title="Senior ML Engineer",
        raw_text="Lead development of PyTorch and Transformers models. 5+ years experience. Master's degree required.",
        must_have_skills=["Python", "PyTorch", "Transformers"],
        nice_to_have_skills=["Docker", "AWS"],
        required_years=5.0,
        required_education_level=4,
        required_education_name="Master's",
        requirements=[
            "Design and deploy deep learning NLP pipelines with PyTorch",
            "Fine-tune Transformers and large language models",
            "Containerize microservices with Docker on cloud platforms",
        ],
    )


def test_components_in_zero_to_one(scorer, sample_jd):
    skills = {
        "Python": ExtractedSkill("Python", "Languages", "Python", 1.0, ["EXPERIENCE"]),
        "PyTorch": ExtractedSkill("PyTorch", "ML/NLP", "PyTorch", 1.0, ["EXPERIENCE"]),
    }
    edu = CandidateEducation(
        entries=[EducationEntry("Master's", 4, "Master of Science", "M.S. in CS")],
        highest_level=4,
        highest_degree_name="Master's",
    )
    exp = CandidateExperience(total_years=6.0, relevant_years=5.0)

    res = scorer.score_candidate(
        candidate_id="cand_1",
        file_name="alex_chen.pdf",
        clean_text="Senior ML Engineer with 6 years experience in PyTorch and Python.",
        extracted_skills=skills,
        candidate_edu=edu,
        candidate_exp=exp,
        jd=sample_jd,
    )

    # Every component must be in [0, 1]
    assert 0.0 <= res.skill_score <= 1.0
    assert 0.0 <= res.semantic_score <= 1.0
    assert 0.0 <= res.experience_score <= 1.0
    assert 0.0 <= res.education_score <= 1.0

    # Final score in [0, 100]
    assert 0.0 <= res.final_score <= 100.0


def test_scoring_determinism(scorer, sample_jd):
    # SPEC: Re-runs give identical scores
    skills = {
        "Python": ExtractedSkill("Python", "Languages", "Python", 1.0, ["EXPERIENCE"]),
        "PyTorch": ExtractedSkill("PyTorch", "ML/NLP", "PyTorch", 1.0, ["EXPERIENCE"]),
    }
    edu = CandidateEducation(
        entries=[EducationEntry("Master's", 4, "Master of Science", "M.S. in CS")],
        highest_level=4,
        highest_degree_name="Master's",
    )
    exp = CandidateExperience(total_years=5.0, relevant_years=5.0)
    text = "Machine learning engineer experienced in PyTorch and Python."

    run1 = scorer.score_candidate("c1", "res.pdf", text, skills, edu, exp, sample_jd)
    run2 = scorer.score_candidate("c1", "res.pdf", text, skills, edu, exp, sample_jd)

    assert run1.final_score == run2.final_score
    assert run1.skill_score == run2.skill_score
    assert run1.semantic_score == run2.semantic_score
    assert run1.experience_score == run2.experience_score
    assert run1.education_score == run2.education_score
    assert run1.tier == run2.tier
    assert run1.advisory_flags == run2.advisory_flags


def test_tier_assignments(scorer, sample_jd):
    # SPEC: Strong >= 75, Moderate 50-74.9, Weak < 50

    # Strong candidate (perfect skills, experience, education)
    all_skills = {
        "Python": ExtractedSkill("Python", "Languages", "Python", 1.0, ["EXPERIENCE"]),
        "PyTorch": ExtractedSkill("PyTorch", "ML/NLP", "PyTorch", 1.0, ["EXPERIENCE"]),
        "Transformers": ExtractedSkill("Transformers", "ML/NLP", "Transformers", 1.0, ["EXPERIENCE"]),
        "Docker": ExtractedSkill("Docker", "Cloud", "Docker", 1.0, ["EXPERIENCE"]),
        "AWS": ExtractedSkill("AWS", "Cloud", "AWS", 1.0, ["EXPERIENCE"]),
    }
    strong_edu = CandidateEducation(highest_level=4, highest_degree_name="Master's")
    strong_exp = CandidateExperience(total_years=7.0, relevant_years=6.0)
    strong_text = "Senior ML engineer with extensive PyTorch, Transformers, Docker, and AWS experience."

    strong_res = scorer.score_candidate("c_strong", "strong.pdf", strong_text, all_skills, strong_edu, strong_exp, sample_jd)
    assert strong_res.final_score >= 75.0
    assert strong_res.tier == "Strong"

    # Weak candidate (no skills, 1 year experience, High School)
    weak_edu = CandidateEducation(highest_level=1, highest_degree_name="High School")
    weak_exp = CandidateExperience(total_years=1.0, relevant_years=0.5)
    weak_text = "Junior clerk with basic computer usage."

    weak_res = scorer.score_candidate("c_weak", "weak.pdf", weak_text, {}, weak_edu, weak_exp, sample_jd)
    assert weak_res.final_score < 50.0
    assert weak_res.tier == "Weak"
    assert "missing must-have" in weak_res.advisory_flags
    assert "low experience" in weak_res.advisory_flags
    assert "education below requirement" in weak_res.advisory_flags


def test_tie_breaking(scorer, sample_jd):
    # SPEC: Break ranking ties by must-have coverage, then relevant years
    # Candidate A and Candidate B have identical final scores (mocked), but A has higher must-have coverage
    c_a = scorer.score_candidate("ca", "ca.pdf", "text a", {}, CandidateEducation(), CandidateExperience(total_years=3.0, relevant_years=3.0), sample_jd)
    c_b = scorer.score_candidate("cb", "cb.pdf", "text b", {}, CandidateEducation(), CandidateExperience(total_years=2.0, relevant_years=2.0), sample_jd)

    # Force identical score
    c_a.final_score = 65.0
    c_b.final_score = 65.0
    c_a.must_have_coverage = 0.80
    c_b.must_have_coverage = 0.60

    ranked = scorer.rank_candidates([c_b, c_a])
    assert ranked[0].candidate_id == "ca"
    assert ranked[0].rank == 1
    assert ranked[1].candidate_id == "cb"
    assert ranked[1].rank == 2


def test_explanation_generation(scorer, sample_jd):
    skills = {
        "Python": ExtractedSkill("Python", "Languages", "Python", 1.0, ["EXPERIENCE"]),
    }
    edu = CandidateEducation(highest_level=3, highest_degree_name="Bachelor's")
    exp = CandidateExperience(total_years=4.0, relevant_years=4.0)

    res = scorer.score_candidate("c1", "alex.pdf", "ML dev with Python", skills, edu, exp, sample_jd)
    explanation = ExplanationGenerator.generate_explanation(res)

    assert "Screening Assessment" in explanation
    assert "Final Score:" in explanation
    assert "Component Score Breakdown" in explanation
    assert "Matched Must-Have Skills" in explanation
    assert "Missing Must-Have Skills" in explanation
    assert "Advisory Flags" in explanation
