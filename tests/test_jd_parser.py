"""Unit and integration tests for Phase 4: Job Description parser on 5 sample JDs."""

from pathlib import Path
import pytest

from src.jd.models import JobDescription
from src.jd.parser import JobDescriptionParser

JD_DIR = Path("data/sample_jds")


@pytest.fixture
def parser():
    return JobDescriptionParser()


def test_jd_model_serialization_and_weights_validation():
    jd = JobDescription(
        title="ML Lead",
        raw_text="Sample text",
        must_have_skills=["Python", "PyTorch"],
        nice_to_have_skills=["Docker"],
        required_years=5.0,
        required_education_level=4,
        required_education_name="Master's",
        requirements=["Deploy PyTorch models"],
        weights_override={"skill": 0.4, "semantic": 0.3, "experience": 0.2, "education": 0.1},
    )

    # Valid weights test
    assert jd.validate_weights() is True

    # Invalid weights test (does not sum to 1.0)
    jd_invalid = JobDescription(
        title="Role",
        raw_text="",
        weights_override={"skill": 0.5, "semantic": 0.3, "experience": 0.1, "education": 0.05},
    )
    with pytest.raises(ValueError):
        jd_invalid.validate_weights()

    # JSON serialization roundtrip
    json_str = jd.to_json()
    reconstructed = JobDescription.from_json(json_str)
    assert reconstructed.title == jd.title
    assert reconstructed.must_have_skills == jd.must_have_skills
    assert reconstructed.required_years == jd.required_years
    assert reconstructed.required_education_level == jd.required_education_level


def test_parse_jd_01_senior_ml_engineer(parser):
    text = (JD_DIR / "jd_01_senior_ml_engineer.txt").read_text(encoding="utf-8")
    jd = parser.parse(text)

    assert "Machine Learning Engineer" in jd.title
    assert jd.required_years == 5.0
    # Education: Master's mentioned in minimum qualifications
    assert jd.required_education_level == 4
    assert jd.required_education_name == "Master's"

    # Must-have skills
    assert "Python" in jd.must_have_skills
    assert "PyTorch" in jd.must_have_skills
    assert "BERT" in jd.must_have_skills
    assert "Docker" in jd.must_have_skills
    assert "FastAPI" in jd.must_have_skills

    # Nice-to-have skills
    assert "FAISS" in jd.nice_to_have_skills or "AWS" in jd.nice_to_have_skills
    assert "C++" in jd.nice_to_have_skills or "CUDA" in jd.nice_to_have_skills

    assert len(jd.requirements) >= 4


def test_parse_jd_02_data_scientist(parser):
    text = (JD_DIR / "jd_02_data_scientist.txt").read_text(encoding="utf-8")
    jd = parser.parse(text)

    assert "Data Scientist" in jd.title
    assert jd.required_years == 4.0
    assert jd.required_education_level == 3  # Bachelor's
    assert "Python" in jd.must_have_skills
    assert "SQL" in jd.must_have_skills
    assert "Pandas" in jd.must_have_skills
    assert "Scikit-learn" in jd.must_have_skills
    assert "Tableau" in jd.must_have_skills

    assert "R" in jd.nice_to_have_skills or "AWS" in jd.nice_to_have_skills


def test_parse_jd_03_fullstack_engineer(parser):
    text = (JD_DIR / "jd_03_fullstack_engineer.txt").read_text(encoding="utf-8")
    jd = parser.parse(text)

    assert "Full Stack Engineer" in jd.title
    assert jd.required_years == 4.0
    assert jd.required_education_level == 3  # Bachelor's
    assert "React" in jd.must_have_skills
    assert "Node.js" in jd.must_have_skills
    assert "JavaScript" in jd.must_have_skills
    assert "PostgreSQL" in jd.must_have_skills

    assert "Next.js" in jd.nice_to_have_skills or "Kubernetes" in jd.nice_to_have_skills


def test_parse_jd_04_devops_engineer(parser):
    text = (JD_DIR / "jd_04_devops_engineer.txt").read_text(encoding="utf-8")
    jd = parser.parse(text)

    assert "DevOps Engineer" in jd.title
    assert jd.required_years == 5.0
    assert jd.required_education_level == 3  # Bachelor's
    assert "Kubernetes" in jd.must_have_skills
    assert "Terraform" in jd.must_have_skills
    assert "Ansible" in jd.must_have_skills
    assert "Docker" in jd.must_have_skills

    assert "Prometheus" in jd.nice_to_have_skills or "Grafana" in jd.nice_to_have_skills or "Helm" in jd.nice_to_have_skills


def test_parse_jd_05_junior_qa_tester(parser):
    text = (JD_DIR / "jd_05_junior_qa_tester.txt").read_text(encoding="utf-8")
    jd = parser.parse(text)

    assert "QA" in jd.title or "Tester" in jd.title
    assert jd.required_years == 1.0
    assert jd.required_education_level == 2  # Associate degree / Diploma
    assert "Python" in jd.must_have_skills
    assert "Selenium" in jd.must_have_skills
    assert "Pytest" in jd.must_have_skills

    assert "HTML5" in jd.nice_to_have_skills or "JavaScript" in jd.nice_to_have_skills
