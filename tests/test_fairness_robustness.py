"""Fairness and robustness tests: Name/pronoun invariance, adversarial keyword-stuffing resistance, scanned resumes, and odd layouts."""

from pathlib import Path
import pytest

from src.ingestion.extractor import DocumentExtractor
from src.extraction.section_detector import SectionDetector
from src.extraction.skills_matcher import SkillsMatcher
from src.extraction.education_parser import EducationParser
from src.extraction.experience_parser import ExperienceParser
from src.jd.parser import JobDescriptionParser
from src.scoring.scorer import HybridScorer


@pytest.fixture
def pipeline():
    extractor = DocumentExtractor()
    detector = SectionDetector()
    matcher = SkillsMatcher()
    edu_parser = EducationParser()
    exp_parser = ExperienceParser()
    jd_parser = JobDescriptionParser(skills_matcher=matcher, education_parser=edu_parser)
    scorer = HybridScorer(edu_parser=edu_parser, exp_parser=exp_parser)

    jd_text = Path("data/sample_jds/jd_01_senior_ml_engineer.txt").read_text(encoding="utf-8")
    jd = jd_parser.parse(jd_text)

    return {
        "extractor": extractor,
        "detector": detector,
        "matcher": matcher,
        "edu_parser": edu_parser,
        "exp_parser": exp_parser,
        "scorer": scorer,
        "jd": jd,
    }


def score_text(pipeline, text: str, fname: str = "candidate.txt"):
    ext_res = pipeline["extractor"].extract(text.encode("utf-8"), file_name=fname)
    sections = pipeline["detector"].detect_sections(ext_res.clean_text)
    skills = pipeline["matcher"].extract_skills_from_sections(sections)
    edu = pipeline["edu_parser"].parse_education(sections.education_text or ext_res.clean_text)
    exp = pipeline["exp_parser"].parse_experience(sections.experience_text or ext_res.clean_text)
    return pipeline["scorer"].score_candidate(
        candidate_id=fname,
        file_name=fname,
        clean_text=ext_res.clean_text,
        extracted_skills=skills,
        candidate_edu=edu,
        candidate_exp=exp,
        jd=pipeline["jd"],
        extraction_flags=ext_res.advisory_flags,
    )


def test_name_and_pronoun_swap_invariance(pipeline):
    """SPEC Constraint: Name, gender, age, photo, and address must never enter scoring features.

    Scores for identical qualifications under different names and pronouns MUST MATCH EXACTLY.
    """
    resume_male = """JOHNATHAN DOE
He / Him | 123 Main St, New York, NY | Age: 34
SUMMARY
Senior Machine Learning Engineer with 6 years experience in NLP and deep learning. He leads ML infrastructure.

EXPERIENCE
Senior ML Engineer | TechCorp | Jan 2021 - Present
- Architected inference pipelines in Python, PyTorch, and Hugging Face Transformers.
- Fine-tuned BERT and Sentence-Transformers models.
- Deployed microservices with Docker and FastAPI on AWS.

EDUCATION
Master of Science in Computer Science | Stanford University | 2017 - 2019

SKILLS
Python, PyTorch, Transformers, BERT, Docker, FastAPI, AWS, Linux
"""

    resume_female = """JACQUELINE SMITH
She / Her | 456 Elm Ave, Chicago, IL | Age: 28
SUMMARY
Senior Machine Learning Engineer with 6 years experience in NLP and deep learning. She leads ML infrastructure.

EXPERIENCE
Senior ML Engineer | TechCorp | Jan 2021 - Present
- Architected inference pipelines in Python, PyTorch, and Hugging Face Transformers.
- Fine-tuned BERT and Sentence-Transformers models.
- Deployed microservices with Docker and FastAPI on AWS.

EDUCATION
Master of Science in Computer Science | Stanford University | 2017 - 2019

SKILLS
Python, PyTorch, Transformers, BERT, Docker, FastAPI, AWS, Linux
"""

    resume_neutral = """ALEX TAYLOR
They / Them | Austin, TX
SUMMARY
Senior Machine Learning Engineer with 6 years experience in NLP and deep learning. They lead ML infrastructure.

EXPERIENCE
Senior ML Engineer | TechCorp | Jan 2021 - Present
- Architected inference pipelines in Python, PyTorch, and Hugging Face Transformers.
- Fine-tuned BERT and Sentence-Transformers models.
- Deployed microservices with Docker and FastAPI on AWS.

EDUCATION
Master of Science in Computer Science | Stanford University | 2017 - 2019

SKILLS
Python, PyTorch, Transformers, BERT, Docker, FastAPI, AWS, Linux
"""

    score_male = score_text(pipeline, resume_male, "male.txt")
    score_female = score_text(pipeline, resume_female, "female.txt")
    score_neutral = score_text(pipeline, resume_neutral, "neutral.txt")

    assert score_male.final_score == score_female.final_score, "Gender/name swap caused score divergence!"
    assert score_male.final_score == score_neutral.final_score, "Pronoun swap caused score divergence!"
    assert score_male.skill_score == score_female.skill_score
    assert score_male.experience_score == score_female.experience_score
    assert score_male.education_score == score_female.education_score


def test_keyword_stuffed_resume_cannot_outrank_genuine(pipeline):
    """Adversarial Test: Keyword-stuffed resume must NOT outrank genuine experienced candidate."""
    # 1. Genuine experienced candidate
    genuine_path = Path("data/sample_resumes/resume_01_ml_senior.pdf")
    ext_genuine = pipeline["extractor"].extract(genuine_path)
    sec_gen = pipeline["detector"].detect_sections(ext_genuine.clean_text)
    sk_gen = pipeline["matcher"].extract_skills_from_sections(sec_gen)
    edu_gen = pipeline["edu_parser"].parse_education(sec_gen.education_text)
    exp_gen = pipeline["exp_parser"].parse_experience(sec_gen.experience_text)
    score_genuine = pipeline["scorer"].score_candidate(
        "genuine", "genuine.pdf", ext_genuine.clean_text, sk_gen, edu_gen, exp_gen, pipeline["jd"]
    )

    # 2. Adversarial keyword-stuffed resume:
    # A candidate with 1 year experience who stuffed every possible keyword in a block
    stuffed_resume = """NOVICE CANDIDATE
SUMMARY
Beginner seeking any job.

EXPERIENCE
Intern | SmallShop | Jan 2024 - Dec 2024
- Assisted with general office computer tasks and spreadsheets.

EDUCATION
High School Diploma | Local School | 2020 - 2023

SKILLS & KEYWORDS
BERT BERT BERT Docker Docker FastAPI FastAPI Hugging Face LLM Machine Learning Natural Language Processing PyTorch Python Sentence-Transformers Transformers AWS Apache Airflow C++ CUDA Deep Learning FAISS RAG Kubernetes Linux Terraform React Angular Node.js SQL MongoDB Redis PostgreSQL GraphQL
"""
    score_stuffed = score_text(pipeline, stuffed_resume, "stuffed.txt")

    assert score_genuine.final_score > score_stuffed.final_score, "Adversarial keyword stuffer outranked genuine engineer!"
    assert score_genuine.tier == "Strong"
    assert score_stuffed.tier == "Weak"
    # Should flag missing experience / low experience / education below requirement
    assert any("low experience" in f or "education below requirement" in f for f in score_stuffed.advisory_flags)


def test_scanned_resume_handling(pipeline):
    """Robustness Test: Scanned or near-empty text PDF triggers advisory flags without crashing."""
    scanned_path = Path("data/sample_resumes/edge_scanned_resume.pdf")
    ext_scanned = pipeline["extractor"].extract(scanned_path)
    assert ext_scanned.is_success is True
    assert "low-quality extraction" in ext_res_flags if False else "low-quality extraction" in ext_scanned.advisory_flags

    sec_scan = pipeline["detector"].detect_sections(ext_scanned.clean_text)
    sk_scan = pipeline["matcher"].extract_skills_from_sections(sec_scan)
    edu_scan = pipeline["edu_parser"].parse_education(sec_scan.education_text)
    exp_scan = pipeline["exp_parser"].parse_experience(sec_scan.experience_text)

    score_res = pipeline["scorer"].score_candidate(
        "scanned", "scanned.pdf", ext_scanned.clean_text, sk_scan, edu_scan, exp_scan, pipeline["jd"],
        extraction_flags=ext_scanned.advisory_flags,
    )
    assert score_res.tier == "Weak"
    assert "low-quality extraction" in score_res.advisory_flags


def test_odd_layout_handling(pipeline):
    """Robustness Test: Non-standard headings and unconventional layout."""
    odd_resume = """MARIA SANTOS
Developer Profile
WHERE I'VE BEEN
Lead Engineer at FinApp (2020 - 2024)
Designed distributed payment pipelines in Python and PostgreSQL. Managed Docker containers on AWS.

WHAT I'M GOOD AT
Python, Docker, AWS, PostgreSQL, REST API, Git

WHERE I LEARNED
Bachelor of Science in Software Engineering, 2016-2020
"""
    score_res = score_text(pipeline, odd_resume, "odd_layout.txt")
    assert score_res.final_score > 0
    assert "Python" in [d.jd_skill for d in score_res.must_have_details if d.final_skill_score > 0]
    assert "Docker" in [d.jd_skill for d in score_res.must_have_details if d.final_skill_score > 0]
