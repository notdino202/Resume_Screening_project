"""End-to-end demonstration of the Hybrid Resume Screening System."""

from pathlib import Path
import pandas as pd

from src.ingestion.extractor import DocumentExtractor
from src.extraction.section_detector import SectionDetector
from src.extraction.skills_matcher import SkillsMatcher
from src.extraction.education_parser import EducationParser
from src.extraction.experience_parser import ExperienceParser
from src.jd.parser import JobDescriptionParser
from src.scoring.embeddings import EmbeddingManager
from src.scoring.scorer import HybridScorer
from src.scoring.explainer import ExplanationGenerator

SAMPLE_RESUMES = [
    "resume_01_ml_senior.pdf",
    "resume_02_data_scientist.docx",
    "resume_03_fullstack_dev.txt",
    "resume_04_nlp_researcher.pdf",
    "resume_05_devops_cloud.docx",
    "resume_06_backend_python.pdf",
    "resume_07_junior_qa.docx",
    "resume_08_data_analyst.txt",
    "resume_09_cloud_architect.pdf",
    "resume_10_frontend_angular.docx",
]

def main():
    print("=" * 80)
    print("AI RESUME SCREENING SYSTEM: HYBRID SCORING PIPELINE DEMO")
    print("=" * 80)

    # Initialize shared components
    extractor = DocumentExtractor()
    detector = SectionDetector()
    matcher = SkillsMatcher()
    edu_parser = EducationParser()
    exp_parser = ExperienceParser()
    embedder = EmbeddingManager()
    jd_parser = JobDescriptionParser(skills_matcher=matcher, education_parser=edu_parser)
    scorer = HybridScorer(embedder=embedder, edu_parser=edu_parser, exp_parser=exp_parser)

    # 1. Parse JD 1 (Senior Machine Learning Engineer)
    jd_file = Path("data/sample_jds/jd_01_senior_ml_engineer.txt")
    jd = jd_parser.parse(jd_file.read_text(encoding="utf-8"))

    print(f"\nTARGET JOB: {jd.title}")
    print(f"Requirements Count: {len(jd.requirements)} statements")
    print(f"Required Experience: {jd.required_years} years")
    print(f"Required Education: Level {jd.required_education_level} ({jd.required_education_name})")
    print(f"Must-Have Skills ({len(jd.must_have_skills)}): {', '.join(jd.must_have_skills)}")
    print(f"Nice-To-Have Skills ({len(jd.nice_to_have_skills)}): {', '.join(jd.nice_to_have_skills)}")
    print("-" * 80)

    # 2. Process and score all candidates
    candidate_results = []

    for fname in SAMPLE_RESUMES:
        fpath = Path("data/sample_resumes") / fname
        ext_res = extractor.extract(fpath)
        sections = detector.detect_sections(ext_res.clean_text)
        skills = matcher.extract_skills_from_sections(sections)
        edu = edu_parser.parse_education(sections.education_text or ext_res.clean_text)
        exp = exp_parser.parse_experience(sections.experience_text or ext_res.clean_text)

        score_res = scorer.score_candidate(
            candidate_id=fname.replace(".pdf", "").replace(".docx", "").replace(".txt", ""),
            file_name=fname,
            clean_text=ext_res.clean_text,
            extracted_skills=skills,
            candidate_edu=edu,
            candidate_exp=exp,
            jd=jd,
            extraction_flags=ext_res.advisory_flags,
        )
        candidate_results.append(score_res)

    # 3. Apply SPEC tie-breaking & ranking
    ranked_candidates = scorer.rank_candidates(candidate_results)
    for c in ranked_candidates:
        c.explanation = ExplanationGenerator.generate_explanation(c)

    # 4. Display Leaderboard Table
    table_data = []
    for c in ranked_candidates:
        table_data.append({
            "Rank": f"#{c.rank}",
            "Candidate File": c.file_name,
            "Final Score": f"{c.final_score:.1f}",
            "Tier": c.tier,
            "Skill (40%)": f"{c.skill_score*100:.1f}%",
            "Semantic (30%)": f"{c.semantic_score*100:.1f}%",
            "Exp (20%)": f"{c.experience_score*100:.1f}%",
            "Edu (10%)": f"{c.education_score*100:.1f}%",
            "Must Coverage": f"{c.must_have_coverage*100:.1f}%",
            "Flags": ", ".join(c.advisory_flags) if c.advisory_flags else "None",
        })

    df = pd.DataFrame(table_data)
    print("\nRANKED CANDIDATE LEADERBOARD:")
    print(df.to_string(index=False))

    # 5. Print Detailed Explanation for Top Candidate and a Moderate/Weak Candidate
    print("\n" + "=" * 80)
    print("DETAILED EXPLANATION FOR TOP-RANKED CANDIDATE:")
    print("=" * 80)
    print(ranked_candidates[0].explanation)

    print("\n" + "=" * 80)
    print("DETAILED EXPLANATION FOR LOWER-RANKED CANDIDATE:")
    print("=" * 80)
    print(ranked_candidates[4].explanation)

    # Save ranking results to evaluation
    df.to_csv("evaluation/hybrid_screening_results.csv", index=False)
    print(f"\nSaved leaderboard to evaluation/hybrid_screening_results.csv")

if __name__ == "__main__":
    main()
