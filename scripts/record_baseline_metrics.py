"""Runs TF-IDF and Skill-Overlap baseline rankings on 10 resumes against JD 1 and records metrics."""

from pathlib import Path
import pandas as pd

from src.ingestion.extractor import DocumentExtractor
from src.extraction.section_detector import SectionDetector
from src.extraction.skills_matcher import SkillsMatcher
from src.jd.parser import JobDescriptionParser
from src.matching.baselines import TfIdfMatcher, SkillOverlapMatcher

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
    extractor = DocumentExtractor()
    detector = SectionDetector()
    matcher = SkillsMatcher()
    jd_parser = JobDescriptionParser(skills_matcher=matcher)

    # 1. Parse JD 1 (Senior ML Engineer)
    jd_path = Path("data/sample_jds/jd_01_senior_ml_engineer.txt")
    jd = jd_parser.parse(jd_path.read_text(encoding="utf-8"))

    print("=" * 80)
    print(f"JOB DESCRIPTION: {jd.title}")
    print(f"Must-Have Skills ({len(jd.must_have_skills)}): {jd.must_have_skills}")
    print(f"Nice-To-Have Skills ({len(jd.nice_to_have_skills)}): {jd.nice_to_have_skills}")
    print("=" * 80)

    # 2. Ingest and extract all 10 resumes
    parsed_resumes = []
    for fname in SAMPLE_RESUMES:
        fpath = Path("data/sample_resumes") / fname
        ext_res = extractor.extract(fpath)
        sections = detector.detect_sections(ext_res.clean_text)
        skills = matcher.extract_skills_from_sections(sections)
        parsed_resumes.append({
            "filename": fname,
            "clean_text": ext_res.clean_text,
            "skills": skills,
        })

    # 3. TF-IDF Ranking
    tfidf = TfIdfMatcher()
    tfidf_candidates = [(r["filename"], r["filename"], r["clean_text"]) for r in parsed_resumes]
    tfidf_rankings = tfidf.rank_candidates(jd.raw_text, tfidf_candidates)

    # 4. Skill Overlap Ranking
    overlap_matcher = SkillOverlapMatcher()
    skill_candidates = [(r["filename"], r["filename"], r["skills"]) for r in parsed_resumes]
    skill_rankings = overlap_matcher.rank_candidates(jd, skill_candidates)

    # 5. Merge and compare
    records = []
    tfidf_map = {r.file_name: (r.rank, r.score) for r in tfidf_rankings}
    skill_map = {r.file_name: (r.rank, r.score, r.metadata.get("matched_must", [])) for r in skill_rankings}

    for r in parsed_resumes:
        fn = r["filename"]
        t_rank, t_score = tfidf_map[fn]
        s_rank, s_score, matched = skill_map[fn]
        records.append({
            "Resume": fn,
            "TFIDF_Rank": t_rank,
            "TFIDF_Score": t_score,
            "Skill_Rank": s_rank,
            "Skill_Score": s_score,
            "Matched_Must_Skills": len(matched),
        })

    df = pd.DataFrame(records).sort_values(by="Skill_Rank")
    print("\n--- BASELINE RANKING COMPARISON TABLE ---")
    print(df.to_string(index=False))

    # Save to evaluation directory
    eval_dir = Path("evaluation")
    eval_dir.mkdir(exist_ok=True)
    df.to_csv(eval_dir / "baseline_results.csv", index=False)
    print(f"\nSaved metrics to {eval_dir / 'baseline_results.csv'}")

if __name__ == "__main__":
    main()
