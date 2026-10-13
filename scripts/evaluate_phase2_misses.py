"""Analyzes skill extraction and identifies potential misses across 10 sample resumes."""

from pathlib import Path
from src.ingestion.extractor import DocumentExtractor
from src.extraction.section_detector import SectionDetector
from src.extraction.skills_matcher import SkillsMatcher

SAMPLE_DIR = Path("data/sample_resumes")
RESUMES = [
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

    print("=" * 80)
    print("PHASE 2 EVALUATION: SKILL EXTRACTION & MISS ANALYSIS ON 10 SAMPLE RESUMES")
    print("=" * 80)

    for fname in RESUMES:
        fpath = SAMPLE_DIR / fname
        result = extractor.extract(fpath)
        sections = detector.detect_sections(result.clean_text)
        skills = matcher.extract_skills_from_sections(sections)

        print(f"\nDocument: {fname} ({result.file_type})")
        print(f"Detected Sections: {[s.value for s in sections.section_texts.keys()]}")
        print(f"Extracted Skills Count: {len(skills)}")
        skill_summary = [f"{name} (ev={data.evidence_factor}, sec={','.join(data.sections_found)})" for name, data in skills.items()]
        print("Extracted Skills:")
        print("  " + ", ".join(skill_summary[:12]))
        if len(skill_summary) > 12:
            print("  ...and " + ", ".join(skill_summary[12:]))

if __name__ == "__main__":
    main()
