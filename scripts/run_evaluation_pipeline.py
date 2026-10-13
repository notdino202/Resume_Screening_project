"""Complete evaluation pipeline: Skill F1, Experience MAE, NDCG@10, Precision@5, Ablation Study, and Weight Tuning."""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

from src.ingestion.extractor import DocumentExtractor
from src.extraction.section_detector import SectionDetector
from src.extraction.skills_matcher import SkillsMatcher
from src.extraction.education_parser import EducationParser
from src.extraction.experience_parser import ExperienceParser
from src.jd.parser import JobDescriptionParser
from src.scoring.embeddings import EmbeddingManager
from src.scoring.semantic_matcher import SemanticMatcher
from src.scoring.scorer import HybridScorer
from src.matching.baselines import TfIdfMatcher, SkillOverlapMatcher
from evaluation.evaluator import (
    compute_ndcg_at_k,
    compute_precision_at_k,
    compute_skill_f1,
    compute_experience_error,
)

EVAL_CSV = Path("evaluation/labeled_pairs.csv")


def run_evaluation(csv_path: Path = EVAL_CSV):
    if not csv_path.exists():
        raise FileNotFoundError(f"Labeled pairs file not found: {csv_path}")

    df_pairs = pd.read_csv(csv_path)
    print(f"Loaded {len(df_pairs)} labeled pairs across {df_pairs['jd_file'].nunique()} Job Descriptions.")

    # Shared pipelines
    extractor = DocumentExtractor()
    detector = SectionDetector()
    matcher = SkillsMatcher()
    edu_parser = EducationParser()
    exp_parser = ExperienceParser()
    embedder = EmbeddingManager()
    sem_matcher = SemanticMatcher(embedding_manager=embedder)
    jd_parser = JobDescriptionParser(skills_matcher=matcher, education_parser=edu_parser)
    scorer = HybridScorer(embedder=embedder, semantic_matcher=sem_matcher, edu_parser=edu_parser, exp_parser=exp_parser)
    tfidf = TfIdfMatcher()
    overlap_matcher = SkillOverlapMatcher()

    # Pre-parse and cache unique resumes
    unique_resumes = df_pairs["resume_file"].unique()
    parsed_resumes = {}
    print(f"Parsing {len(unique_resumes)} candidate resumes...")
    for rf in unique_resumes:
        rpath = Path("data/sample_resumes") / rf
        ext_res = extractor.extract(rpath)
        sections = detector.detect_sections(ext_res.clean_text)
        skills = matcher.extract_skills_from_sections(sections)
        edu = edu_parser.parse_education(sections.education_text or ext_res.clean_text)
        exp = exp_parser.parse_experience(sections.experience_text or ext_res.clean_text)
        parsed_resumes[rf] = {
            "ext": ext_res,
            "sections": sections,
            "skills": skills,
            "edu": edu,
            "exp": exp,
        }

    # Pre-parse unique JDs
    unique_jds = df_pairs["jd_file"].unique()
    parsed_jds = {}
    for jf in unique_jds:
        jpath = Path("data/sample_jds") / jf
        parsed_jds[jf] = jd_parser.parse(jpath.read_text(encoding="utf-8"))

    # 1. Compute Extraction Metrics: Skill F1 and Experience MAE
    skill_f1_scores = []
    pred_years = []
    true_years = []

    for _, row in df_pairs.drop_duplicates(subset=["resume_file"]).iterrows():
        rf = row["resume_file"]
        rdata = parsed_resumes[rf]
        pred_skills_set = set(rdata["skills"].keys())

        if pd.notna(row.get("ground_truth_skills")):
            gt_skills_set = set(s.strip() for s in str(row["ground_truth_skills"]).split(",") if s.strip())
            _, _, f1 = compute_skill_f1(pred_skills_set, gt_skills_set)
            skill_f1_scores.append(f1)

        if pd.notna(row.get("ground_truth_years")):
            pred_years.append(rdata["exp"].total_years)
            true_years.append(float(row["ground_truth_years"]))

    macro_f1 = float(np.mean(skill_f1_scores)) if skill_f1_scores else 0.0
    exp_error = compute_experience_error(pred_years, true_years) if pred_years else {"mae": 0.0, "rmse": 0.0}

    print("\n" + "=" * 80)
    print("EXTRACTION QUALITY METRICS:")
    print(f"  Macro Skill F1: {macro_f1:.4f} (across {len(skill_f1_scores)} resumes)")
    print(f"  Experience MAE: {exp_error['mae']:.2f} years | RMSE: {exp_error['rmse']:.2f} years")
    print("=" * 80)

    # 2. Run Ablation Study Across 5 Variants:
    # 1: TF-IDF only
    # 2: Skills only
    # 3: Semantic only
    # 4: Skills + Semantic
    # 5: Full Hybrid (Skills + Semantic + Experience + Education)
    variants = ["TF-IDF only", "Skills only", "Semantic only", "Skills + Semantic", "Full Hybrid"]
    results_by_variant = {v: {"ndcg": [], "p5": []} for v in variants}

    for jf in unique_jds:
        jd_obj = parsed_jds[jf]
        jd_df = df_pairs[df_pairs["jd_file"] == jf].copy()

        true_relevances = jd_df["relevance_label"].tolist()

        # Variant scores
        tfidf_scores = []
        skills_scores = []
        semantic_scores = []
        skills_sem_scores = []
        hybrid_scores = []

        for _, row in jd_df.iterrows():
            rf = row["resume_file"]
            rdata = parsed_resumes[rf]

            # 1. TF-IDF
            t_score = tfidf.score_pair(jd_obj.raw_text, rdata["ext"].clean_text)
            tfidf_scores.append(t_score)

            # 2. Skills
            s_score, _, _ = overlap_matcher.score_skills(
                rdata["skills"], jd_obj.must_have_skills, jd_obj.nice_to_have_skills
            )
            skills_scores.append(s_score)

            # 3. Semantic
            sem_res = sem_matcher.score_requirements(jd_obj.requirements, rdata["ext"].clean_text)
            sem_score = sem_res.score
            semantic_scores.append(sem_score)

            # 4. Skills + Semantic
            skills_sem = (0.55 * s_score) + (0.45 * sem_score)
            skills_sem_scores.append(skills_sem)

            # 5. Full Hybrid
            cand_res = scorer.score_candidate(
                candidate_id=rf,
                file_name=rf,
                clean_text=rdata["ext"].clean_text,
                extracted_skills=rdata["skills"],
                candidate_edu=rdata["edu"],
                candidate_exp=rdata["exp"],
                jd=jd_obj,
            )
            hybrid_scores.append(cand_res.final_score / 100.0)

        # Compute ranking metrics for each variant
        score_dict = {
            "TF-IDF only": tfidf_scores,
            "Skills only": skills_scores,
            "Semantic only": semantic_scores,
            "Skills + Semantic": skills_sem_scores,
            "Full Hybrid": hybrid_scores,
        }

        for vname, scores in score_dict.items():
            ndcg = compute_ndcg_at_k(true_relevances, scores, k=10)
            p5 = compute_precision_at_k(true_relevances, scores, k=5, relevance_threshold=2.0)
            results_by_variant[vname]["ndcg"].append(ndcg)
            results_by_variant[vname]["p5"].append(p5)

    # Ablation summary table
    ablation_summary = []
    for vname in variants:
        mean_ndcg = float(np.mean(results_by_variant[vname]["ndcg"]))
        mean_p5 = float(np.mean(results_by_variant[vname]["p5"]))
        ablation_summary.append({
            "Variant / Model": vname,
            "Mean NDCG@10": round(mean_ndcg, 4),
            "Mean Precision@5": round(mean_p5, 4),
        })

    df_ablation = pd.DataFrame(ablation_summary)
    print("\n" + "=" * 80)
    print("ABLATION STUDY RESULTS (Average across all JDs):")
    print("=" * 80)
    print(df_ablation.to_string(index=False))

    # 3. Weight Tuning on Split (Train: JD1-3, Held-Out: JD4-5)
    train_jds = unique_jds[:3]
    test_jds = unique_jds[3:]

    print(f"\nTuning weights on Train Split ({list(train_jds)}) and testing on Held-Out Split ({list(test_jds)})...")

    # Grid search candidate weight combinations summing to 1.0
    weight_candidates = [
        {"skill": 0.40, "semantic": 0.30, "experience": 0.20, "education": 0.10},  # Default
        {"skill": 0.45, "semantic": 0.25, "experience": 0.20, "education": 0.10},
        {"skill": 0.35, "semantic": 0.35, "experience": 0.20, "education": 0.10},
        {"skill": 0.40, "semantic": 0.35, "experience": 0.15, "education": 0.10},
        {"skill": 0.50, "semantic": 0.20, "experience": 0.20, "education": 0.10},
    ]

    best_weights = weight_candidates[0]
    best_train_ndcg = -1.0

    for w_cand in weight_candidates:
        train_ndcgs = []
        for jf in train_jds:
            jd_obj = parsed_jds[jf]
            jd_obj.weights_override = w_cand
            jd_df = df_pairs[df_pairs["jd_file"] == jf]
            true_rels = jd_df["relevance_label"].tolist()

            cand_scores = []
            for _, row in jd_df.iterrows():
                rdata = parsed_resumes[row["resume_file"]]
                cr = scorer.score_candidate(row["resume_file"], row["resume_file"], rdata["ext"].clean_text, rdata["skills"], rdata["edu"], rdata["exp"], jd_obj)
                cand_scores.append(cr.final_score)
            train_ndcgs.append(compute_ndcg_at_k(true_rels, cand_scores, k=10))

        mean_tr_ndcg = float(np.mean(train_ndcgs))
        if mean_tr_ndcg > best_train_ndcg:
            best_train_ndcg = mean_tr_ndcg
            best_weights = w_cand

    # Evaluate best weights on held-out test split
    test_ndcgs = []
    test_p5s = []
    for jf in test_jds:
        jd_obj = parsed_jds[jf]
        jd_obj.weights_override = best_weights
        jd_df = df_pairs[df_pairs["jd_file"] == jf]
        true_rels = jd_df["relevance_label"].tolist()

        cand_scores = []
        for _, row in jd_df.iterrows():
            rdata = parsed_resumes[row["resume_file"]]
            cr = scorer.score_candidate(row["resume_file"], row["resume_file"], rdata["ext"].clean_text, rdata["skills"], rdata["edu"], rdata["exp"], jd_obj)
            cand_scores.append(cr.final_score)
        test_ndcgs.append(compute_ndcg_at_k(true_rels, cand_scores, k=10))
        test_p5s.append(compute_precision_at_k(true_rels, cand_scores, k=5, relevance_threshold=2.0))

    held_out_ndcg = float(np.mean(test_ndcgs))
    held_out_p5 = float(np.mean(test_p5s))

    print(f"Optimal Weights: {best_weights}")
    print(f"Train Split Mean NDCG@10: {best_train_ndcg:.4f}")
    print(f"Held-Out Split Mean NDCG@10: {held_out_ndcg:.4f}")
    print(f"Held-Out Split Mean Precision@5: {held_out_p5:.4f}")

    # Save results
    df_ablation.to_csv("evaluation/ablation_study_results.csv", index=False)
    return {
        "macro_f1": macro_f1,
        "exp_error": exp_error,
        "ablation_df": df_ablation,
        "best_weights": best_weights,
        "train_ndcg": best_train_ndcg,
        "held_out_ndcg": held_out_ndcg,
        "held_out_p5": held_out_p5,
    }


if __name__ == "__main__":
    run_evaluation()
