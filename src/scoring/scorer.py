"""Hybrid scoring engine combining skill, semantic, experience, and education components."""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence
import numpy as np
import yaml

from src.jd.models import JobDescription
from src.extraction.education_parser import EducationParser, CandidateEducation
from src.extraction.experience_parser import ExperienceParser, CandidateExperience
from src.extraction.skills_matcher import SkillsMatcher, ExtractedSkill
from .embeddings import EmbeddingManager
from .semantic_matcher import SemanticMatcher, SemanticMatchResult


@dataclass
class SkillMatchDetail:
    """Detailed match evaluation for a single JD skill."""
    jd_skill: str
    match_type: str  # "exact", "semantic", "missing"
    matched_candidate_skill: Optional[str]
    similarity: float
    evidence_factor: float
    final_skill_score: float
    sections_found: list[str] = field(default_factory=list)


@dataclass
class CandidateScoreResult:
    """Comprehensive score and diagnostic record for a screened candidate."""
    candidate_id: str
    file_name: str
    final_score: float  # [0.0, 100.0]
    tier: str  # "Strong", "Moderate", "Weak"
    rank: int = 0

    # Component Scores [0.0, 1.0]
    skill_score: float = 0.0
    semantic_score: float = 0.0
    experience_score: float = 0.0
    education_score: float = 0.0

    # Breakdowns & Intermediate values
    must_have_coverage: float = 0.0
    nice_to_have_coverage: float = 0.0
    must_have_details: list[SkillMatchDetail] = field(default_factory=list)
    nice_to_have_details: list[SkillMatchDetail] = field(default_factory=list)

    # Experience details
    total_experience_years: float = 0.0
    relevant_experience_years: float = 0.0
    required_years: Optional[float] = None

    # Education details
    candidate_education_level: int = 0
    candidate_education_name: str = "None"
    required_education_level: int = 3
    education_gap: int = 0

    # Semantic details
    semantic_match_result: Optional[SemanticMatchResult] = None

    # Weights used
    weights_used: dict[str, float] = field(default_factory=dict)

    # Flags
    advisory_flags: list[str] = field(default_factory=list)

    # Explanation
    explanation: str = ""


class HybridScorer:
    """Deterministic hybrid scoring engine implementing the 4-component specification."""

    def __init__(
        self,
        config_path: Optional[str | Path] = None,
        embedder: Optional[EmbeddingManager] = None,
        semantic_matcher: Optional[SemanticMatcher] = None,
        edu_parser: Optional[EducationParser] = None,
        exp_parser: Optional[ExperienceParser] = None,
    ):
        self.config = self._load_config(config_path)
        self.embedder = embedder or EmbeddingManager()
        self.semantic_matcher = semantic_matcher or SemanticMatcher(embedding_manager=self.embedder)
        self.edu_parser = edu_parser or EducationParser()
        self.exp_parser = exp_parser or ExperienceParser()

    def _load_config(self, config_path: Optional[str | Path]) -> dict:
        """Loads default weights and thresholds from config.yaml."""
        if config_path is None:
            config_path = Path(__file__).resolve().parent.parent.parent / "config.yaml"
        path = Path(config_path)
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        # Default fallback config if file missing
        return {
            "weights": {"skill": 0.40, "semantic": 0.30, "experience": 0.20, "education": 0.10},
            "skill_scoring": {"must_have_weight": 0.75, "nice_to_have_weight": 0.25},
            "tiers": {"strong": 75.0, "moderate": 50.0},
        }

    def score_candidate(
        self,
        candidate_id: str,
        file_name: str,
        clean_text: str,
        extracted_skills: dict[str, ExtractedSkill],
        candidate_edu: CandidateEducation,
        candidate_exp: CandidateExperience,
        jd: JobDescription,
        extraction_flags: Optional[list[str]] = None,
    ) -> CandidateScoreResult:
        """Calculates deterministic scores for all 4 components and compiles flags and tier.

        SPEC Formula:
        Final Score = 100 x (0.40 x Skill + 0.30 x Semantic + 0.20 x Experience + 0.10 x Education)
        """
        flags: list[str] = list(extraction_flags or [])

        # 1. Weights Resolution
        weights = jd.weights_override or self.config.get("weights", {
            "skill": 0.40, "semantic": 0.30, "experience": 0.20, "education": 0.10
        })
        # Validate weights sum to 1.0
        total_w = sum(weights.values())
        if not (0.999 <= total_w <= 1.001):
            raise ValueError(f"Scoring weights must sum to 1.0, got {total_w:.4f}: {weights}")

        # 2. Skill Component Scoring
        skill_score, must_cov, nice_cov, must_details, nice_details = self._score_skills(
            extracted_skills, jd.must_have_skills, jd.nice_to_have_skills
        )

        # Check for missing must-have flag
        has_missing_must = any(d.final_skill_score == 0.0 for d in must_details)
        if has_missing_must and "missing must-have" not in flags:
            flags.append("missing must-have")

        # 3. Semantic Component Scoring
        sem_res = self.semantic_matcher.score_requirements(jd.requirements, clean_text)
        semantic_score = sem_res.score

        # 4. Experience Component Scoring
        exp_score, exp_flags = self.exp_parser.score_experience(
            candidate_years=candidate_exp.total_years,
            required_years=jd.required_years,
            relevant_years=candidate_exp.relevant_years,
        )
        for ef in exp_flags:
            if ef not in flags:
                flags.append(ef)

        # 5. Education Component Scoring
        edu_score, edu_gap, edu_flags = self.edu_parser.score_education(
            candidate_level=candidate_edu.highest_level,
            required_level=jd.required_education_level,
        )
        for edf in edu_flags:
            if edf not in flags:
                flags.append(edf)

        # 6. Keyword Stuffing Detection Heuristic
        # Flag if total extracted skills count > 45 or skills density relative to words is excessive
        words = clean_text.split()
        word_count = len(words)
        if word_count > 0 and (len(extracted_skills) / word_count) > 0.25:
            if "possible keyword stuffing" not in flags:
                flags.append("possible keyword stuffing")

        # 7. Final Score Computation
        # Final Score = 100 x (w_skill * Skill + w_sem * Semantic + w_exp * Experience + w_edu * Education)
        final_score_raw = 100.0 * (
            weights["skill"] * skill_score
            + weights["semantic"] * semantic_score
            + weights["experience"] * exp_score
            + weights["education"] * edu_score
        )
        final_score = round(float(np.clip(final_score_raw, 0.0, 100.0)), 2)

        # 8. Tier Assignment
        # Strong >= 75, Moderate 50-74.9, Weak < 50
        tier_cfg = self.config.get("tiers", {"strong": 75.0, "moderate": 50.0})
        if final_score >= tier_cfg.get("strong", 75.0):
            tier = "Strong"
        elif final_score >= tier_cfg.get("moderate", 50.0):
            tier = "Moderate"
        else:
            tier = "Weak"

        return CandidateScoreResult(
            candidate_id=candidate_id,
            file_name=file_name,
            final_score=final_score,
            tier=tier,
            skill_score=round(skill_score, 4),
            semantic_score=round(semantic_score, 4),
            experience_score=round(exp_score, 4),
            education_score=round(edu_score, 4),
            must_have_coverage=round(must_cov, 4),
            nice_to_have_coverage=round(nice_cov, 4),
            must_have_details=must_details,
            nice_to_have_details=nice_details,
            total_experience_years=candidate_exp.total_years,
            relevant_experience_years=candidate_exp.relevant_years,
            required_years=jd.required_years,
            candidate_education_level=candidate_edu.highest_level,
            candidate_education_name=candidate_edu.highest_degree_name,
            required_education_level=jd.required_education_level,
            education_gap=edu_gap,
            semantic_match_result=sem_res,
            weights_used=weights,
            advisory_flags=flags,
        )

    def _score_skills(
        self,
        candidate_skills: dict[str, ExtractedSkill],
        must_have: list[str],
        nice_to_have: list[str],
    ) -> tuple[float, float, float, list[SkillMatchDetail], list[SkillMatchDetail]]:
        """Evaluates skill match details with embedding fallback and evidence factors."""
        must_details = [self._evaluate_single_skill(s, candidate_skills) for s in must_have]
        nice_details = [self._evaluate_single_skill(s, candidate_skills) for s in nice_to_have]

        must_cov = (
            sum(d.final_skill_score for d in must_details) / len(must_details)
            if must_details
            else 1.0
        )
        nice_cov = (
            sum(d.final_skill_score for d in nice_details) / len(nice_details)
            if nice_details
            else 1.0
        )

        must_w = self.config.get("skill_scoring", {}).get("must_have_weight", 0.75)
        nice_w = self.config.get("skill_scoring", {}).get("nice_to_have_weight", 0.25)

        if must_details and nice_details:
            skill_score = (must_w * must_cov) + (nice_w * nice_cov)
        elif must_details:
            skill_score = must_cov
        elif nice_details:
            skill_score = nice_cov
        else:
            skill_score = 1.0

        return float(np.clip(skill_score, 0.0, 1.0)), must_cov, nice_cov, must_details, nice_details

    def _evaluate_single_skill(
        self, jd_skill: str, candidate_skills: dict[str, ExtractedSkill]
    ) -> SkillMatchDetail:
        """Evaluates a single required skill against candidate skills."""
        # 1. Exact or alias match
        if jd_skill in candidate_skills:
            cand = candidate_skills[jd_skill]
            ev = cand.evidence_factor
            score = 1.0 * ev
            return SkillMatchDetail(
                jd_skill=jd_skill,
                match_type="exact",
                matched_candidate_skill=jd_skill,
                similarity=1.0,
                evidence_factor=ev,
                final_skill_score=round(score, 4),
                sections_found=cand.sections_found,
            )

        # 2. Semantic skill match fallback (embedding similarity >= 0.75, capped at 0.8)
        if candidate_skills:
            jd_vec = self.embedder.get_embedding(jd_skill)
            cand_skills_list = list(candidate_skills.keys())
            cand_vecs = [self.embedder.get_embedding(s) for s in cand_skills_list]

            sims = [float(np.dot(jd_vec, cv)) for cv in cand_vecs]
            best_idx = int(np.argmax(sims))
            best_sim = sims[best_idx]
            best_match_name = cand_skills_list[best_idx]

            threshold = self.config.get("skill_scoring", {}).get("semantic_match_threshold", 0.75)
            cap = self.config.get("skill_scoring", {}).get("semantic_match_cap", 0.80)

            if best_sim >= threshold:
                matched_cand = candidate_skills[best_match_name]
                ev = matched_cand.evidence_factor
                effective_sim = min(best_sim, cap)
                score = effective_sim * ev
                return SkillMatchDetail(
                    jd_skill=jd_skill,
                    match_type="semantic",
                    matched_candidate_skill=best_match_name,
                    similarity=round(best_sim, 4),
                    evidence_factor=ev,
                    final_skill_score=round(score, 4),
                    sections_found=matched_cand.sections_found,
                )

        # 3. Missing
        return SkillMatchDetail(
            jd_skill=jd_skill,
            match_type="missing",
            matched_candidate_skill=None,
            similarity=0.0,
            evidence_factor=0.0,
            final_skill_score=0.0,
            sections_found=[],
        )

    def rank_candidates(
        self, candidates: Sequence[CandidateScoreResult]
    ) -> list[CandidateScoreResult]:
        """Ranks candidates according to final score with SPEC tie-breaking.

        SPEC Rule:
        Break ranking ties by must-have coverage, then relevant years.
        """
        sorted_candidates = sorted(
            candidates,
            key=lambda c: (
                c.final_score,
                c.must_have_coverage,
                c.relevant_experience_years,
            ),
            reverse=True,
        )

        for rank, cand in enumerate(sorted_candidates, start=1):
            cand.rank = rank

        return sorted_candidates
