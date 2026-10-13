"""Baseline ranking models: TF-IDF cosine similarity and skill-overlap-only matching."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.jd.models import JobDescription
from src.extraction.skills_matcher import ExtractedSkill


@dataclass
class BaselineRankingResult:
    """Individual candidate ranking result from baseline models."""
    candidate_id: str
    file_name: str
    score: float
    rank: int = 0
    metadata: dict = field(default_factory=dict)


class TfIdfMatcher:
    """Ranks candidates against a Job Description using TF-IDF representation and cosine similarity."""

    def __init__(self, stop_words: str = "english", ngram_range: tuple[int, int] = (1, 2)):
        self.vectorizer = TfidfVectorizer(stop_words=stop_words, ngram_range=ngram_range)

    def score_pair(self, jd_text: str, resume_text: str) -> float:
        """Computes cosine similarity in [0, 1] between JD and single resume."""
        if not jd_text.strip() or not resume_text.strip():
            return 0.0

        try:
            tfidf_matrix = self.vectorizer.fit_transform([jd_text, resume_text])
            sim = float(cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0])
            return round(float(np.clip(sim, 0.0, 1.0)), 4)
        except Exception:
            return 0.0

    def rank_candidates(
        self,
        jd_text: str,
        candidates: Sequence[tuple[str, str, str]],  # (candidate_id, file_name, clean_text)
    ) -> list[BaselineRankingResult]:
        """Ranks a list of candidate documents against the JD."""
        if not candidates or not jd_text.strip():
            return []

        corpus = [jd_text] + [c[2] for c in candidates]
        tfidf_matrix = self.vectorizer.fit_transform(corpus)
        jd_vector = tfidf_matrix[0:1]
        resume_vectors = tfidf_matrix[1:]

        sims = cosine_similarity(jd_vector, resume_vectors)[0]

        results = []
        for idx, (cid, fname, _) in enumerate(candidates):
            score = round(float(np.clip(sims[idx], 0.0, 1.0)), 4)
            results.append(BaselineRankingResult(candidate_id=cid, file_name=fname, score=score))

        # Sort descending by score
        results.sort(key=lambda r: r.score, reverse=True)
        for rank, res in enumerate(results, start=1):
            res.rank = rank

        return results


class SkillOverlapMatcher:
    """Ranks candidates solely based on raw skill overlap against the JD requirements."""

    def __init__(self, must_have_weight: float = 0.75, nice_to_have_weight: float = 0.25):
        self.must_have_weight = must_have_weight
        self.nice_to_have_weight = nice_to_have_weight

    def score_skills(
        self,
        candidate_skills: dict[str, ExtractedSkill] | set[str] | list[str],
        jd_must_have: list[str],
        jd_nice_to_have: list[str],
    ) -> tuple[float, set[str], set[str]]:
        """Computes skill coverage score in [0.0, 1.0].

        Returns:
            Tuple of (score, matched_must_have, matched_nice_to_have).
        """
        c_skills = set(candidate_skills.keys()) if isinstance(candidate_skills, dict) else set(candidate_skills)
        must_set = set(jd_must_have)
        nice_set = set(jd_nice_to_have)

        matched_must = c_skills.intersection(must_set)
        matched_nice = c_skills.intersection(nice_set)

        must_cov = len(matched_must) / len(must_set) if must_set else 1.0
        nice_cov = len(matched_nice) / len(nice_set) if nice_set else 1.0

        if must_set and nice_set:
            score = (self.must_have_weight * must_cov) + (self.nice_to_have_weight * nice_cov)
        elif must_set:
            score = must_cov
        elif nice_set:
            score = nice_cov
        else:
            score = 1.0

        return round(float(np.clip(score, 0.0, 1.0)), 4), matched_must, matched_nice

    def rank_candidates(
        self,
        jd: JobDescription,
        candidates: Sequence[tuple[str, str, dict[str, ExtractedSkill]]],  # (cid, fname, skills_dict)
    ) -> list[BaselineRankingResult]:
        """Ranks candidates based on skill coverage."""
        results = []
        for cid, fname, skills_dict in candidates:
            score, matched_must, matched_nice = self.score_skills(
                skills_dict, jd.must_have_skills, jd.nice_to_have_skills
            )
            results.append(
                BaselineRankingResult(
                    candidate_id=cid,
                    file_name=fname,
                    score=score,
                    metadata={
                        "matched_must": list(matched_must),
                        "matched_nice": list(matched_nice),
                    },
                )
            )

        results.sort(key=lambda r: r.score, reverse=True)
        for rank, res in enumerate(results, start=1):
            res.rank = rank

        return results
