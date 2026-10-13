"""Requirement-level semantic matching module using dense embeddings and linear rescaling."""

from __future__ import annotations
from dataclasses import dataclass, field
import re
from typing import Optional
import numpy as np

from .embeddings import EmbeddingManager


@dataclass
class RequirementMatchDetail:
    """Details of matching a single JD requirement against the best resume chunk."""
    jd_requirement: str
    best_chunk: str
    cosine_similarity: float
    rescaled_similarity: float


@dataclass
class SemanticMatchResult:
    """Overall semantic matching result for a candidate."""
    score: float  # Rescaled into [0, 1]
    raw_average_similarity: float
    requirement_details: list[RequirementMatchDetail] = field(default_factory=list)


class SemanticMatcher:
    """Performs requirement-level semantic matching between JD requirements and resume chunks."""

    def __init__(
        self,
        embedding_manager: Optional[EmbeddingManager] = None,
        sim_low: float = 0.20,
        sim_high: float = 0.70,
        chunk_size_words: int = 60,
        chunk_overlap_words: int = 15,
    ):
        self.embedder = embedding_manager or EmbeddingManager()
        self.sim_low = sim_low
        self.sim_high = sim_high
        self.chunk_size_words = chunk_size_words
        self.chunk_overlap_words = chunk_overlap_words

    def chunk_text(self, text: str) -> list[str]:
        """Splits resume text into overlapping word chunks and distinct paragraphs."""
        if not text or not text.strip():
            return []

        chunks: list[str] = []

        # 1. Paragraph / bullet chunks
        paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 30]
        chunks.extend(paragraphs)

        # 2. Sliding window of words for thoroughness
        words = text.split()
        if len(words) > self.chunk_size_words:
            step = max(1, self.chunk_size_words - self.chunk_overlap_words)
            for i in range(0, len(words), step):
                window = words[i : i + self.chunk_size_words]
                if len(window) >= 15:  # Minimum useful chunk length
                    chunks.append(" ".join(window))
        elif words and not chunks:
            chunks.append(" ".join(words))

        # Deduplicate while preserving order
        unique_chunks = list(dict.fromkeys(chunks))
        return unique_chunks

    def score_requirements(
        self, jd_requirements: list[str], resume_text: str
    ) -> SemanticMatchResult:
        """Evaluates each JD requirement against the best resume chunk, averages, and rescales.

        SPEC Rule:
        For each JD requirement, take the best-matching resume chunk by cosine similarity,
        average, then rescale with sim_low=0.20 and sim_high=0.70, clipped to [0, 1].

        Args:
            jd_requirements: List of requirement strings.
            resume_text: Full clean text of candidate resume.

        Returns:
            SemanticMatchResult with rescaled score and individual requirement details.
        """
        if not jd_requirements or not resume_text.strip():
            return SemanticMatchResult(score=0.0, raw_average_similarity=0.0)

        resume_chunks = self.chunk_text(resume_text)
        if not resume_chunks:
            return SemanticMatchResult(score=0.0, raw_average_similarity=0.0)

        # Compute embeddings
        req_embeddings = self.embedder.get_embeddings_batch(jd_requirements)
        chunk_embeddings = self.embedder.get_embeddings_batch(resume_chunks)

        chunk_matrix = np.array(chunk_embeddings)  # shape (num_chunks, 384)

        details: list[RequirementMatchDetail] = []
        best_similarities: list[float] = []

        for req_idx, req_str in enumerate(jd_requirements):
            req_vec = req_embeddings[req_idx]
            # Cosine similarity against all chunks
            sims = np.dot(chunk_matrix, req_vec)
            best_chunk_idx = int(np.argmax(sims))
            best_sim = float(sims[best_chunk_idx])
            best_similarities.append(best_sim)

            # Rescale individual sim
            rescaled_ind = (best_sim - self.sim_low) / (self.sim_high - self.sim_low)
            rescaled_ind = float(np.clip(rescaled_ind, 0.0, 1.0))

            details.append(
                RequirementMatchDetail(
                    jd_requirement=req_str,
                    best_chunk=resume_chunks[best_chunk_idx],
                    cosine_similarity=round(best_sim, 4),
                    rescaled_similarity=round(rescaled_ind, 4),
                )
            )

        avg_raw_sim = float(np.mean(best_similarities))

        # Overall rescaling per SPEC:
        # rescale with sim_low=0.20 and sim_high=0.70, clipped to [0, 1]
        rescaled_score = (avg_raw_sim - self.sim_low) / (self.sim_high - self.sim_low)
        final_score = float(np.clip(rescaled_score, 0.0, 1.0))

        return SemanticMatchResult(
            score=round(final_score, 4),
            raw_average_similarity=round(avg_raw_sim, 4),
            requirement_details=details,
        )
