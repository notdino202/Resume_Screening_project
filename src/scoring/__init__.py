"""Scoring package combining embeddings, semantic matching, hybrid scoring, and explanations."""

from .embeddings import EmbeddingManager
from .semantic_matcher import SemanticMatcher, SemanticMatchResult, RequirementMatchDetail
from .scorer import HybridScorer, CandidateScoreResult, SkillMatchDetail
from .explainer import ExplanationGenerator

__all__ = [
    "EmbeddingManager",
    "SemanticMatcher",
    "SemanticMatchResult",
    "RequirementMatchDetail",
    "HybridScorer",
    "CandidateScoreResult",
    "SkillMatchDetail",
    "ExplanationGenerator",
]
