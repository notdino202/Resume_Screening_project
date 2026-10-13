"""Evaluation module: NDCG@10, Precision@5, Skill F1, Experience MAE, and ablation studies."""

from __future__ import annotations
import math
from typing import Sequence
import numpy as np
import pandas as pd


def compute_dcg_at_k(relevances: Sequence[float], k: int = 10) -> float:
    """Computes Discounted Cumulative Gain at rank k."""
    dcg = 0.0
    for idx, rel in enumerate(relevances[:k]):
        dcg += (2.0 ** rel - 1.0) / math.log2(idx + 2.0)
    return dcg


def compute_ndcg_at_k(y_true: Sequence[float], y_score: Sequence[float], k: int = 10) -> float:
    """Computes Normalized Discounted Cumulative Gain (NDCG) at rank k."""
    if len(y_true) == 0:
        return 0.0

    # Sort true labels by predicted score descending
    sorted_pairs = sorted(zip(y_score, y_true), key=lambda x: x[0], reverse=True)
    sorted_relevances = [p[1] for p in sorted_pairs]

    actual_dcg = compute_dcg_at_k(sorted_relevances, k=k)

    # Ideal ranking (sorted by true label descending)
    ideal_relevances = sorted(y_true, reverse=True)
    ideal_dcg = compute_dcg_at_k(ideal_relevances, k=k)

    if ideal_dcg == 0.0:
        return 0.0
    return float(np.clip(actual_dcg / ideal_dcg, 0.0, 1.0))


def compute_precision_at_k(
    y_true: Sequence[float], y_score: Sequence[float], k: int = 5, relevance_threshold: float = 2.0
) -> float:
    """Computes Precision@k considering items with true label >= relevance_threshold as relevant."""
    if len(y_true) == 0 or k == 0:
        return 0.0

    sorted_pairs = sorted(zip(y_score, y_true), key=lambda x: x[0], reverse=True)
    top_k = sorted_pairs[:k]
    relevant_count = sum(1 for _, rel in top_k if rel >= relevance_threshold)
    return float(relevant_count / len(top_k))


def compute_skill_f1(predicted_skills: set[str], ground_truth_skills: set[str]) -> tuple[float, float, float]:
    """Computes precision, recall, and F1 between extracted and ground-truth skills."""
    if not ground_truth_skills and not predicted_skills:
        return 1.0, 1.0, 1.0
    if not ground_truth_skills or not predicted_skills:
        return 0.0, 0.0, 0.0

    tp = len(predicted_skills.intersection(ground_truth_skills))
    precision = tp / len(predicted_skills) if predicted_skills else 0.0
    recall = tp / len(ground_truth_skills) if ground_truth_skills else 0.0

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2.0 * (precision * recall) / (precision + recall)

    return round(precision, 4), round(recall, 4), round(f1, 4)


def compute_experience_error(
    predicted_years: Sequence[float], ground_truth_years: Sequence[float]
) -> dict[str, float]:
    """Computes MAE and RMSE between extracted and ground-truth experience years."""
    p_arr = np.array(predicted_years, dtype=float)
    g_arr = np.array(ground_truth_years, dtype=float)

    diff = p_arr - g_arr
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff ** 2)))

    return {"mae": round(mae, 3), "rmse": round(rmse, 3)}
