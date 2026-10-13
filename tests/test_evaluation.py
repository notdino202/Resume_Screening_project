"""Unit tests for evaluation metrics: NDCG@10, Precision@5, Skill F1, Experience Error."""

from evaluation.evaluator import (
    compute_ndcg_at_k,
    compute_precision_at_k,
    compute_skill_f1,
    compute_experience_error,
)


def test_ndcg_at_k():
    # Perfect ranking: highest predicted score corresponds to highest true relevance
    y_true = [3.0, 2.0, 1.0, 0.0]
    y_score = [0.95, 0.70, 0.40, 0.10]
    ndcg_perfect = compute_ndcg_at_k(y_true, y_score, k=4)
    assert ndcg_perfect == 1.0

    # Inverted ranking
    y_score_inv = [0.10, 0.40, 0.70, 0.95]
    ndcg_inv = compute_ndcg_at_k(y_true, y_score_inv, k=4)
    assert ndcg_inv < ndcg_perfect
    assert 0.0 <= ndcg_inv <= 1.0


def test_precision_at_k():
    # Top 5 items: 4 items have rel >= 2 -> Precision@5 = 4/5 = 0.80
    y_true = [3.0, 3.0, 2.0, 2.0, 1.0, 0.0]
    y_score = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
    p5 = compute_precision_at_k(y_true, y_score, k=5, relevance_threshold=2.0)
    assert p5 == 0.80


def test_skill_f1():
    pred = {"Python", "PyTorch", "Docker"}
    truth = {"Python", "PyTorch", "FastAPI", "Kubernetes"}

    p, r, f1 = compute_skill_f1(pred, truth)
    # tp = 2 (Python, PyTorch)
    # precision = 2 / 3 = 0.6667
    # recall = 2 / 4 = 0.5000
    # f1 = 2 * (2/3 * 0.5) / (2/3 + 0.5) = (2/3) / (7/6) = 4/7 = 0.5714
    assert 0.66 <= p <= 0.67
    assert r == 0.50
    assert 0.57 <= f1 <= 0.58


def test_experience_error():
    pred = [5.0, 3.0, 2.0]
    truth = [5.0, 4.0, 1.0]
    err = compute_experience_error(pred, truth)
    # diffs = [0.0, -1.0, 1.0] -> abs diffs = [0, 1, 1] -> MAE = 2/3 = 0.667
    assert 0.66 <= err["mae"] <= 0.67
    assert err["rmse"] > 0
