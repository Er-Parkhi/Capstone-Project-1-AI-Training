import numpy as np
from sklearn.metrics import fbeta_score
from src.train_imbalance import choose_threshold, metrics_at_threshold


def test_threshold_is_valid_and_f2_is_reported():
    y = np.array([0, 0, 0, 1, 1, 1])
    p = np.array([0.05, 0.1, 0.2, 0.3, 0.7, 0.9])
    threshold, score = choose_threshold(y, p)
    assert 0.05 <= threshold <= 0.95
    assert score == fbeta_score(y, p >= threshold, beta=2, zero_division=0)


def test_metrics_confusion_counts_sum_to_sample_count():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.8, 0.7, 0.2])
    result = metrics_at_threshold(y, p, 0.5)
    assert result["tn"] + result["fp"] + result["fn"] + result["tp"] == len(y)
    assert result["tp"] == 1
    assert result["fn"] == 1
