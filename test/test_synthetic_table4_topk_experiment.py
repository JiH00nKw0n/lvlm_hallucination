from __future__ import annotations

import numpy as np

from synthetic_table4_topk_experiment import (
    _compute_recovery_metrics,
    summarize_trends,
)


def test_recovery_metrics_perfect_alignment():
    rng = np.random.default_rng(0)
    d = 12
    n = 7
    gt = rng.standard_normal((d, n))
    gt /= np.linalg.norm(gt, axis=0, keepdims=True)

    learned = gt.T.copy()
    metrics = _compute_recovery_metrics(learned, gt, threshold=0.9)

    assert abs(metrics["gt_recovery"] - 1.0) < 1e-8
    assert abs(metrics["mip"] - 1.0) < 1e-8


def test_summarize_trends_flags_upward_mgt():
    rows = [
        {"feature_dim": 800, "k": 64, "mgt_full_mean": 0.10, "mip_full_mean": 0.80},
        {"feature_dim": 1000, "k": 64, "mgt_full_mean": 0.15, "mip_full_mean": 0.78},
        {"feature_dim": 1200, "k": 64, "mgt_full_mean": 0.14, "mip_full_mean": 0.77},
    ]

    summary = summarize_trends(rows)

    assert summary[64]["mgt_non_increasing"] is False
    assert summary[64]["mip_non_increasing"] is True
