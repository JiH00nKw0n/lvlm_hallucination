from __future__ import annotations

import numpy as np

from src.datasets.synthetic_multimodal_feature import (
    SyntheticMultimodalFeatureDatasetBuilder,
)
from synthetic_multimodal_experiment import _compute_recovery_metrics


def test_multimodal_builder_pair_structure_and_min_active():
    builder = SyntheticMultimodalFeatureDatasetBuilder(
        feature_dim=16,
        representation_dim=8,
        vl_split_ratio=(1, 2, 1),
        num_train=64,
        num_eval=16,
        num_test=16,
        sparsity_shared=0.999,
        sparsity_image=0.999,
        sparsity_text=0.999,
        min_active_shared=1,
        min_active_image=1,
        min_active_text=1,
        strategy="random",
        seed=123,
        return_ground_truth=True,
    )

    data = builder.build_numpy_dataset()["train"]
    n_image, n_shared, n_text = builder.feature_block_dims

    shared_mask = data["shared_ground_truth_feature"]
    image_mask = data["image_private_ground_truth_feature"]
    text_mask = data["text_private_ground_truth_feature"]
    image_lin = data["image_linear_coefficient"]
    text_lin = data["text_linear_coefficient"]

    assert shared_mask.shape == (64, n_shared)
    assert image_mask.shape == (64, n_image)
    assert text_mask.shape == (64, n_text)

    assert np.all(shared_mask.sum(axis=1) >= 1.0)
    assert np.all(image_mask.sum(axis=1) >= 1.0)
    assert np.all(text_mask.sum(axis=1) >= 1.0)

    # Shared coefficient block is identical for paired image/text samples.
    shared_start = n_image
    shared_end = n_image + n_shared
    np.testing.assert_allclose(
        image_lin[:, shared_start:shared_end],
        text_lin[:, shared_start:shared_end],
        rtol=0.0,
        atol=0.0,
    )

    # Cross-block orthogonality is guaranteed by default.
    w_img = builder.w_image
    w_shared = builder.w_shared
    w_txt = builder.w_text
    assert np.max(np.abs(w_img.T @ w_shared)) < 1e-5
    assert np.max(np.abs(w_img.T @ w_txt)) < 1e-5
    assert np.max(np.abs(w_shared.T @ w_txt)) < 1e-5


def test_recovery_metrics_perfect_alignment():
    rng = np.random.default_rng(0)
    d = 10
    n = 6
    gt = rng.standard_normal((d, n))
    gt /= np.linalg.norm(gt, axis=0, keepdims=True)

    learned = gt.T.copy()  # perfect one-to-one
    metrics = _compute_recovery_metrics(learned, gt, threshold=0.8)

    assert abs(metrics["gt_recovery"] - 1.0) < 1e-8
    assert abs(metrics["mip"] - 1.0) < 1e-8
