"""
Synthetic multimodal feature dataset builder.

This builder generates paired image/text representations with shared and
modality-private latent factors:

    x_img = W_img @ z_img + W_shared @ z_shared
    x_txt = W_txt @ z_txt + W_shared @ z_shared

By construction, each pair shares the same shared latent coefficients.
"""

from __future__ import annotations

import logging
from typing import Literal, Optional

import numpy as np
from datasets import Dataset, DatasetDict
from pydantic import PrivateAttr, field_validator

from src.common.registry import registry
from src.datasets.base import BaseBuilder
from src.datasets.synthetic_feature import (
    SyntheticFeatureDatasetBuilder,
    _normalize_columns,
)

logger = logging.getLogger(__name__)

__all__ = [
    "SyntheticMultimodalFeatureDatasetBuilder",
]


def _split_dims(total_dim: int, ratio: tuple[int, int, int]) -> tuple[int, int, int]:
    """Split total_dim into (image_private, shared, text_private)."""
    denom = ratio[0] + ratio[1] + ratio[2]
    image_dim = total_dim * ratio[0] // denom
    shared_dim = total_dim * ratio[1] // denom
    text_dim = total_dim - image_dim - shared_dim
    return image_dim, shared_dim, text_dim


def _safe_float_array(arr: np.ndarray) -> np.ndarray:
    return arr.astype(np.float32, copy=False)


@registry.register_builder("SyntheticMultimodalFeatureDatasetBuilder")
class SyntheticMultimodalFeatureDatasetBuilder(BaseBuilder):
    """
    Synthetic paired multimodal dataset with shared/private latent factors.

    Notes:
        - feature_dim is the total GT feature width and is split by vl_split_ratio.
        - image/text pairs share the same shared latent coefficients.
        - sparsity and min_active can be controlled independently per block.
        - when enforce_block_orthogonality=True (default), image/shared/text GT
          dictionaries lie in mutually orthogonal representation subspaces.
    """

    # Dimensions
    feature_dim: int = 128
    representation_dim: int = 128
    vl_split_ratio: tuple[int, int, int] = (1, 2, 1)

    # Samples per split
    num_train: int = 50_000
    num_eval: int = 10_000
    num_test: int = 10_000

    # Block sparsity controls
    sparsity_shared: float = 0.999
    sparsity_image: float = 0.999
    sparsity_text: float = 0.999
    min_active_shared: int = 1
    min_active_image: int = 1
    min_active_text: int = 1

    # Coefficient distribution: cmin + Exp(beta)
    cmin: float = 0.0
    beta: float = 1.0

    # Dictionary generation controls
    max_interference: float = 0.3
    strategy: Literal["gradient", "sdp", "random"] = "gradient"
    enforce_block_orthogonality: bool = True
    eps_margin: float = 1e-3
    lambda_neg: float = 1e-3
    lr: float = 0.05
    max_iters: int = 2000
    tol: float = 1e-4
    sdp_restarts: int = 10
    sdp_refine_steps: int = 2000

    # Importance -- probability scaling
    importance_probability_decay: float = 1.0  # decay^i per feature; 1.0 = uniform
    importance_target: Literal["shared", "image", "text"] = "shared"  # which block gets decay

    # General
    seed: int = 0
    return_ground_truth: bool = False
    verbose: bool = False

    # Private state
    _w_image: Optional[np.ndarray] = PrivateAttr(default=None)
    _w_shared: Optional[np.ndarray] = PrivateAttr(default=None)
    _w_text: Optional[np.ndarray] = PrivateAttr(default=None)

    # ------------------------------------------------------------------ #
    # Validators                                                          #
    # ------------------------------------------------------------------ #

    @field_validator("feature_dim", "representation_dim", "num_train", "num_eval", "num_test")
    @classmethod
    def _validate_positive_int(cls, v: int) -> int:
        if v <= 0:
            raise ValueError(f"value must be positive, got {v}")
        return v

    @field_validator("sparsity_shared", "sparsity_image", "sparsity_text")
    @classmethod
    def _validate_sparsity(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"sparsity must be in [0, 1], got {v}")
        return v

    @field_validator("min_active_shared", "min_active_image", "min_active_text")
    @classmethod
    def _validate_min_active(cls, v: int) -> int:
        if v < 0:
            raise ValueError(f"min_active must be >= 0, got {v}")
        return v

    @field_validator("beta")
    @classmethod
    def _validate_beta(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"beta must be > 0, got {v}")
        return v

    @field_validator("importance_probability_decay")
    @classmethod
    def _validate_importance_decay(cls, v: float) -> float:
        if not 0.0 < v <= 1.0:
            raise ValueError(f"importance_probability_decay must be in (0, 1], got {v}")
        return v

    @field_validator("vl_split_ratio")
    @classmethod
    def _validate_vl_ratio(cls, v: tuple[int, int, int]) -> tuple[int, int, int]:
        if len(v) != 3:
            raise ValueError(f"vl_split_ratio must have 3 values, got {v}")
        if any(x <= 0 for x in v):
            raise ValueError(f"vl_split_ratio entries must be positive, got {v}")
        return v

    # ------------------------------------------------------------------ #
    # Properties                                                          #
    # ------------------------------------------------------------------ #

    @property
    def feature_block_dims(self) -> tuple[int, int, int]:
        """(image_private_dim, shared_dim, text_private_dim)."""
        return _split_dims(self.feature_dim, self.vl_split_ratio)

    @property
    def representation_block_dims(self) -> tuple[int, int, int]:
        """
        (image_private_rep_dim, shared_rep_dim, text_private_rep_dim).

        Used only when enforce_block_orthogonality=True.
        """
        return _split_dims(self.representation_dim, self.vl_split_ratio)

    @property
    def w_image(self) -> np.ndarray:
        self._ensure_dictionaries()
        assert self._w_image is not None
        return self._w_image

    @property
    def w_shared(self) -> np.ndarray:
        self._ensure_dictionaries()
        assert self._w_shared is not None
        return self._w_shared

    @property
    def w_text(self) -> np.ndarray:
        self._ensure_dictionaries()
        assert self._w_text is not None
        return self._w_text

    @property
    def w_full(self) -> np.ndarray:
        """Ground-truth matrix in [image_private | shared | text_private] order."""
        return np.concatenate([self.w_image, self.w_shared, self.w_text], axis=1)

    # ------------------------------------------------------------------ #
    # Public methods                                                      #
    # ------------------------------------------------------------------ #

    def build_numpy_dataset(self) -> dict[str, dict[str, np.ndarray]]:
        """Build numpy splits for fast experiment loops."""
        self._ensure_dictionaries()
        seeds = np.random.SeedSequence(self.seed).spawn(3)
        return {
            "train": self._build_numpy_split(self.num_train, seeds[0]),
            "eval": self._build_numpy_split(self.num_eval, seeds[1]),
            "test": self._build_numpy_split(self.num_test, seeds[2]),
        }

    def build_dataset(self) -> DatasetDict:
        """Build Hugging Face DatasetDict splits."""
        splits = self.build_numpy_dataset()
        return DatasetDict({
            "train": self._to_hf_dataset(splits["train"]),
            "eval": self._to_hf_dataset(splits["eval"]),
            "test": self._to_hf_dataset(splits["test"]),
        })

    # ------------------------------------------------------------------ #
    # Internals                                                           #
    # ------------------------------------------------------------------ #

    def _to_hf_dataset(self, split: dict[str, np.ndarray]) -> Dataset:
        data: dict[str, list] = {
            "idx": split["idx"].tolist(),
            "image_representation": split["image_representation"].tolist(),
            "text_representation": split["text_representation"].tolist(),
        }
        for key in (
            "shared_ground_truth_feature",
            "image_private_ground_truth_feature",
            "text_private_ground_truth_feature",
            "image_linear_coefficient",
            "text_linear_coefficient",
        ):
            if key in split:
                data[key] = split[key].tolist()
        return Dataset.from_dict(data)

    def _ensure_dictionaries(self) -> None:
        if self._w_image is not None and self._w_shared is not None and self._w_text is not None:
            return

        n_image, n_shared, n_text = self.feature_block_dims
        if min(n_image, n_shared, n_text) <= 0:
            raise ValueError(
                "feature_dim is too small for vl_split_ratio. "
                f"Got feature_dim={self.feature_dim}, dims={self.feature_block_dims}."
            )

        if self.enforce_block_orthogonality:
            # Build three orthogonal subspaces in representation space and generate
            # each block dictionary inside its own subspace. This guarantees:
            # shared ⟂ image-private, shared ⟂ text-private, image-private ⟂ text-private.
            r_image, r_shared, r_text = self.representation_block_dims
            if min(r_image, r_shared, r_text) <= 0:
                raise ValueError(
                    "representation_dim is too small for vl_split_ratio with orthogonal blocks. "
                    f"Got representation_dim={self.representation_dim}, dims={self.representation_block_dims}."
                )

            seeds = np.random.SeedSequence(self.seed).spawn(4)
            q = self._sample_orthonormal_basis(
                dim=self.representation_dim,
                seed=int(seeds[0].generate_state(1)[0]),
            )
            b_image = q[:, :r_image]
            b_shared = q[:, r_image:r_image + r_shared]
            b_text = q[:, r_image + r_shared:r_image + r_shared + r_text]

            w_image_local = self._generate_dictionary(
                num_features=n_image,
                seed=int(seeds[1].generate_state(1)[0]),
                representation_dim=r_image,
            )
            w_shared_local = self._generate_dictionary(
                num_features=n_shared,
                seed=int(seeds[2].generate_state(1)[0]),
                representation_dim=r_shared,
            )
            w_text_local = self._generate_dictionary(
                num_features=n_text,
                seed=int(seeds[3].generate_state(1)[0]),
                representation_dim=r_text,
            )

            self._w_image = _safe_float_array(b_image @ w_image_local)
            self._w_shared = _safe_float_array(b_shared @ w_shared_local)
            self._w_text = _safe_float_array(b_text @ w_text_local)
        else:
            seeds = np.random.SeedSequence(self.seed).spawn(3)
            self._w_image = self._generate_dictionary(
                num_features=n_image,
                seed=int(seeds[0].generate_state(1)[0]),
                representation_dim=self.representation_dim,
            )
            self._w_shared = self._generate_dictionary(
                num_features=n_shared,
                seed=int(seeds[1].generate_state(1)[0]),
                representation_dim=self.representation_dim,
            )
            self._w_text = self._generate_dictionary(
                num_features=n_text,
                seed=int(seeds[2].generate_state(1)[0]),
                representation_dim=self.representation_dim,
            )

        if self.verbose:
            logger.info(
                "Generated multimodal dictionaries: image=%d shared=%d text=%d rep=%d orthogonal_blocks=%s",
                n_image,
                n_shared,
                n_text,
                self.representation_dim,
                self.enforce_block_orthogonality,
            )

    def _sample_orthonormal_basis(self, dim: int, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        mat = rng.standard_normal((dim, dim))
        q, _ = np.linalg.qr(mat)
        return _safe_float_array(q)

    def _generate_dictionary(self, num_features: int, seed: int, representation_dim: int) -> np.ndarray:
        if self.strategy == "random":
            rng = np.random.default_rng(seed)
            w = rng.standard_normal((representation_dim, num_features))
            return _safe_float_array(_normalize_columns(w))

        builder = SyntheticFeatureDatasetBuilder(
            feature_latent_dim=num_features,
            representation_dim=representation_dim,
            num_train=1,
            num_val=1,
            num_test=1,
            sparsity=0.99,
            min_active=0,
            max_interference=self.max_interference,
            strategy="sdp" if self.strategy == "sdp" else "gradient",
            eps_margin=self.eps_margin,
            lambda_neg=self.lambda_neg,
            lr=self.lr,
            max_iters=self.max_iters,
            tol=self.tol,
            sdp_restarts=self.sdp_restarts,
            sdp_refine_steps=self.sdp_refine_steps,
            seed=seed,
            return_ground_truth=False,
            verbose=False,
        )
        w = builder._get_or_generate_wp().copy()
        return _safe_float_array(w)

    def _sample_block(
        self,
        num_samples: int,
        num_features: int,
        sparsity: float,
        min_active: int,
        rng: np.random.Generator,
        decay: float = 1.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sample sparse coefficients and masks for one latent block.

        Args:
            decay: importance probability decay for this block. 1.0 = uniform.
        """
        base_prob = 1.0 - sparsity
        if decay < 1.0:
            raw = decay ** np.arange(num_features)
            importance = raw / raw.mean()
            probs = np.clip(importance * base_prob, 0.0, 1.0).astype(np.float64)
        else:
            probs = np.full(num_features, base_prob, dtype=np.float64)
        mask = (rng.random((num_samples, num_features)) < probs).astype(np.float32)

        if min_active > 0:
            if min_active > num_features:
                raise ValueError(
                    f"min_active={min_active} cannot exceed num_features={num_features}"
                )
            row_counts = mask.sum(axis=1)
            deficient = np.where(row_counts < min_active)[0]
            for idx in deficient:
                active = int(row_counts[idx])
                need = min_active - active
                inactive = np.where(mask[idx] == 0.0)[0]
                chosen = rng.choice(inactive, size=need, replace=False)
                mask[idx, chosen] = 1.0

        coeffs = self.cmin + rng.exponential(self.beta, size=(num_samples, num_features))
        coeffs = _safe_float_array(coeffs)
        values = mask * coeffs
        return values, mask

    def _build_numpy_split(
        self, num_samples: int, seed_seq: np.random.SeedSequence
    ) -> dict[str, np.ndarray]:
        rng = np.random.default_rng(seed_seq)
        n_image, n_shared, n_text = self.feature_block_dims

        # Route importance decay to the target block only
        d = self.importance_probability_decay
        decay_shared = d if self.importance_target == "shared" else 1.0
        decay_image = d if self.importance_target == "image" else 1.0
        decay_text = d if self.importance_target == "text" else 1.0

        shared_values, shared_mask = self._sample_block(
            num_samples=num_samples,
            num_features=n_shared,
            sparsity=self.sparsity_shared,
            min_active=self.min_active_shared,
            rng=rng,
            decay=decay_shared,
        )
        image_values, image_mask = self._sample_block(
            num_samples=num_samples,
            num_features=n_image,
            sparsity=self.sparsity_image,
            min_active=self.min_active_image,
            rng=rng,
            decay=decay_image,
        )
        text_values, text_mask = self._sample_block(
            num_samples=num_samples,
            num_features=n_text,
            sparsity=self.sparsity_text,
            min_active=self.min_active_text,
            rng=rng,
            decay=decay_text,
        )

        image_rep = image_values @ self.w_image.T + shared_values @ self.w_shared.T
        text_rep = text_values @ self.w_text.T + shared_values @ self.w_shared.T

        split: dict[str, np.ndarray] = {
            "idx": np.arange(num_samples, dtype=np.int64),
            "image_representation": _safe_float_array(image_rep),
            "text_representation": _safe_float_array(text_rep),
        }

        if self.return_ground_truth:
            split["shared_ground_truth_feature"] = shared_mask
            split["image_private_ground_truth_feature"] = image_mask
            split["text_private_ground_truth_feature"] = text_mask
            split["image_linear_coefficient"] = _safe_float_array(
                np.concatenate([image_values, shared_values, np.zeros_like(text_values)], axis=1)
            )
            split["text_linear_coefficient"] = _safe_float_array(
                np.concatenate([np.zeros_like(image_values), shared_values, text_values], axis=1)
            )

        return split
