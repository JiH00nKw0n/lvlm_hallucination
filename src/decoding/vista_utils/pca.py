"""
PCA implementation for VISTA.

Reference: VISTA/myutils.py:104-145
"""

import torch
import torch.nn as nn


def svd_flip(u: torch.Tensor, v: torch.Tensor):
    """
    Sign correction to ensure deterministic output from SVD.

    Reference: VISTA/myutils.py:104-111
    """
    max_abs_cols = torch.argmax(torch.abs(u), 0)
    i = torch.arange(u.shape[1]).to(u.device)
    signs = torch.sign(u[max_abs_cols, i])
    u *= signs
    v *= signs.view(-1, 1)
    return u, v


class PCA(nn.Module):
    """
    Simple PCA implementation using SVD.

    Reference: VISTA/myutils.py:114-145
    """

    def __init__(self, n_components: int):
        """
        Initialize PCA.

        Args:
            n_components: Number of principal components to keep
        """
        super().__init__()
        self.n_components = n_components

    @torch.no_grad()
    def fit(self, X: torch.Tensor) -> "PCA":
        """
        Fit PCA on data.

        Args:
            X: Data tensor of shape (n_samples, n_features)

        Returns:
            self
        """
        n, d = X.size()
        if self.n_components is not None:
            d = min(self.n_components, d)

        self.register_buffer("mean_", X.mean(0, keepdim=True))
        Z = X - self.mean_  # center

        U, S, Vh = torch.linalg.svd(Z, full_matrices=False)
        Vt = Vh
        U, Vt = svd_flip(U, Vt)

        self.register_buffer("components_", Vt[:d])
        return self

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Transform data."""
        return self.transform(X)

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        """
        Project data onto principal components.

        Args:
            X: Data tensor of shape (n_samples, n_features)

        Returns:
            Projected data of shape (n_samples, n_components)
        """
        assert hasattr(self, "components_"), "PCA must be fit before use."
        return torch.matmul(X - self.mean_, self.components_.t())

    def fit_transform(self, X: torch.Tensor) -> torch.Tensor:
        """Fit and transform in one step."""
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, Y: torch.Tensor) -> torch.Tensor:
        """
        Transform data back to original space.

        Args:
            Y: Projected data of shape (n_samples, n_components)

        Returns:
            Reconstructed data of shape (n_samples, n_features)
        """
        assert hasattr(self, "components_"), "PCA must be fit before use."
        return torch.matmul(Y, self.components_) + self.mean_
