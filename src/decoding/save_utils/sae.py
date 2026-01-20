import torch
import torch.nn as nn


class SAE(nn.Module):
    """Sparse autoencoder used by SAVE (mirrors SAVE/SAE definition)."""

    def __init__(self, d_model: int = 4096, expansion_factor: int = 8):
        super().__init__()
        hidden = int(expansion_factor * d_model)
        self.fc1 = nn.Linear(d_model, hidden, bias=True)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden, d_model, bias=True)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
        w_dec = torch.randn(self.fc2.weight.size())
        w_dec = w_dec / w_dec.norm(dim=1, keepdim=True) * 0.1
        self.fc2.weight.data = w_dec
        self.fc1.weight.data = w_dec.t()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.relu1(self.fc1(x))
        x_hat = self.fc2(features)
        return x_hat, features
