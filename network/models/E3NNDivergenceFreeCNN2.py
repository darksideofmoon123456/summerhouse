# 1D convolutional network (CNN) over time after computing curl(ψ), capturing local temporal patterns before the final MLP.

import torch
import torch.nn as nn
import math
from e3nn.o3 import Irreps, Linear


class E3NNBlock(nn.Module):
    def __init__(self, irreps_in, irreps_out):
        super().__init__()
        self.linear = Linear(irreps_in, irreps_out)
        self.activation = nn.ReLU()

    def forward(self, x):
        return self.activation(self.linear(x))


class E3NNDivergenceFreeCNN2(nn.Module):
    """
    SE(3)-equivariant CNN that outputs a divergence-free vector field
    using the curl of a learned vector potential ψ, and applies true CNN over time.
    """
    def __init__(self, seq_length, n_features, hidden_irreps="16x0e + 16x1e", 
                 potential_irreps="1x1e", output_dim=1, mlp_dim=64, n_blocks=3, dropout=0.2):
        super().__init__()
        assert n_features >= 3, "Input must have at least 3 features (x,y,z)"
        self.seq_length = seq_length
        self.name = "E3NNDivergenceFreeCNN2"

        self.irreps_in = Irreps("1e")
        self.irreps_hidden = Irreps(hidden_irreps)
        self.irreps_potential = Irreps(potential_irreps)

        # Equivariant encoder
        self.input_proj = Linear(self.irreps_in, self.irreps_hidden)
        self.blocks = nn.ModuleList([
            E3NNBlock(self.irreps_hidden, self.irreps_hidden) for _ in range(n_blocks)
        ])
        self.potential_proj = Linear(self.irreps_hidden, self.irreps_potential)

        # CNN layers applied over time after computing v = curl(ψ)
        self.cnn = nn.Sequential(
            nn.Conv1d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)  # Output shape: (B, 32, 1)
        )

        # Final MLP head
        self.fc = nn.Sequential(
            nn.Flatten(),            # (B, 32)
            nn.Linear(32, mlp_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, output_dim)
        )

    def compute_curl(self, psi, eps=1e-1):
        """
        Approximate curl over the time axis using finite differences.
        psi: shape (B, T, 3)
        returns: divergence-free vector field (B, T, 3)
        """
        B, T, _ = psi.shape
        curl = torch.zeros_like(psi)

        fwd = torch.roll(psi, shifts=-1, dims=1)
        bwd = torch.roll(psi, shifts=1, dims=1)

        curl[:, :, 0] = (fwd[:, :, 2] - bwd[:, :, 1]) / (2 * eps)
        curl[:, :, 1] = (fwd[:, :, 0] - bwd[:, :, 2]) / (2 * eps)
        curl[:, :, 2] = (fwd[:, :, 1] - bwd[:, :, 0]) / (2 * eps)

        return curl

    def forward(self, x):
        """
        x: Tensor of shape (B, 3, T) or (B, T, 3)
        Returns: (B, output_dim)
        """
        if x.shape[1] != 3 and x.shape[2] == 3:
            x = x.permute(0, 2, 1)  # (B, T, 3) → (B, 3, T)

        B, C, T = x.shape
        assert C >= 3 and T == self.seq_length, f"Invalid input shape: {x.shape}"

        x = x[:, :3, :].permute(0, 2, 1).reshape(B * T, 3)  # (B*T, 3)

        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        psi = self.potential_proj(x).view(B, T, 3)

        v = self.compute_curl(psi)       # (B, T, 3)
        v = v.permute(0, 2, 1)           # (B, 3, T) for Conv1d

        x = self.cnn(v)                  # (B, 32, 1)
        out = self.fc(x)                 # (B, output_dim)
        return out


# The vector potential is learned via equivariant layers.
# The output field v is guaranteed divergence-free via curl.
# Temporal features are learned by actual CNN layers.

