# curl-regularized MLP with E3NN input layers... first draft
# doesn’t apply any convolutional layers over the time sequence.

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


class E3NNDivergenceFreeCNN(nn.Module):
    """
    SE(3)-equivariant CNN that outputs a divergence-free vector field
    using the curl of a learned vector potential ψ.
    """
    def __init__(self, seq_length, n_features, hidden_irreps="16x0e + 16x1e", 
                 potential_irreps="1x1e", output_dim=1, mlp_dim=64, n_blocks=3, dropout=0.2):
        super().__init__()
        assert n_features >= 3, "Input must have at least 3 features (x,y,z)"
        self.seq_length = seq_length
        self.name = "E3NNDivergenceFreeCNN"

        self.irreps_in = Irreps("1e")  # Input: 3D vector
        self.irreps_hidden = Irreps(hidden_irreps)
        self.irreps_potential = Irreps(potential_irreps)  # ψ: vector potential
        self.irreps_out = Irreps("1x0e")

        # Input projection to hidden irreps
        self.input_proj = Linear(self.irreps_in, self.irreps_hidden)

        # Stack of equivariant blocks
        self.blocks = nn.ModuleList([
            E3NNBlock(self.irreps_hidden, self.irreps_hidden) for _ in range(n_blocks)
        ])

        # Project to vector potential
        self.potential_proj = Linear(self.irreps_hidden, self.irreps_potential)

        # MLP head for final output
        self.fc1 = nn.Linear(3 * seq_length, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def compute_curl(self, psi, eps=1e-1):
        """
        Approximate curl over the time axis using finite differences.
        psi: shape (B, T, 3)
        returns: divergence-free field of shape (B, T, 3)
        """
        B, T, _ = psi.shape
        curl = torch.zeros_like(psi)

        # forward and backward shift
        fwd = torch.roll(psi, shifts=-1, dims=1)
        bwd = torch.roll(psi, shifts=1, dims=1)

        curl[:, :, 0] = (fwd[:, :, 2] - bwd[:, :, 1]) / (2 * eps)
        curl[:, :, 1] = (fwd[:, :, 0] - bwd[:, :, 2]) / (2 * eps)
        curl[:, :, 2] = (fwd[:, :, 1] - bwd[:, :, 0]) / (2 * eps)

        return curl

    def forward(self, x):
        """
        x: Tensor of shape (B, 3, T) or (B, T, 3)
        Returns:
            Output of shape (B, output_dim)
        """
        # Automatically permute if input is (B, T, 3)
        if x.shape[1] != 3 and x.shape[2] == 3:
            x = x.permute(0, 2, 1)  # (B, T, 3) → (B, 3, T)

        B, C, T = x.shape
        assert C >= 3 and T == self.seq_length, f"Invalid input shape: {x.shape}"

        x = x[:, :3, :].permute(0, 2, 1).reshape(B * T, 3)  # (B*T, 3)

        # Pass through equivariant layers
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)

        # Project to vector potential ψ
        psi = self.potential_proj(x)      # (B*T, 3)
        psi = psi.view(B, T, 3)           # (B, T, 3)

        # Compute divergence-free output: v = ∇ × ψ
        v = self.compute_curl(psi)        # (B, T, 3)

        # Flatten and pass through MLP head
        v_flat = v.reshape(B, -1)
        x = self.relu(self.fc1(v_flat))
        x = self.dropout(x)
        out = self.fc2(x)
        return out
