import torch
import torch.nn as nn
from e3nn.o3 import Irreps, Linear


class E3NNBlock(nn.Module):
    def __init__(self, irreps_in, irreps_out):
        super().__init__()
        self.linear = Linear(irreps_in, irreps_out)
        self.activation = nn.ReLU()

    def forward(self, x):
        return self.activation(self.linear(x))


class E3NNDivergenceFreeCNN3(nn.Module):
    """
    Lightweight SE(3)-equivariant model that outputs a divergence-free vector field
    using the curl of a learned vector potential ψ, followed by a smaller MLP head.
    """
    def __init__(self, seq_length, n_features, hidden_irreps="8x0e + 8x1e", 
                 potential_irreps="1x1e", output_dim=1, mlp_dim=32, n_blocks=1, dropout=0.3):
        super().__init__()
        assert n_features >= 3, "Input must have at least 3 features (x,y,z)"
        self.seq_length = seq_length
        self.name = "E3NNDivergenceFreeCNN3"

        self.irreps_in = Irreps("1e")
        self.irreps_hidden = Irreps(hidden_irreps)
        self.irreps_potential = Irreps(potential_irreps)

        self.input_proj = Linear(self.irreps_in, self.irreps_hidden)
        self.blocks = nn.ModuleList([
            E3NNBlock(self.irreps_hidden, self.irreps_hidden) for _ in range(n_blocks)
        ])
        self.potential_proj = Linear(self.irreps_hidden, self.irreps_potential)

        # Small MLP head instead of CNN
        self.fc = nn.Sequential(
            nn.Flatten(),            # (B, T * 3)
            nn.Linear(seq_length * 3, mlp_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, output_dim)
        )

    def compute_curl(self, psi, eps=1e-1):
        B, T, _ = psi.shape
        curl = torch.zeros_like(psi)
        fwd = torch.roll(psi, shifts=-1, dims=1)
        bwd = torch.roll(psi, shifts=1, dims=1)

        curl[:, :, 0] = (fwd[:, :, 2] - bwd[:, :, 1]) / (2 * eps)
        curl[:, :, 1] = (fwd[:, :, 0] - bwd[:, :, 2]) / (2 * eps)
        curl[:, :, 2] = (fwd[:, :, 1] - bwd[:, :, 0]) / (2 * eps)

        return curl

    def forward(self, x):
        if x.shape[1] != 3 and x.shape[2] == 3:
            x = x.permute(0, 2, 1)  # (B, T, 3) → (B, 3, T)

        B, C, T = x.shape
        assert C >= 3 and T == self.seq_length, f"Expected T={self.seq_length}, got {T}"

        x = x[:, :3, :].permute(0, 2, 1).reshape(B * T, 3)  # (B*T, 3)

        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)

        psi = self.potential_proj(x).view(B, T, 3)
        v = self.compute_curl(psi)  # (B, T, 3)

        out = self.fc(v)            # (B, output_dim)
        return out
