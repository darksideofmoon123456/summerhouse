
import torch
import torch.nn as nn
from e3nn.o3 import Irreps, Linear
import math


class E3NNBlock(nn.Module):
    def __init__(self, irreps_in, irreps_out):
        super().__init__()
        self.linear = Linear(irreps_in, irreps_out)
        self.activation = nn.ReLU()

    def forward(self, x):
        return self.activation(self.linear(x))


class E3NNDivergenceFreeMLP(nn.Module):
    def __init__(self, seq_length, n_features, hidden_dim=32, n_layers=3, dropout=0.2):
        super().__init__()
        assert n_features >= 3, "Input must have at least 3 features (for 3D vector)"
        self.seq_length = seq_length
        self.name = "E3NNDivergenceFreeMLP"

        # Define irreps
        self.irreps_in = Irreps("1e")  # input: 3D vector
        self.irreps_hidden = Irreps("16x0e + 16x1e")
        self.irreps_potential = Irreps("1x1e")  # vector potential ψ
        self.irreps_out = Irreps("1x0e")  # scalar output

        # Input projection
        self.input_proj = Linear(self.irreps_in, self.irreps_hidden)

        # Stack of equivariant blocks
        self.layers = nn.ModuleList([
            E3NNBlock(self.irreps_hidden, self.irreps_hidden) for _ in range(n_layers)
        ])

        # Project to vector potential field ψ (dim=3)
        self.potential_proj = Linear(self.irreps_hidden, self.irreps_potential)

        # Fully connected MLP head
        self.fc1 = nn.Linear(3 * seq_length, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

        # Temporal encoding scale factors
        self.position_vec = torch.tensor([
            math.pow(10000.0, 2.0 * (i // 2) / hidden_dim)
            for i in range(hidden_dim)
        ])

    def temporal_encoding(self, t):
        encoding = t.unsqueeze(-1) / self.position_vec.to(t.device)
        encoding[:, :, 0::2] = torch.sin(encoding[:, :, 0::2])
        encoding[:, :, 1::2] = torch.cos(encoding[:, :, 1::2])
        return encoding

    def compute_curl_approx(self, psi, eps=1e-1):
        B, T, _ = psi.shape
        curl = torch.zeros_like(psi)

        # Forward and backward shift
        fwd = torch.roll(psi, shifts=-1, dims=1)
        bwd = torch.roll(psi, shifts=1, dims=1)

        # Approximate finite difference curl: v = ∇ × ψ
        curl[:, :, 0] = (fwd[:, :, 2] - bwd[:, :, 1]) / (2 * eps)
        curl[:, :, 1] = (fwd[:, :, 0] - bwd[:, :, 2]) / (2 * eps)
        curl[:, :, 2] = (fwd[:, :, 1] - bwd[:, :, 0]) / (2 * eps)

        return curl

    def forward(self, x, t=None):
        B, C, T = x.shape
        assert C >= 3, f"Expected input with at least 3 channels, got {C}"
        assert T == self.seq_length, f"Expected T = {self.seq_length}, got {T}"

        x = x[:, :3, :]            # (B, 3, T)
        x = x.permute(0, 2, 1)     # (B, T, 3)
        x = x.reshape(B * T, 3)    # (B*T, 3)

        # Equivariant layers
        x = self.input_proj(x)
        for layer in self.layers:
            x = layer(x)

        # Vector potential
        psi = self.potential_proj(x)         # (B*T, 3)
        psi = psi.view(B, T, 3)              # (B, T, 3)

        # Compute divergence-free field using curl
        curl = self.compute_curl_approx(psi)  # (B, T, 3)

        # Add temporal encoding if provided
        if t is not None:
            enc = self.temporal_encoding(t)  # (B, T, D)
            min_dim = min(enc.shape[-1], curl.shape[-1])
            curl[:, :, :min_dim] += enc[:, :, :min_dim]

        # Flatten and feed to MLP head
        curl_flat = curl.reshape(B, -1)       # (B, 3*T)
        x = self.relu(self.fc1(curl_flat))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


