import torch
import torch.nn as nn
from e3nn.o3 import Irreps, Linear
import math


class E3NNCNNBlock(nn.Module):
    def __init__(self, irreps_in, irreps_out):
        super().__init__()
        self.linear = Linear(irreps_in, irreps_out)
        self.activation = nn.ReLU()  # Can be replaced with e3nn nonlinearity

    def forward(self, x):
        return self.activation(self.linear(x))


class E3NNCNN(nn.Module):
    """
    E3NN-based CNN for SE(3)-equivariant learning on 1D sequences.
    """
    def __init__(self, seq_length, n_features, hidden_irreps="16x0e + 16x1e", out_irreps="1x0e", n_blocks=3, mlp_dim=64):
        super().__init__()
        self.name = "E3NNCNN"
        self.seq_length = seq_length

        self.irreps_in = Irreps("1e")  # 3D vector input
        self.irreps_hidden = Irreps(hidden_irreps)
        self.irreps_out = Irreps(out_irreps)

        # Initial linear projection
        self.input_proj = Linear(self.irreps_in, self.irreps_hidden)

        # Equivariant convolutional blocks
        self.blocks = nn.ModuleList([
            E3NNCNNBlock(self.irreps_hidden, self.irreps_hidden) for _ in range(n_blocks)
        ])

        # Final projection to scalar field
        self.final_linear = Linear(self.irreps_hidden, self.irreps_out)

        # MLP head: initialized with dummy in_features, will be fixed at runtime
        self.fc1 = nn.Linear(1, mlp_dim)  # dummy input dim
        self.fc2 = nn.Linear(mlp_dim, 1)

        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (B, C=3, T)
        Returns:
            Output of shape (B, 1)
        """
        B, C, T = x.shape
        assert C >= 3, f"Expected 3D input (B, 3, T), got {C}"
        x = x[:, :3, :].permute(0, 2, 1).reshape(B * T, 3)  # (B*T, 3)

        # Apply E3NN layers
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        x = self.final_linear(x)  # (B*T, 1)

        # Reshape and flatten
        x = x.view(B, T, -1)  # (B, T, 1)
        x = x.reshape(B, -1)  # (B, T)

        # Dynamically adjust the input dim to the MLP head
        if self.fc1.in_features != x.shape[1]:
            self.fc1 = nn.Linear(x.shape[1], self.fc1.out_features).to(x.device)

        # MLP head
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
