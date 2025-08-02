import torch
import torch.nn as nn
import math
from e3nn.o3 import Irreps, Linear
from e3nn.nn import Gate


class E3NNBlock(nn.Module):
    def __init__(self, irreps_in, irreps_scalars, irreps_gates, irreps_gated):
        super().__init__()
        self.irreps_scalars = Irreps(irreps_scalars)
        self.irreps_gates = Irreps(irreps_gates)
        self.irreps_gated = Irreps(irreps_gated)

        # Compute the correct irreps_out expected by Gate
        self.irreps_out = self.irreps_scalars + self.irreps_gates + self.irreps_gated

        self.linear = Linear(irreps_in, self.irreps_out)
        self.gate = Gate(
            self.irreps_scalars, [torch.relu],
            self.irreps_gates, [torch.sigmoid],
            self.irreps_gated
        )

    def forward(self, x):
        x = self.linear(x)
        return self.gate(x)


class E3NNMLP(nn.Module):
    def __init__(self, seq_length, n_features, hidden_dim=32, n_layers=3, dropout=0.2):
        super().__init__()
        assert n_features >= 3
        self.seq_length = seq_length
        self.name = "E3NNMLP"

        # Irreps definitions
        self.irreps_in = Irreps("1x1e")  # input: 3D vector
        self.irreps_scalars = Irreps("16x0e")
        self.irreps_gates = Irreps("16x0e")
        self.irreps_gated = Irreps("16x1e")
        self.irreps_hidden = self.irreps_scalars + self.irreps_gates + self.irreps_gated
        self.irreps_out = Irreps("1x0e")

        self.irreps_scalars = "16x0e"
        self.irreps_gates = "16x0e"
        self.irreps_gated = "16x1e"

        self.block_irreps = Irreps(self.irreps_scalars) + Irreps(self.irreps_gates) + Irreps(self.irreps_gated)

        self.input_proj = Linear(self.irreps_in, self.block_irreps)

        self.blocks = nn.ModuleList([
            E3NNBlock(self.block_irreps, self.irreps_scalars, self.irreps_gates, self.irreps_gated)
            for _ in range(n_layers)
        ])

        # Final projection from GATED irreps only
        self.final_proj = Linear(Irreps(self.irreps_gated), self.irreps_out)



        # Final MLP
        self.fc1 = nn.Linear(self.irreps_out.dim * seq_length, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # Positional encoding
        self.register_buffer("position_vec", torch.tensor([
            math.pow(10000.0, 2.0 * (i // 2) / hidden_dim)
            for i in range(hidden_dim)
        ]))

    def temporal_encoding(self, t):
        encoding = t.unsqueeze(-1) / self.position_vec.to(t.device)
        encoding[:, :, 0::2] = torch.sin(encoding[:, :, 0::2])
        encoding[:, :, 1::2] = torch.cos(encoding[:, :, 1::2])
        return encoding

    def forward(self, x, t=None):
        # x: (B, C, T)
        B, C, T = x.shape
        assert C >= 3 and T == self.seq_length

        x = x[:, :3, :].permute(0, 2, 1).reshape(B * T, 3)  # (B*T, 3)
        print("After reshape:", x.shape)  # Debug print

        assert x.shape[1] == self.irreps_in.dim, f"Input dim mismatch: expected {self.irreps_in.dim}, got {x.shape[1]}"

        x = self.input_proj(x)
        print("After input_proj:", x.shape)  # Debug print

        for block in self.blocks:
            x = block(x)

        x = self.final_proj(x)  # (B*T, 1)
        print("After final_proj:", x.shape)  # Debug print

        x = x.view(B, T, -1)  # Reshape to (B, T, 1)
        print("After reshape to (B, T, -1):", x.shape)  # Debug print

        if t is not None:
            enc = self.temporal_encoding(t)  # (B, T, D)
            enc_crop = enc[:, :, :x.shape[-1]]
            x[:, :, :enc_crop.shape[-1]] += enc_crop

        x = x.reshape(B, -1)  # Flatten temporal dimension
        print("Before final fc:", x.shape)  # Debug print

        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)  # (B, 1)




"""
import torch
import torch.nn as nn
from e3nn.o3 import Irreps, Linear
import math


class E3NNBlock(nn.Module):
    def __init__(self, irreps_in, irreps_out):
        super().__init__()
        self.linear = Linear(irreps_in, irreps_out)
        self.activation = nn.ReLU()  # Can be replaced with e3nn nonlinearity if needed

    def forward(self, x):
        return self.activation(self.linear(x))


class E3NNMLP(nn.Module):

    def __init__(self, seq_length, n_features, hidden_dim=32, n_layers=3, dropout=0.2):
        super().__init__()
        assert n_features >= 3, "Input must have at least 3 features (for 3D vector)"
        self.seq_length = seq_length
        self.name = "E3NNMLP"

        # Define irreps
        self.irreps_in = Irreps("1e")  # 3D vector
        self.irreps_hidden = Irreps("16x0e + 16x1e")
        self.irreps_out = Irreps("1x0e")  # scalar output

        # Input projection
        self.input_proj = Linear(self.irreps_in, self.irreps_hidden)

        # Stack of equivariant blocks
        self.layers = nn.ModuleList([
            E3NNBlock(self.irreps_hidden, self.irreps_hidden) for _ in range(n_layers)
        ])

        # Final projection to scalar output
        self.final_linear = Linear(self.irreps_hidden, self.irreps_out)

        # Fully connected MLP head
        self.fc1 = nn.Linear(self.irreps_out.dim * seq_length, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

        # Temporal encoding scale factors
        self.position_vec = torch.tensor([
            math.pow(10000.0, 2.0 * (i // 2) / hidden_dim)
            for i in range(hidden_dim)
        ])

    def temporal_encoding(self, t):

        # t: (B, T)
        encoding = t.unsqueeze(-1) / self.position_vec.to(t.device)
        encoding[:, :, 0::2] = torch.sin(encoding[:, :, 0::2])
        encoding[:, :, 1::2] = torch.cos(encoding[:, :, 1::2])
        return encoding

    def forward(self, x, t=None):

        B, C, T = x.shape
        assert C >= 3, f"Expected input with at least 3 channels, got {C}"
        assert T == self.seq_length, f"Expected T = {self.seq_length}, got {T}"

        x = x[:, :3, :]                # (B, 3, T)
        x = x.permute(0, 2, 1)         # (B, T, 3)
        x = x.reshape(B * T, 3)        # (B*T, 3)

        # E(3)-equivariant layers
        x = self.input_proj(x)
        for layer in self.layers:
            x = layer(x)
        x = self.final_linear(x)      # (B*T, 1)

        # Reshape to (B, T, 1) then flatten
        x = x.view(B, T, -1)

        # Add temporal encoding if t is provided
        if t is not None:
            enc = self.temporal_encoding(t)  # (B, T, D)
            min_dim = min(enc.shape[-1], x.shape[-1])
            x[:, :, :min_dim] += enc[:, :, :min_dim]

        x = x.view(B, -1)  # flatten to (B, T)

        # MLP head
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
"""







"""
import torch
import torch.nn as nn
from e3nn.o3 import Irreps, Linear

class E3NNMLP(nn.Module):
    def __init__(self, seq_length, n_features):
        super().__init__()
        assert n_features >= 3, "Input must have at least 3 features (for 3D vector)"

        self.seq_length = seq_length
        self.irreps_in = Irreps("1e")
        self.irreps_hidden = Irreps("16x0e + 16x1e")
        self.irreps_out = Irreps("1x0e")  # scalar output

        # Shared equivariant layers
        self.linear_in = Linear(self.irreps_in, self.irreps_hidden)
        self.linear_out = Linear(self.irreps_hidden, self.irreps_out)

        # Fully connected head
        self.fc1 = nn.Linear(self.irreps_out.dim * seq_length, 64)
        self.fc2 = nn.Linear(64, 1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.name = "E3NNMLP"

    def forward(self, x):
        print("Input shape:", x.shape)  # e.g. (B, 15, T)

        # x: (B, C, T)
        x = x[:, :3, :]  # Keep only the first 3 channels: (B, 3, T)
        x = x.permute(0, 2, 1).contiguous()  # (B, T, 3)

        B, T, C = x.shape
        assert C == 3 and T == self.seq_length, f"Got (T={T}, C={C}), expected (T={self.seq_length}, C=3)"

        # Flatten sequence
        x = x.view(B * T, C)  # (B*T, 3)

        # Apply E(3)-equivariant layers
        x = self.linear_in(x)   # (B*T, hidden)
        x = self.linear_out(x)  # (B*T, 1)

        # Reshape back to (B, T, out_dim=1), then flatten
        x = x.view(B, T, -1)  # (B, T, 1)
        x = x.reshape(B, -1)  # (B, T)

        # MLP head
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
"""