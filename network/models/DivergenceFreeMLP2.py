import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return x + self.net(x)


class MLP4Backbone(nn.Module):
    """MLP4-like architecture used as the base function for divergence-free computation"""
    def __init__(self, input_dim):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        self.resblocks = nn.Sequential(
            ResidualBlock(512, dropout=0.1),
            ResidualBlock(512, dropout=0.1),
            ResidualBlock(512, dropout=0.1),
            ResidualBlock(512, dropout=0.1)
        )
        self.output_layer = nn.Linear(512, input_dim)  # for antisymmetric trick

    def forward(self, x):
        x = self.input_layer(x)
        x = self.resblocks(x)
        return self.output_layer(x)


class DivergenceFreeMLP2(nn.Module):
    def __init__(self, seq_length, n_features, output_dim=1):
        super().__init__()
        self.input_dim = seq_length * n_features
        self.output_dim = output_dim
        self.flatten = nn.Flatten()

        self.base_mlp = MLP4Backbone(self.input_dim)
        self.out_proj = nn.Linear(self.input_dim, self.output_dim)

        self.name = "DivergenceFreeMLP2"

    def forward(self, x):
        with torch.enable_grad():
            batch_size = x.size(0)
            x_flat = self.flatten(x).detach().clone().requires_grad_(True)
            outputs = []
            for i in range(batch_size):
                xi = x_flat[i].detach().clone().requires_grad_(True)

                def dot_product(inp):
                    return torch.sum(self.base_mlp(inp) * inp)

                b = self.base_mlp(xi)

                _, jvp_out = torch.autograd.functional.jvp(
                    lambda inp: self.base_mlp(inp),
                    (xi,),
                    (xi,),
                    create_graph=self.training
                )
                if isinstance(jvp_out, (tuple, list)):
                    jvp_out = jvp_out[0]

                grad_dot = torch.autograd.grad(
                    outputs=dot_product(xi),
                    inputs=xi,
                    create_graph=self.training,
                    retain_graph=True
                )[0]
                v_jt_x = grad_dot - b

                v = jvp_out - v_jt_x
                y = self.out_proj(v)
                outputs.append(y.unsqueeze(0))

            result = torch.cat(outputs, dim=0)

        if not self.training:
            result = result.detach()
        return result
