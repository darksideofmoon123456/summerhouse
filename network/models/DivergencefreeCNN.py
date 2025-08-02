import torch
import torch.nn as nn
import torch.nn.functional as F

class DivergenceFreeCNN(nn.Module):
    """
    CNN-based model that outputs a divergence-free vector field
    using the antisymmetric Jacobian trick (v = Jb·x - Jbᵗ·x)
    """
    def __init__(self, seq_length, n_features, output_dim=1):
        super(DivergenceFreeCNN, self).__init__()
        self.seq_length = seq_length
        self.n_features = n_features
        self.input_dim = seq_length * n_features
        self.output_dim = output_dim

        # CNN backbone for base feature extraction
        self.cnn = nn.Sequential(
            nn.Conv1d(n_features, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Flatten()
        )

        # Dummy forward to get the flattened dimension after CNN
        with torch.no_grad():
            dummy = torch.zeros(1, n_features, seq_length)
            dummy_out = self.cnn(dummy)
            self.cnn_out_dim = dummy_out.shape[1]

        # Fully connected layers after CNN
        self.fc = nn.Sequential(
            nn.Linear(self.cnn_out_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, self.input_dim)  # same size as input vector
        )

        self.out_proj = nn.Linear(self.input_dim, self.output_dim)
        self.name = "DivergenceFreeCNN"

    def base_forward(self, z):
        h = self.cnn(z)
        return self.fc(h)

    def forward(self, x):
        """
        x: shape (B, C, T)  [C = n_features, T = seq_length]
        Returns divergence-free output using antisymmetric Jacobian trick.
        """
        with torch.enable_grad():
            batch_size = x.size(0)
            x_in = x.detach().clone().requires_grad_(True)
            outputs = []

            for i in range(batch_size):
                xi = x_in[i].unsqueeze(0).detach().clone().requires_grad_(True)

                # Make sure the shape is [1, C=n_features, T=seq_len]
                if xi.shape[1] != self.n_features:
                    xi = xi.permute(0, 2, 1)

                b = self.base_forward(xi)

                def dot_product(inp):
                    return torch.sum(self.base_forward(inp) * inp.view(1, -1))

                v = torch.ones_like(xi)

                # Compute J·x
                _, jvp_out = torch.autograd.functional.jvp(
                    lambda inp: self.base_forward(inp), 
                    (xi,), 
                    (v,),
                    create_graph=self.training
                )

                # Compute Jᵗ·x
                grad_dot = torch.autograd.grad(
                    outputs=dot_product(xi),
                    inputs=xi,
                    create_graph=self.training,
                    retain_graph=True
                )[0]
                v_jt_x = grad_dot.view(1, -1) - b

                # Antisymmetric part
                v_out = jvp_out - v_jt_x
                y = self.out_proj(v_out)
                outputs.append(y)

            result = torch.cat(outputs, dim=0)

        if not self.training:
            result = result.detach()

        return result
