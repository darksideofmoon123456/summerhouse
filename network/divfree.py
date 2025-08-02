# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import torch.nn as nn
import torch.autograd as autograd


class CurlVectorField(nn.Module):
    """
    Learns ψ: R^d → R^d  (vector potential)  and returns v = curl ψ  (d=3)  *or*
    the antisymmetric-Jacobian divergence-free field for general d.
    """

    def __init__(self, dim: int, hidden: int = 64):
        super().__init__()
        self.dim = dim
        # simple 2-layer MLP for ψ
        self.psi = nn.Sequential(
            nn.Linear(dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, dim)            # ψ(x)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (..., d)  →  v(x): (..., d)   with  div v = 0  (up to autograd precision).
        """
        # We need to enable gradients for the autograd operations
        with torch.enable_grad():
            # Ensure x requires grad for autograd operations
            if not x.requires_grad:
                x = x.detach().requires_grad_(True)
            
            if self.dim == 3:                     # explicit curl
                psi = self.psi(x)                 # (...,3)
                grads = []
                for i in range(3):
                    grad_i = autograd.grad(
                        psi[..., i].sum(), x, create_graph=True, retain_graph=True
                    )[0]                          # (...,3)
                    grads.append(grad_i)
                # grads[i][..., j] = ∂ψ_i/∂x_j
                curl = torch.stack((
                    grads[2][..., 1] - grads[1][..., 2],
                    grads[0][..., 2] - grads[2][..., 0],
                    grads[1][..., 0] - grads[0][..., 1]
                ), dim=-1)
                return curl

            # general-d antisymmetric-Jacobian construction
            psi = self.psi(x)                     # (...,d)
            J = []
            for i in range(self.dim):
                J_i = autograd.grad(
                    psi[..., i].sum(), x, create_graph=True, retain_graph=True
                )[0]                              # (...,d)
                J.append(J_i)
            J = torch.stack(J, dim=-2)            # (...,d,d)
            A = J - J.transpose(-1, -2)           # antisymmetric

            # row-wise divergence: v_i = Σ_j ∂A_ij/∂x_j
            v = []
            for i in range(self.dim):
                div_row = 0
                for j in range(self.dim):
                    div_row += autograd.grad(
                        A[..., i, j].sum(), x, create_graph=True, retain_graph=True
                    )[0][..., j]
                v.append(div_row)
            v = torch.stack(v, dim=-1)            # (...,d)
            return v


class DivergenceFreeODEFunc(nn.Module):
    """
    ODE function wrapper that uses CurlVectorField to ensure divergence-free dynamics.
    This can replace the build_fc_odefunc in the original implementation.
    """
    
    def __init__(self, dim: int, hidden_dims=None, **kwargs):
        super().__init__()
        # Use default hidden size if not specified
        hidden_size = hidden_dims[0] if hidden_dims else 64
        self.vfield = CurlVectorField(dim, hidden_size)
        
    def forward(self, t, x):
        """
        t: time (can be ignored for autonomous systems)
        x: state vector
        Returns divergence-free vector field
        """
        return self.vfield(x)


def check_divergence_free(model, dims=3, n=128):
    """
    Verify that the vector field is divergence-free
    """
    x = torch.randn(n, dims, requires_grad=True)
    
    # For DivergenceFreeODEFunc, we need to call it with time parameter
    if hasattr(model, 'vfield'):
        # This is a DivergenceFreeODEFunc, call it with time
        t = torch.zeros(1)  # dummy time
        with torch.enable_grad():
            v = model(t, x)
    else:
        # Direct CurlVectorField
        with torch.enable_grad():
            v = model(x)
    
    # Compute divergence
    div = torch.zeros(n)
    for i in range(dims):
        # We need to compute gradients one at a time to avoid graph issues
        v_i = v[:, i]
        grad_i = torch.autograd.grad(
            outputs=v_i.sum(), 
            inputs=x, 
            retain_graph=True,  # Keep graph for next iteration
            create_graph=False
        )[0][:, i]
        div += grad_i
    
    max_div = div.abs().max().item()
    print(f'max |div|: {max_div:.2e}')
    
    # Divergence-free means div(v) = 0, but due to numerical precision,
    # we get very small values instead of exactly zero
    if max_div < 1e-6:
        print(f'✓ Vector field is divergence-free (within numerical precision)')
    else:
        print(f'✗ Vector field is NOT divergence-free')
    
    return max_div < 1e-5 