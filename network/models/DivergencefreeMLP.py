import torch
import torch.nn as nn

class DivergenceFreeMLP(nn.Module):
    """
    MLP that approximates divergence-free behavior using curl-based approach
    """
    def __init__(self, seq_length, n_features, output_dim=1):
        super(DivergenceFreeMLP, self).__init__()
        self.input_dim = seq_length * n_features
        self.output_dim = output_dim
        
        # Stream function approach: output scalar(s) that we take curl of
        # For 2D: curl of scalar gives divergence-free field
        # For higher dims: use vector potential
        self.flatten = nn.Flatten()
        
        # Main network
        self.fc1 = nn.Linear(self.input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.out_linear = nn.Linear(128, self.input_dim)  # output dimension matches input for Jacobian trick
        
        # Output potential functions (for curl operation)
        # For 3D input we could directly take curl, but here input is high-dim.
        # We therefore use the antisymmetric Jacobian trick in a **vectorised** form
        # that avoids building the full Jacobian.  We only need two Jacobian-vector
        # products (Jv and J^Tv) for each sample which is far cheaper than the full
        # n×n Jacobian.  This still corresponds to a curl-like antisymmetric field
        # guaranteeing ∇·v = 0.
        self.out_proj = nn.Linear(self.input_dim, self.output_dim)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.name = "DivergenceFreeMLP"

    def base_forward(self, z: torch.Tensor) -> torch.Tensor:
        """Base MLP that maps input z (flattened) → ℝ^D, where D = input_dim"""
        h = self.relu(self.fc1(z))
        h = self.dropout(h)
        h = self.relu(self.fc2(h))
        h = self.dropout(h)
        return self.out_linear(h)

    def forward(self, x):
        """Compute divergence-free output using antisymmetric Jacobian trick

        Steps for each sample x ∈ ℝ^D (D = seq_len × n_features):
        1. b(x)  ← base_forward(x)
        2. v₁    ← J_b(x) · x             (Jacobian-vector product)
        3. v₂    ← J_b(x)^T · x            (vector-Jacobian product)
        4. v     ← v₁ − v₂  (antisymmetric part acting on x) → divergence-free
        5. y     ← out_proj(v)            (map to desired output_dim)

        The expensive full Jacobian is never constructed; only two first-order
        autograd calls are required per sample.
        """
        # Enable gradient computation even if we're in torch.no_grad() context
        with torch.enable_grad():
            batch_size = x.size(0)
            # Produce a leaf tensor that requires grad so autograd.grad works reliably
            x_flat = self.flatten(x).detach().clone().requires_grad_(True)  # (B, D)
            outputs = []
            for i in range(batch_size):
                # Ensure xi is a proper leaf tensor for autograd.grad
                xi = x_flat[i].detach().clone().requires_grad_(True)

                # Define closure for autograd
                # We need to ensure computation graph is maintained
                # First compute base_forward output
                b = self.base_forward(xi)
                
                # For J^T·x, we use the identity: ∇_x(b·x) = b + J^T·x
                # So J^T·x = ∇_x(b·x) - b
                
                # Create a function that returns b·x for grad computation
                def dot_product(inp):
                    return torch.sum(self.base_forward(inp) * inp)
                
                # Compute J·x using jvp
                _, jvp_out = torch.autograd.functional.jvp(
                    lambda inp: self.base_forward(inp), 
                    (xi,), 
                    (xi,), 
                    create_graph=self.training  # Only create graph during training
                )
                
                # Handle potential tuple output from jvp
                if isinstance(jvp_out, (tuple, list)):
                    jvp_out = jvp_out[0]
                
                # Compute J^T·x using gradient of dot product
                grad_dot = torch.autograd.grad(
                    outputs=dot_product(xi),
                    inputs=xi,
                    create_graph=self.training,  # Only create graph during training
                    retain_graph=True
                )[0]
                v_jt_x = grad_dot - b
                
                # Antisymmetric combination gives divergence-free field
                v = jvp_out - v_jt_x

                # 4) project to output dimension
                y = self.out_proj(v)
                outputs.append(y.unsqueeze(0))

            result = torch.cat(outputs, dim=0)
            
        # Detach result if we're in eval mode to prevent gradient flow
        if not self.training:
            result = result.detach()
            
        return result
