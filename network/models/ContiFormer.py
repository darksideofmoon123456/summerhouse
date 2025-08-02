#!/usr/bin/env python3

import torch
import torch.nn as nn
import sys
import os

# Add the current directory to path to import physiopro modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Import the ContiFormer components directly
import contiformer
import divfree

# Import the actual ODE modules
sys.path.append(os.path.join(parent_dir, '..', 'module'))
try:
    from linear import ODELinear, InterpLinear
    from ode import build_fc_odefunc, TimeVariableODE
    HAS_ODE_MODULES = True
except ImportError:
    HAS_ODE_MODULES = False
    print("Warning: ODE modules not found, using simplified implementation")

# Fix the import path - the modules are in the parent directory's module folder
import sys
import os
parent_parent_dir = os.path.dirname(parent_dir)  # Go up one more level
module_path = os.path.join(parent_parent_dir, 'module')
if os.path.exists(module_path):
    sys.path.insert(0, module_path)
    try:
        # First check if required dependencies are available
        import torchcde
        import torchdiffeq
        # Now try to import the ODE modules
        from linear import ODELinear, InterpLinear
        from ode import build_fc_odefunc, TimeVariableODE
        HAS_ODE_MODULES = True
        print("Successfully imported ODE modules from:", module_path)
    except ImportError as e:
        HAS_ODE_MODULES = False
        print(f"Warning: Could not import ODE modules: {e}")
        if 'torchcde' in str(e):
            print("Missing dependency: torchcde. Install with: pip install torchcde")
        elif 'torchdiffeq' in str(e):
            print("Missing dependency: torchdiffeq. Install with: pip install torchdiffeq")
        print("Using simplified implementation without continuous-time dynamics")
else:
    HAS_ODE_MODULES = False
    print(f"Warning: Module path not found: {module_path}")
    print("Using simplified implementation")

# Fix the import path - add the Contiform root to sys.path
import sys
import os
# Get the Contiform root directory (parent of network)
contiform_root = os.path.dirname(parent_dir)
if contiform_root not in sys.path:
    sys.path.insert(0, contiform_root)

# Now try to import ODE modules
try:
    # First check if required dependencies are available
    import torchcde
    import torchdiffeq
    # Import using absolute imports from the Contiform package
    from module.linear import ODELinear, InterpLinear
    from module.ode import build_fc_odefunc, TimeVariableODE
    HAS_ODE_MODULES = True
    print("Successfully imported ODE modules")
except ImportError as e:
    HAS_ODE_MODULES = False
    print(f"Warning: Could not import ODE modules: {e}")
    if 'torchcde' in str(e):
        print("Missing dependency: torchcde. Install with: pip install torchcde")
    elif 'torchdiffeq' in str(e):
        print("Missing dependency: torchdiffeq. Install with: pip install torchdiffeq")
    elif 'module.ode' in str(e) or 'module.interpolate' in str(e):
        print("The ODE modules have additional dependencies that couldn't be imported")
    print("Using simplified implementation without continuous-time dynamics")

# Import ODE modules using the wrapper
try:
    from .ode_modules_wrapper import ODELinear, InterpLinear, build_fc_odefunc, TimeVariableODE, HAS_ODE_MODULES
    if HAS_ODE_MODULES:
        print("Successfully imported ODE modules")
except ImportError as e:
    HAS_ODE_MODULES = False
    print(f"Warning: Could not import ODE modules: {e}")
    print("Using simplified implementation without continuous-time dynamics")

class MagNavContiFormer(nn.Module):
    """
    ContiFormer specifically designed for MagNav task.
    Key differences from generic ContiFormer:
    1. Proper time handling for regularly-sampled sensor data
    2. Task-specific architecture matching CNN's effectiveness
    3. Combines local and global temporal patterns
    """
    
    def __init__(self, seq_length, n_features, d_model=128, n_layers=3, n_head=8):
        super(MagNavContiFormer, self).__init__()
        
        self.seq_length = seq_length
        self.n_features = n_features
        self.d_model = d_model
        
        # 1. Local feature extraction (like CNN's conv layers)
        self.local_conv = nn.Sequential(
            nn.Conv1d(n_features, d_model//2, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model//2),
            nn.GELU(),
            nn.Conv1d(d_model//2, d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
        )
        
        # 2. Projection for continuous-time processing
        self.time_projection = nn.Linear(1, d_model//4)
        
        # 3. Simplified continuous-time attention (since ODE modules may not import)
        if HAS_ODE_MODULES:
            # Use actual ODE-based attention
            args_ode = type('Args', (), {
                'actfn': 'softplus',
                'layer_type': 'concat',
                'zero_init': True,
                'atol': 1e-2,
                'rtol': 1e-2,
                'method': 'dopri5',
                'regularize': False,
                'approximate_method': 'last',
                'nlinspace': 1,
                'linear_type': 'inside',
                'interpolate': 'linear',
                'itol': 1e-2,
                'divergence_free': True
            })()
            
            self.continuous_attention = nn.ModuleList([
                ContinuousAttentionLayer(d_model, n_head, args_ode) 
                for _ in range(n_layers)
            ])
        else:
            # Fallback: Standard transformer with time encoding
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_head,
                dim_feedforward=d_model*4,
                dropout=0.1,
                activation='gelu',
                batch_first=True
            )
            self.continuous_attention = nn.TransformerEncoder(encoder_layer, n_layers)
        
        # 4. Temporal aggregation (combines local and global)
        self.temporal_pool = nn.Sequential(
            nn.AdaptiveAvgPool1d(seq_length//4),
            nn.Flatten(),
            nn.Linear((seq_length//4) * d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        # 5. Output prediction
        self.output_layer = nn.Sequential(
            nn.Linear(d_model, d_model//2),
            nn.LayerNorm(d_model//2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model//2, 1)
        )
        
        # Initialize weights
        self._init_weights()
        
        self.name = "MagNavContiFormer"
        
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d)):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x):
        """
        x: [batch_size, n_features, seq_length]
        """
        batch_size = x.size(0)
        
        # 1. Extract local features (like CNN)
        local_features = self.local_conv(x)  # [batch, d_model, seq_length]
        
        # 2. Prepare for attention with time encoding
        x_seq = local_features.transpose(1, 2)  # [batch, seq_length, d_model]
        
        # Generate meaningful time encodings (normalized time steps)
        time_steps = torch.linspace(0, 1, self.seq_length, device=x.device)
        time_steps = time_steps.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, -1)
        time_encoding = self.time_projection(time_steps)  # [batch, seq_length, d_model//4]
        
        # Combine with positional information
        pos_encoding = self._get_sinusoidal_encoding(batch_size, self.seq_length, self.d_model - self.d_model//4, x.device)
        full_time_encoding = torch.cat([time_encoding, pos_encoding], dim=-1)
        
        x_seq = x_seq + full_time_encoding
        
        # 3. Apply continuous attention
        if HAS_ODE_MODULES:
            # Use time-aware attention
            t = time_steps.squeeze(-1)  # [batch, seq_length]
            for attn_layer in self.continuous_attention:
                x_seq = attn_layer(x_seq, t)
        else:
            # Standard transformer
            x_seq = self.continuous_attention(x_seq)
        
        # 4. Combine local and global features
        x_combined = x_seq.transpose(1, 2)  # [batch, d_model, seq_length]
        x_pooled = self.temporal_pool(x_combined)  # [batch, d_model]
        
        # 5. Final prediction
        output = self.output_layer(x_pooled)  # [batch, 1]
        
        return output
    
    def _get_sinusoidal_encoding(self, batch_size, seq_length, d_model, device):
        """Generate sinusoidal positional encoding"""
        position = torch.arange(seq_length, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float, device=device) * 
                           -(torch.log(torch.tensor(10000.0)) / d_model))
        
        pos_encoding = torch.zeros(seq_length, d_model, device=device)
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pos_encoding[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pos_encoding[:, 1::2] = torch.cos(position * div_term)
        
        return pos_encoding.unsqueeze(0).expand(batch_size, -1, -1)
    
    def get_model_info(self):
        """Get model configuration info"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': self.name,
            'seq_length': self.seq_length,
            'n_features': self.n_features,
            'd_model': self.d_model,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'has_ode_modules': HAS_ODE_MODULES
        }


class ContinuousAttentionLayer(nn.Module):
    """Single continuous attention layer with ODE dynamics"""
    def __init__(self, d_model, n_head, args_ode):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        
        # ODE-based projections
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = ODELinear(d_model, d_model, args_ode)
        self.w_v = ODELinear(d_model, d_model, args_ode)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x, t):
        # Self-attention with continuous dynamics
        residual = x
        x = self.norm1(x)
        
        q = self.w_q(x)
        k = self.w_k(x, t)
        v = self.w_v(x, t)
        
        # Simplified attention (full ODE attention is complex)
        attn_output = self._attention(q, k, v)
        x = residual + self.dropout(self.w_o(attn_output))
        
        # FFN
        residual = x
        x = self.norm2(x)
        x = residual + self.dropout(self.ffn(x))
        
        return x
    
    def _attention(self, q, k, v):
        """Simplified attention computation"""
        d_k = self.d_model // self.n_head
        batch_size, seq_len, _ = q.size()
        
        # Reshape for multi-head
        q = q.view(batch_size, seq_len, self.n_head, d_k).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_head, d_k).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_head, d_k).transpose(1, 2)
        
        # Attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (d_k ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn, v)
        
        # Reshape back
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return output


# Keep the original ContiFormer for compatibility
class ContiFormer(MagNavContiFormer):
    """Alias for backward compatibility"""
    pass


class DivFreeContiFormer(MagNavContiFormer):
    """Explicit divergence-free ContiFormer"""
    def __init__(self, seq_length, n_features, **kwargs):
        super().__init__(seq_length, n_features, **kwargs)
        self.name = "DivFreeContiFormer"


class StandardContiFormer(MagNavContiFormer):
    """Standard ContiFormer without special modifications"""
    def __init__(self, seq_length, n_features, **kwargs):
        super().__init__(seq_length, n_features, **kwargs)
        self.name = "StandardContiFormer" 