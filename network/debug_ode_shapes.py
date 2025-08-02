import torch
import sys
import os

# Add paths
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
module_path = os.path.join(parent_dir, 'module')
sys.path.insert(0, module_path)

# Import modules
from linear import ODELinear, InterpLinear
from ode import build_fc_odefunc, TimeVariableODE

# Create a simple test
class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self

# Test parameters
batch_size = 128
seq_len = 20
d_model = 256

# Create ODE args
args_ode = AttrDict({
    "actfn": "softplus",
    "layer_type": "concat",
    "zero_init": True,
    "atol": 1e-3,
    "rtol": 1e-3,
    "method": "rk4",
    "regularize": False,
    "approximate_method": "last",
    "nlinspace": 3,
    "linear_type": "inside",
    "interpolate": "linear",
    "itol": 1e-2,
    "divergence_free": False,
})

# Create modules
ode_linear = ODELinear(d_model, d_model, args_ode)
interp_linear = InterpLinear(d_model, d_model, args_ode)

# Create test inputs
x = torch.randn(batch_size, seq_len, d_model)
t = torch.linspace(0, 1, seq_len).unsqueeze(0).repeat(batch_size, 1)

print("Input shapes:")
print(f"x: {x.shape}")
print(f"t: {t.shape}")

# Test ODE forward
print("\nTesting ODELinear...")
try:
    ode_out = ode_linear(x, t)
    print(f"ODELinear output shape: {ode_out.shape}")
    print(f"Expected shape: ({batch_size}, {seq_len}, {d_model})")
except Exception as e:
    print(f"Error in ODELinear: {e}")

# Test Interp forward
print("\nTesting InterpLinear...")
try:
    interp_out = interp_linear(x, t)
    print(f"InterpLinear output shape: {interp_out.shape}")
    print(f"Expected shape: ({batch_size}, {seq_len}, {d_model})")
except Exception as e:
    print(f"Error in InterpLinear: {e}") 