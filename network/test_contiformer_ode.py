"""
Test script for ODE-based ContiFormer implementation
"""

import torch
import torch.nn as nn
from contiformer import ContiFormer
import sys

def test_contiformer_ode():
    """Test the ODE-based ContiFormer with MagNav-like data"""
    
    print("Testing ODE-based ContiFormer implementation...")
    print("-" * 50)
    
    # Test parameters
    batch_size = 4
    seq_len = 20
    n_features = 15
    
    # Create model
    model = ContiFormer(
        input_size=n_features,
        d_model=256,
        d_inner=1024,
        n_layers=2,  # Reduced for testing
        n_head=4,
        d_k=64,
        d_v=64,
        dropout=0.1,
        max_length=seq_len,
        # ODE parameters
        actfn_ode="softplus",
        layer_type_ode="concat",
        zero_init_ode=True,
        atol_ode=1e-3,
        rtol_ode=1e-3,
        method_ode="rk4",
        linear_type_ode="inside",
        regularize=False,
        approximate_method="last",
        nlinspace=3,
        interpolate_ode="linear",
        itol_ode=1e-2,
        divergence_free=True,
        add_pe=False,
        normalize_before=False
    )
    
    print(f"Model created successfully!")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test with MagNav input shape: [batch, features, seq_len]
    x_magnav = torch.randn(batch_size, n_features, seq_len)
    print(f"\nTesting with MagNav input shape: {x_magnav.shape}")
    
    try:
        # Forward pass
        with torch.no_grad():
            output = model(x_magnav)
        print(f"Output shape: {output.shape}")
        print(f"Expected shape: [{batch_size}, 1]")
        print(f"✓ MagNav format works correctly!")
        
    except Exception as e:
        print(f"✗ Error with MagNav format: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
    
    # Test with standard transformer input shape: [batch, seq_len, features]
    x_standard = torch.randn(batch_size, seq_len, n_features)
    print(f"\nTesting with standard input shape: {x_standard.shape}")
    
    try:
        with torch.no_grad():
            output = model(x_standard)
        print(f"Output shape: {output.shape}")
        print(f"✓ Standard format works correctly!")
        
    except Exception as e:
        print(f"✗ Error with standard format: {e}")
        print(f"Error type: {type(e).__name__}")
    
    # Test gradient flow
    print("\nTesting gradient flow...")
    x = torch.randn(batch_size, n_features, seq_len, requires_grad=True)
    output = model(x)
    loss = output.mean()
    
    try:
        loss.backward()
        print("✓ Gradient computation successful!")
        
        # Check if gradients are flowing
        grad_norms = []
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norms.append((name, param.grad.norm().item()))
        
        print(f"\nGradient norms for first 5 layers:")
        for name, norm in grad_norms[:5]:
            print(f"  {name}: {norm:.6f}")
            
    except Exception as e:
        print(f"✗ Gradient computation failed: {e}")
    
    print("\n" + "-" * 50)
    print("Testing complete!")


if __name__ == "__main__":
    test_contiformer_ode() 