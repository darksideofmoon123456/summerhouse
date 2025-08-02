"""
Verification script for divergence-free ContiFormer
Tests that the vector fields in the ODE are truly divergence-free
"""

import torch
import torch.nn as nn
from contiformer import ContiFormer
from divfree import check_divergence_free, CurlVectorField


def test_curl_vector_field():
    """Test the CurlVectorField directly"""
    print("=" * 50)
    print("Testing CurlVectorField")
    print("=" * 50)
    
    # Test different dimensions
    for dim in [3, 4, 8, 16]:
        print(f"\nTesting {dim}D vector field:")
        model = CurlVectorField(dim=dim, hidden=64)
        
        # Test on random batch
        x = torch.randn(32, dim, requires_grad=True)
        v = model(x)
        
        # Compute divergence
        div = 0
        for i in range(dim):
            div += torch.autograd.grad(v[:, i].sum(), x, create_graph=False)[0][:, i]
        
        max_div = div.abs().max().item()
        mean_div = div.abs().mean().item()
        
        print(f"  Max |div|: {max_div:.2e}")
        print(f"  Mean |div|: {mean_div:.2e}")
        print(f"  Divergence-free: {'✓' if max_div < 1e-5 else '✗'}")


def test_contiformer_ode():
    """Test the ODEs inside ContiFormer"""
    print("\n" + "=" * 50)
    print("Testing ContiFormer ODEs")
    print("=" * 50)
    
    # Create a small ContiFormer model
    model = ContiFormer(
        input_size=10,
        d_model=16,
        d_inner=64,
        n_layers=2,
        n_head=4,
        d_k=4,
        d_v=4,
        divergence_free=True,  # Enable divergence-free
        atol_ode=1e-3,
        rtol_ode=1e-3,
    )
    
    # Test input
    batch_size = 8
    seq_len = 20
    x = torch.randn(batch_size, seq_len, 10)
    t = torch.linspace(0, 1, seq_len).unsqueeze(0).repeat(batch_size, 1)
    
    print("\nChecking ODE functions in attention layers...")
    
    # Access the ODE functions in the encoder layers
    for i, layer in enumerate(model.encoder.layer_stack):
        print(f"\nLayer {i+1}:")
        
        # Check keys ODE
        if hasattr(layer.slf_attn.w_ks, 'ode'):
            ode_func = layer.slf_attn.w_ks.ode_func
            print("  Keys ODE:")
            if hasattr(ode_func, 'vfield'):
                check_divergence_free(ode_func, dims=model.d_model, n=64)
            else:
                print("    Not using divergence-free field")
        
        # Check values ODE  
        if hasattr(layer.slf_attn.w_vs, 'ode'):
            ode_func = layer.slf_attn.w_vs.ode_func
            print("  Values ODE:")
            if hasattr(ode_func, 'vfield'):
                check_divergence_free(ode_func, dims=model.d_model, n=64)
            else:
                print("    Not using divergence-free field")


def test_forward_pass():
    """Test that the model can still do forward passes"""
    print("\n" + "=" * 50)
    print("Testing Forward Pass")
    print("=" * 50)
    
    model = ContiFormer(
        input_size=10,
        d_model=16,
        d_inner=64,
        n_layers=2,
        n_head=4,
        d_k=4,
        d_v=4,
        divergence_free=True,
    )
    
    # Test input
    x = torch.randn(4, 10, 10)
    
    try:
        output, last = model(x)
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Last hidden state shape: {last.shape}")
        print("Forward pass: ✓")
    except Exception as e:
        print(f"Forward pass failed: {e}")
        print("Forward pass: ✗")


def main():
    """Run all tests"""
    print("Testing Divergence-Free ContiFormer Implementation")
    print("=" * 70)
    
    # Test individual components
    test_curl_vector_field()
    
    # Test integrated system
    test_contiformer_ode()
    
    # Test forward pass
    test_forward_pass()
    
    print("\n" + "=" * 70)
    print("Testing complete!")


if __name__ == "__main__":
    main() 