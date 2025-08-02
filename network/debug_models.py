import torch
import torch.nn as nn
from models.CNN import CNN
from models.ContiFormer import ContiFormer
import numpy as np

# Create models with same params as in training
seq_len = 20
n_features = 15  # len(features) - 2 (LINE and TRUTH)

print("="*60)
print("MODEL COMPARISON")
print("="*60)

# Create models
cnn = CNN(seq_len, n_features)
contiformer = ContiFormer(seq_len, n_features)

# Count parameters
cnn_params = sum(p.numel() for p in cnn.parameters())
contiformer_params = sum(p.numel() for p in contiformer.parameters())

print(f'\nCNN total parameters: {cnn_params:,}')
print(f'ContiFormer total parameters: {contiformer_params:,}')

# Test forward pass
batch_size = 256
dummy_input = torch.randn(batch_size, n_features, seq_len)

print(f"\nInput shape: {dummy_input.shape}")
print(f"Expected: [batch_size={batch_size}, n_features={n_features}, seq_len={seq_len}]")

# CNN forward
cnn_output = cnn(dummy_input)
print(f"\nCNN output shape: {cnn_output.shape}")

# ContiFormer forward
contiformer_output = contiformer(dummy_input)
print(f"ContiFormer output shape: {contiformer_output.shape}")

# Analyze CNN architecture
print("\n" + "="*60)
print("CNN ARCHITECTURE DETAILS")
print("="*60)
print(cnn)

# Check CNN layer dimensions
print("\nCNN Layer Analysis:")
x = dummy_input
print(f"Input: {x.shape}")

# Conv1: in=15, out=16, kernel=3, stride=1, padding=1
x = nn.Conv1d(15, 16, 3, 1, 1)(x)
print(f"After Conv1: {x.shape}")

# MaxPool1: kernel=2, stride=2
x = nn.MaxPool1d(2, 2)(x)
print(f"After MaxPool1: {x.shape}")

# Conv2: in=16, out=32, kernel=3, stride=1, padding=1
x = nn.Conv1d(16, 32, 3, 1, 1)(x)
print(f"After Conv2: {x.shape}")

# MaxPool2: kernel=2, stride=2
x = nn.MaxPool1d(2, 2)(x)
print(f"After MaxPool2: {x.shape}")

# Flatten
x = x.view(x.size(0), -1)
print(f"After Flatten: {x.shape}")
print(f"  Features: {x.shape[1]} = 32 * (20 // 4) = 32 * 5 = 160")

# Linear layers
print(f"\nCNN Linear layers:")
print(f"  Linear1: 160 -> 64")
print(f"  Linear2: 64 -> 8")
print(f"  Linear3: 8 -> 1")

# Analyze ContiFormer architecture
print("\n" + "="*60)
print("CONTIFORMER ARCHITECTURE DETAILS")
print("="*60)
print(f"Model config: {contiformer.get_model_info()}")

# Check key differences
print("\n" + "="*60)
print("KEY DIFFERENCES")
print("="*60)

print("\n1. INPUT PROCESSING:")
print("   CNN: Direct convolution on [n_features, seq_len]")
print("   ContiFormer: Transpose to [seq_len, n_features] then project")

print("\n2. TEMPORAL MODELING:")
print("   CNN: Local patterns via convolution + pooling")
print("   ContiFormer: Global attention + ODE dynamics")

print("\n3. PARAMETER EFFICIENCY:")
cnn_param_per_feat = cnn_params / n_features
contiformer_param_per_feat = contiformer_params / n_features
print(f"   CNN params per feature: {cnn_param_per_feat:.0f}")
print(f"   ContiFormer params per feature: {contiformer_param_per_feat:.0f}")

print("\n4. INDUCTIVE BIAS:")
print("   CNN: Strong locality bias (good for local patterns)")
print("   ContiFormer: Global dependencies (needs more data)")

# Test with different sequence lengths
print("\n" + "="*60)
print("SEQUENCE LENGTH SENSITIVITY")
print("="*60)

for test_seq_len in [10, 20, 40]:
    test_input = torch.randn(1, n_features, test_seq_len)
    
    # CNN needs fixed architecture for different seq lengths
    if test_seq_len == 20:
        cnn_out = cnn(test_input)
        print(f"Seq length {test_seq_len}: CNN output shape = {cnn_out.shape}")
    else:
        print(f"Seq length {test_seq_len}: CNN requires architecture change")
    
    # ContiFormer should handle variable lengths (but our wrapper might not)
    if test_seq_len == 20:
        cf_out = contiformer(test_input)
        print(f"Seq length {test_seq_len}: ContiFormer output shape = {cf_out.shape}")

# Analyze gradients
print("\n" + "="*60)
print("GRADIENT FLOW ANALYSIS")
print("="*60)

# Create dummy target
target = torch.randn(batch_size, 1)
criterion = nn.MSELoss()

# CNN gradient flow
cnn.zero_grad()
cnn_output = cnn(dummy_input)
cnn_loss = criterion(cnn_output, target)
cnn_loss.backward()

cnn_grad_norms = []
for name, param in cnn.named_parameters():
    if param.grad is not None:
        grad_norm = param.grad.norm().item()
        cnn_grad_norms.append(grad_norm)
        if 'conv' in name or 'linear' in name:
            print(f"CNN {name}: grad_norm = {grad_norm:.6f}")

# ContiFormer gradient flow
contiformer.zero_grad()
contiformer_output = contiformer(dummy_input)
cf_loss = criterion(contiformer_output, target)
cf_loss.backward()

cf_grad_norms = []
for name, param in contiformer.named_parameters():
    if param.grad is not None:
        grad_norm = param.grad.norm().item()
        cf_grad_norms.append(grad_norm)
        if 'linear' in name or 'output' in name:
            print(f"ContiFormer {name}: grad_norm = {grad_norm:.6f}")

print(f"\nAverage gradient norms:")
print(f"  CNN: {np.mean(cnn_grad_norms):.6f}")
print(f"  ContiFormer: {np.mean(cf_grad_norms):.6f}")

print("\n" + "="*60)
print("POTENTIAL ISSUES")
print("="*60)

print("\n1. DATA SCALE:")
print("   - ContiFormer needs normalized inputs (standardization)")
print("   - CNN is more robust to scale variations")

print("\n2. LEARNING DYNAMICS:")
print("   - ContiFormer needs lower learning rate (0.0001 vs 0.001)")
print("   - Different optimizer (AdamW vs Adam)")

print("\n3. ARCHITECTURE MISMATCH:")
print("   - CNN: Designed for this exact task")
print("   - ContiFormer: Generic architecture, may need task-specific tuning")

print("\n4. SEQUENCE MODELING:")
print("   - MagNav might benefit more from local patterns (CNN strength)")
print("   - Global dependencies might add noise (ContiFormer weakness)")

print("\n5. TRAINING TIME:")
print("   - ContiFormer likely needs more epochs to converge")
print("   - Early stopping might be premature") 