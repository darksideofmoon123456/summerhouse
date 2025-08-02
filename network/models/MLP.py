#!/usr/bin/env python3

import torch
import torch.nn as nn

class MLP(nn.Module):
    """Simple MLP model for MagNav"""
    def __init__(self, seq_length, n_features):
        super(MLP, self).__init__()
        input_size = seq_length * n_features
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.name = "MLP"
        
    def forward(self, x):
        # x shape: (batch_size, n_features, seq_length)
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x 