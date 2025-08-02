#!/usr/bin/env python3

import torch
import torch.nn as nn
class MLP2(nn.Module):
    def __init__(self, seq_length, n_features):
        super(MLP2, self).__init__()
        input_size = seq_length * n_features
        self.flatten = nn.Flatten()

        self.net = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.GELU(),
            nn.Dropout(0.1),

            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.1),

            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(0.1),

            nn.Linear(64, 1)
        )
        self.name = "MLP2"

    def forward(self, x):
        x = self.flatten(x)
        x = self.net(x)
        return x
