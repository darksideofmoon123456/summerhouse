#!/usr/bin/env python3

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return x + self.net(x)  # Residual connection


class MLP4(nn.Module):
    def __init__(self, seq_length, n_features):
        super().__init__()
        input_size = seq_length * n_features
        self.flatten = nn.Flatten()

        self.input_layer = nn.Sequential(
            nn.Linear(input_size, 512),  # Wider
            nn.GELU(),
            nn.Dropout(0.1)
        )

        # Residual blocks
        self.resblocks = nn.Sequential(
            ResidualBlock(512, dropout=0.1),
            ResidualBlock(512, dropout=0.1),
            ResidualBlock(512, dropout=0.1),
            ResidualBlock(512, dropout=0.1)
        )

        self.output_layer = nn.Sequential(
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )

        self.name = "MLP4"

    def forward(self, x):
        x = self.flatten(x)
        x = self.input_layer(x)
        x = self.resblocks(x)
        x = self.output_layer(x)
        return x
