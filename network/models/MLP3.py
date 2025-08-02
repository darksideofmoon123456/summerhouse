#!/usr/bin/env python3

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = self.linear(x)
        out = self.activation(out)
        out = self.dropout(out)
        return x + out  # Residual connection


class MLP3(nn.Module):
    def __init__(self, seq_length, n_features):
        super().__init__()
        input_size = seq_length * n_features
        self.flatten = nn.Flatten()

        self.input_layer = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        self.resblock1 = ResidualBlock(256)
        self.resblock2 = ResidualBlock(256)

        self.output_layer = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )

        self.name = "MLP3"

    def forward(self, x):
        x = self.flatten(x)
        x = self.input_layer(x)
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.output_layer(x)
        return x
