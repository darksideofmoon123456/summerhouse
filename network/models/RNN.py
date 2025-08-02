#!/usr/bin/env python3

import torch
import torch.nn as nn

class LSTM(nn.Module):
    """Simple LSTM model for MagNav"""
    def __init__(self, seq_length, hidden_size, num_layers, num_LSTM, num_linear, num_neurons, device):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size[0] if isinstance(hidden_size, list) else hidden_size
        self.num_layers = num_layers[0] if isinstance(num_layers, list) else num_layers
        self.lstm = nn.LSTM(seq_length, self.hidden_size, self.num_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, 1)
        self.name = "LSTM"
        
    def forward(self, x):
        # x shape: (batch_size, n_features, seq_length)
        x = x.transpose(1, 2)  # (batch_size, seq_length, n_features)
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Take last output
        return out

class GRU(nn.Module):
    """Simple GRU model for MagNav"""
    def __init__(self, seq_length, n_features, hidden_size):
        super(GRU, self).__init__()
        self.gru = nn.GRU(n_features, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.name = "GRU"
        
    def forward(self, x):
        # x shape: (batch_size, n_features, seq_length)
        x = x.transpose(1, 2)  # (batch_size, seq_length, n_features)
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])  # Take last output
        return out 