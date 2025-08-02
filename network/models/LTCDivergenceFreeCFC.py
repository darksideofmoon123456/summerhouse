import sys
sys.path.append('CfC')

# clone: https://github.com/raminmh/CfC.git

import torch
import torch.nn as nn
from torch_cfc import Cfc

class CfC_DivFree(nn.Module):
    def __init__(self, input_size, seq_len, hidden_size=64):
        super().__init__()
        self.seq_len = seq_len
        self.cfc = Cfc(in_features=input_size,
                       hidden_size=hidden_size,
                       out_feature=3,
                       hparams=dict(backbone_activation='silu',
                                    backbone_units=128,
                                    backbone_layers=2,
                                    init=1.0),
                       return_sequences=True,
                       use_mixed=True,
                       use_ltc=True)

    @staticmethod
    def curl(potential_A, coords):
        spacing = (coords[:, 1:, :] - coords[:, :-1, :])
        spacing = torch.cat([spacing[:,:1,:], spacing], dim=1)

        grad_A = torch.gradient(potential_A, spacing=spacing, dim=1)
        
        # grad_A is a list of tensors [d/dx, d/dy, d/dz]
        dAx_dy = grad_A[1][..., 0]
        dAx_dz = grad_A[2][..., 0]
        dAy_dx = grad_A[0][..., 1]
        dAy_dz = grad_A[2][..., 1]
        dAz_dx = grad_A[0][..., 2]
        dAz_dy = grad_A[1][..., 2]

        # curl components
        Bx = dAz_dy - dAy_dz
        By = dAx_dz - dAz_dx
        Bz = dAy_dx - dAx_dy

        B = torch.stack([Bx, By, Bz], dim=-1)
        return B

    def forward(self, x, delta_t, coords):
        potential_A = self.cfc(x, timespans=delta_t)
        B = self.curl(potential_A, coords)
        B_mag = torch.sqrt((B**2).sum(dim=-1, keepdim=True))
        return B_mag[:, -1, :]