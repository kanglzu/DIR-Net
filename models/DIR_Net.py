import torch
import torch.nn as nn
import math

from layers.DCC import DCC
from layers.B_Spline_KAN import KAN
from layers.DTAM import DTAM
from layers.MLSTM import MLSTMBlock
from layers.utils import CausalConv1D, BlockDiagonal
from layers.decomp import DECOMP

# RevIN Module
class RevIN(nn.Module):
    def __init__(self, num_features, affine=True, subtract_last=False):
        super().__init__()
        self.affine = affine
        self.subtract_last = subtract_last
        if self.affine:
            self.gamma = nn.Parameter(torch.ones(1, 1, num_features))
            self.beta = nn.Parameter(torch.zeros(1, 1, num_features))

    def forward(self, x, mode):
        if mode == 'norm':
            if self.subtract_last:
                self.last = x[:, -1:, :]
                x = x - self.last
            self.mean = x.mean(dim=1, keepdim=True)
            self.std = x.std(dim=1, keepdim=True)
            x = (x - self.mean) / (self.std + 1e-5)
            if self.affine:
                x = x * self.gamma + self.beta
            return x
        elif mode == 'denorm':
            x = (x - self.beta) / (self.gamma + 1e-5) if self.affine else x
            x = x * (self.std + 1e-5) + self.mean
            if self.subtract_last:
                x = x + self.last
            return x
        else:
            raise ValueError('mode must be norm or denorm')

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.num_heads = args.num_heads
        self.layers2 = args.layers2
        self.hidden_size = args.hidden_size
        self.fc1_size = args.fc1_size
        self.fc2_size = args.fc2_size
        self.pred_len = args.pred_len
        self.seq_len = args.seq_len
        self.revin = getattr(args, 'revin', False)
        self.revin_affine = getattr(args, 'revin_affine', True)
        self.revin_subtract_last = getattr(args, 'revin_subtract_last', False)
        self.ma_type = getattr(args, 'ma_type', 'ema')
        self.alpha = getattr(args, 'alpha', 0.3)
        self.beta = getattr(args, 'beta', 0.3)
        self.enc_in = args.enc_in
        
        self.input_linear = nn.Linear(args.enc_in, self.hidden_size)
        self.DTAM_output = nn.Linear(self.seq_len, self.pred_len)
        self.fc_tcn_output = nn.Linear(self.seq_len + 2 * (2 ** 3 - 1), self.pred_len)
        # mLSTM layer
        self.DTAM = DTAM(self.num_heads, self.layers2, self.hidden_size, self.fc1_size)
        # TCN layer
        self.DCC = DCC(self.hidden_size, self.fc1_size, num_layers=3)
        # Dynamic gating
        self.fusion_gate = nn.Sequential(
            nn.Linear(self.fc1_size * 2, self.fc1_size),
            nn.Sigmoid()
        )
        # KAN layer
        kan_hidden = [self.fc1_size, self.fc2_size]
        self.kan = KAN(layers_hidden=kan_hidden)
        # Decomposition
        self.decomp = DECOMP(self.ma_type, self.alpha, self.beta)
        # RevIN
        if self.revin:
            self.revin_layer = RevIN(1, affine=self.revin_affine, subtract_last=self.revin_subtract_last)

    def forward(self, x):
        # RevIN normalization
        if self.revin:
            x = self.revin_layer(x, 'norm')
        # Decomposition
        if self.ma_type == 'reg':
            seasonal_init, trend_init = x, x
        else:
            seasonal_init, trend_init = self.decomp(x)
        # Input linear transformation
        trend_init = self.input_linear(trend_init)
        seasonal_init = self.input_linear(seasonal_init)
        # mLSTM
        DTAM_out = self.DTAM(trend_init)
        DTAM_out = DTAM_out.transpose(1, 2)
        DTAM_out = self.DTAM_output(DTAM_out)
        DTAM_out = DTAM_out.transpose(1, 2)
        
        # TCN
        DCC_in = seasonal_init.transpose(1, 2)  # (B, C, L)
        DCC_out = self.DCC(DCC_in)
        DCC_out = self.fc_tcn_output(DCC_out)
        DCC_out = DCC_out.transpose(1, 2)  # (B, L, C)
        # Concatenate and gate
        combined = torch.cat((DTAM_out, DCC_out), dim=-1)
        gate_value = self.fusion_gate(combined)
        gated_output = gate_value * DTAM_out + (1 - gate_value) * DCC_out
        # KAN
        output = self.kan(gated_output)
        # RevIN denormalization
        if self.revin:
            output = self.revin_layer(output, 'denorm')
        return output 