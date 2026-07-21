import torch
import torch.nn as nn
import torch.nn.functional as F

class DCCLayer(nn.Module):
    def __init__(self, input_size, output_size, kernel_size=3, dilation=1):
        super(DCCLayer, self).__init__()
        self.conv = nn.Conv1d(input_size, output_size, kernel_size, padding=(kernel_size - 1) * dilation,
                              dilation=dilation)
        self.layer_norm = nn.LayerNorm(output_size)

    def forward(self, x):
        x = self.conv(x)
        x = self.layer_norm(x.transpose(1, 2)).transpose(1, 2)  # Apply layer norm
        return F.relu(x)

class DCC(nn.Module):
    def __init__(self, input_size, output_size, num_layers, kernel_size=3):
        super(DCC, self).__init__()
        layers = []
        for i in range(num_layers):
            dilation = 2 ** i  # Exponential dilation
            layers.append(DCCLayer(input_size if i == 0 else output_size, output_size, kernel_size, dilation))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x) 