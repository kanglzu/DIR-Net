import torch
import torch.nn as nn
from layers.MLSTM import MLSTMBlock

class DTAM(nn.Module):
    def __init__(self, num_heads, layers, input_size, output,
                 hidden_size=64,
                 batch_first=True, proj_factor_mlstm=2):
        super(DTAM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.layers = layers
        self.num_layers = len(layers)
        self.batch_first = batch_first
        self.proj_factor_mlstm = proj_factor_mlstm
        self.layers = nn.ModuleList()
        self.fc1 = nn.Linear(input_size, output)

        for layer_type in layers:
            if layer_type == 'm':
                layer = MLSTMBlock(input_size, hidden_size, num_heads, proj_factor_mlstm)
            else:
                raise ValueError(f"Invalid layer type: {layer_type}. Only 'm' for mLSTM is supported.")
            self.layers.append(layer)

    def forward(self, x, state=None):
        assert x.ndim == 3
        device = x.device  # Get input tensor device

        if self.batch_first:
            x = x.transpose(0, 1)
        seq_len, batch_size, _ = x.size()

        if state is not None:
            state = torch.stack(list(state))
            assert state.ndim == 4
            num_hidden, state_num_layers, state_batch_size, state_input_size = state.size()
            assert num_hidden == 4
            assert state_num_layers == self.num_layers
            assert state_batch_size == batch_size
            assert state_input_size == self.input_size
            state = state.transpose(0, 1)
        else:
            # Ensure state tensor is on correct device
            state = torch.zeros(self.num_layers, 4, batch_size, self.hidden_size, device=device)

        output = []
        for t in range(seq_len):
            x_t = x[t]
            for layer in range(self.num_layers):
                x_t, state_tuple = self.layers[layer](x_t, tuple(state[layer].clone()))
                state[layer] = torch.stack(list(state_tuple))
            output.append(x_t)

        output = torch.stack(output)
        if self.batch_first:
            output = output.transpose(0, 1)
        state = tuple(state.transpose(0, 1))
        output = self.fc1(output)
        return output 