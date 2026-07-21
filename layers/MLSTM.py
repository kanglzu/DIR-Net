import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.utils import CausalConv1D, BlockDiagonal
import math

# === TPA Class Implementation ===
class RotaryPositionalEncoding(nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x, seq_dim=1):
        seq_len = x.shape[seq_dim]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.cos_cached = emb.cos()[:, :2 * self.inv_freq.size(0)]
            self.sin_cached = emb.sin()[:, :2 * self.inv_freq.size(0)]
        return self.cos_cached, self.sin_cached

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(x, cos, sin):
    return x * cos + rotate_half(x) * sin

class RoPE_TPA(nn.Module):
    def __init__(self, input_size, head_size, tpa_rank=8):
        super().__init__()
        self.tpa_rank = tpa_rank
        self.head_size = head_size
        self.AQ = nn.Linear(input_size, tpa_rank)
        self.BQ = nn.Linear(input_size, tpa_rank)
        self.AK = nn.Linear(input_size, tpa_rank)
        self.BK = nn.Linear(input_size, tpa_rank)
        self.AV = nn.Linear(input_size, tpa_rank)
        self.BV = nn.Linear(input_size, tpa_rank)
        self.rotary = RotaryPositionalEncoding(tpa_rank)
        self.out_proj = nn.Linear(tpa_rank, head_size)

    def forward(self, x):
        assert x.dim() == 3, f"Input must be 3D [B, T, D], got {x.shape}"
        batch_size, seq_len, _ = x.shape
        AQ = self.AQ(x)
        BQ = self.BQ(x)
        AK = self.AK(x)
        BK = self.BK(x)
        AV = self.AV(x)
        BV = self.BV(x)
        cos, sin = self.rotary(x)
        BQ = apply_rotary_pos_emb(
            BQ.view(-1, seq_len, self.tpa_rank),
            cos.unsqueeze(0).expand(batch_size, -1, -1),
            sin.unsqueeze(0).expand(batch_size, -1, -1)
        ).view(batch_size, seq_len, -1)
        Q = torch.einsum('btr,btr->btr', AQ, BQ)
        K = torch.einsum('btr,btr->btr', AK, BK)
        V = torch.einsum('btr,btr->btr', AV, BV)
        return self.out_proj(Q), self.out_proj(K), self.out_proj(V)

# === mLSTMBlock Class ===
class MLSTMBlock(nn.Module):
    def __init__(self, input_size, hidden_size, num_heads, proj_factor=2, tpa_rank=8):
        super(MLSTMBlock, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads
        self.proj_factor = proj_factor
        self.tpa_rank = tpa_rank
        self.attention_linear = nn.Linear(16, 64)  # Can be adjusted based on hidden_size
        assert hidden_size % num_heads == 0
        assert proj_factor > 0
        self.layer_norm = nn.LayerNorm(input_size)
        self.up_proj_left = nn.Linear(input_size, int(input_size * proj_factor))
        self.up_proj_right = nn.Linear(input_size, hidden_size)
        self.down_proj = nn.Linear(hidden_size, input_size)
        self.causal_conv = CausalConv1D(1, 1, 4)
        self.skip_connection = nn.Linear(int(input_size * proj_factor), hidden_size)
        self.RoPE_TPA = RoPE_TPA(
            input_size=int(input_size * proj_factor),
            head_size=self.head_size,
            tpa_rank=tpa_rank
        )
        self.Wi = nn.Linear(int(input_size * proj_factor), hidden_size)
        self.Wf = nn.Linear(int(input_size * proj_factor), hidden_size)
        self.Wo = nn.Linear(int(input_size * proj_factor), hidden_size)
        self.group_norm = nn.GroupNorm(num_heads, hidden_size)

    def forward(self, x, prev_state):
        h_prev, c_prev, n_prev, m_prev = prev_state
        x_norm = self.layer_norm(x)
        x_up_left = self.up_proj_left(x_norm)
        if x_up_left.dim() == 2:
            x_up_left = x_up_left.unsqueeze(1)
        x_up_right = self.up_proj_right(x_norm)
        qt, kt, vt = self.RoPE_TPA(x_up_left)
        batch_size, seq_len, _ = qt.shape
        qt = qt.view(batch_size, seq_len, self.head_size)
        kt = kt.view(batch_size, seq_len, self.head_size)
        vt = vt.view(batch_size, seq_len, self.head_size)
        attn_scores = torch.einsum('bqd,bkd->bqk', qt, kt) / (self.head_size ** 0.5)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attention_output = torch.einsum('bqk,bkd->bqd', attn_weights, vt)
        attention_output = attention_output.reshape(batch_size, seq_len, -1)
        attention_output = self.attention_linear(attention_output.squeeze(1))
        x_up_left = x_up_left.squeeze(1)
        i_tilde = self.Wi(x_up_left)
        f_tilde = self.Wf(x_up_left)
        o = torch.sigmoid(self.Wo(x_up_left))
        m_t = torch.max(f_tilde + m_prev, i_tilde)
        i = torch.exp(i_tilde - m_t)
        f = torch.exp(f_tilde + m_prev - m_t)
        c_t = f * c_prev + i * attention_output
        n_t = f * n_prev + i * attention_output
        h_t = o * c_t / torch.clamp(attn_scores, min=1e-6).mean(dim=1)
        output = h_t
        output_norm = self.group_norm(output)
        output = output_norm + self.skip_connection(x_up_left)
        output = output * F.silu(x_up_right)
        output = self.down_proj(output)
        final_output = output + x
        return final_output, (h_t, c_t, n_t, m_t) 