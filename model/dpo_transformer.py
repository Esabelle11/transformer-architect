import math
import torch
from torch import nn
from torch.nn import functional as F


class CausalSelfAttention(nn.Module):

    def __init__(self, d_model, n_heads, dropout):
        super().__init__()

        assert d_model % n_heads == 0

        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.qkv = nn.Linear(d_model, d_model * 3)

        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        B, T, C = x.shape

        qkv = self.qkv(x)

        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1))

        scores = scores / math.sqrt(self.head_dim)

        causal_mask = torch.tril(
            torch.ones(T, T, device=x.device)
        ).bool()

        scores = scores.masked_fill(
            ~causal_mask,
            float("-inf")
        )

        attn = F.softmax(scores, dim=-1)

        attn = self.dropout(attn)

        out = torch.matmul(attn, v)

        out = out.transpose(1, 2).contiguous()

        out = out.view(B, T, C)

        out = self.out_proj(out)

        return out
        
class FeedForward(nn.Module):

    def __init__(self, d_model, dropout):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):

    def __init__(self, d_model, n_heads, dropout):
        super().__init__()

        self.ln1 = nn.LayerNorm(d_model)

        self.attn = CausalSelfAttention(
            d_model,
            n_heads,
            dropout
        )

        self.ln2 = nn.LayerNorm(d_model)

        self.ff = FeedForward(
            d_model,
            dropout
        )

    def forward(self, x):

        attn_out = self.attn(self.ln1(x))

        x = x + attn_out

        ff_out = self.ff(self.ln2(x))

        x = x + ff_out

        return x

class DPO_transformer(nn.Module):

    def __init__(self,vocab_size,config):
        super().__init__()

        self.token_emb = nn.Embedding(
            vocab_size,
            config.d_model
        )

        self.pos_emb = nn.Embedding(
            config.max_length,
            config.d_model
        )

        self.blocks = nn.ModuleList([
            TransformerBlock(
                config.d_model,
                config.n_heads,
                config.dropout
            )
            for _ in range(config.n_layers)
        ])

        self.ln_f = nn.LayerNorm(config.d_model)

        self.head = nn.Linear(
            config.d_model,
            vocab_size,
            bias=False
        )

    def forward(self, input_ids):

        B, T = input_ids.shape

        positions = torch.arange(
            0,
            T,
            device=input_ids.device
        ).unsqueeze(0)

        tok = self.token_emb(input_ids)

        pos = self.pos_emb(positions)

        x = tok + pos

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)

        logits = self.head(x)

        return logits