import math
import torch
from torch import nn
from torch.nn import functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout, max_len):
        super().__init__()

        assert d_model % n_heads == 0

        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out_proj = nn.Linear(d_model, d_model)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # 🔥 precompute mask (IMPORTANT for RL speed)
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(max_len, max_len)).bool()
        )

    def forward(self, x):
        B, T, C = x.shape

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        scores = scores.masked_fill(
            ~self.causal_mask[:T, :T],
            float("-inf")
        )

        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)

        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        out = self.resid_dropout(out)

        return self.out_proj(out)




class FeedForward(nn.Module):
    def __init__(self, d_model, dropout):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)




class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout, max_len):
        super().__init__()

        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout, max_len)

        self.ln2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class grpo_transformer(nn.Module):
    def __init__(self,vocab_size,config):
        super().__init__()

        self.token_emb = nn.Embedding(vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.max_length, config.d_model)

        self.blocks = nn.ModuleList([
            TransformerBlock(
                config.d_model,
                config.n_heads,
                config.dropout,
                config.max_length   # 🔥 needed for mask
            )
            for _ in range(config.n_layers)
        ])

        self.ln_f = nn.LayerNorm(config.d_model)

        self.head = nn.Linear(config.d_model, vocab_size, bias=False)

    def forward(self, input_ids):
        B, T = input_ids.shape

        pos = torch.arange(T, device=input_ids.device).unsqueeze(0)

        x = self.token_emb(input_ids) + self.pos_emb(pos)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)

        logits = self.head(x)

        return logits