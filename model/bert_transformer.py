import math
import torch
from torch import nn
from torch.nn import functional as F


#############################################
# TRANSFORMER ENCODER FROM SCRATCH
#############################################

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads
        self.n_heads = n_heads

        self.qkv = nn.Linear(d_model, d_model * 3)
        self.fc = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)

        # if mask is not None:
        #     attn = attn.masked_fill(mask == 0, float("-inf"))
        if mask is not None:
            # (B, T) -> (B, 1, 1, T)
            mask = mask.unsqueeze(1).unsqueeze(2)

            attn = attn.masked_fill(mask == 0, float("-inf"))

        attn = torch.softmax(attn, dim=-1)
        out = attn @ v

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.fc(out)

class EncoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, ff_hidden=512):
        super().__init__()
        self.attn = MultiHeadSelfAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_hidden),
            nn.ReLU(),
            nn.Linear(ff_hidden, d_model)
        )

    def forward(self, x, mask=None):
        x = self.norm1(x + self.attn(x, mask))
        x = self.norm2(x + self.ff(x))
        return x

class EncoderTransformer(nn.Module):
    def __init__(self, vocab_size,config):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.max_length, config.d_model)

        self.layers = nn.ModuleList([
            EncoderBlock(config.d_model, config.n_heads)
            for _ in range(config.n_layers)
        ])

        self.classifier = nn.Linear(config.d_model, config.n_classes)

    def forward(self, input_ids, attention_mask):
        B, T = input_ids.shape
        pos = torch.arange(0, T, device=input_ids.device).unsqueeze(0)

        x = self.token_emb(input_ids) + self.pos_emb(pos)

        for layer in self.layers:
            x = layer(x, attention_mask)

        # CLS pooling (simple mean pooling)
        x = x.mean(dim=1)
        return self.classifier(x)
