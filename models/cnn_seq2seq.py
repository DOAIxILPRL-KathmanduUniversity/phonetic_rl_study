"""Gated Convolutional Seq2Seq Architecture (ConvSeq2Seq with GLU).
6 Gated Convolutional Encoder layers + 6 Causal Gated Decoder layers with Attention.
Includes dual Policy logits head and Critic value head.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseSeq2Seq

class ConvEncoderLayer(nn.Module):
    def __init__(self, embed_dim: int, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        self.conv = nn.Conv1d(embed_dim, embed_dim * 2, kernel_size=kernel_size, padding=kernel_size // 2)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D] -> transpose to [B, D, T]
        residual = x
        x_conv = self.dropout(x.transpose(1, 2))
        x_conv = self.conv(x_conv)
        # GLU gating
        a, b = x_conv.chunk(2, dim=1)
        x_glu = (a * torch.sigmoid(b)).transpose(1, 2)
        return self.norm(residual + x_glu)

class ConvDecoderLayer(nn.Module):
    def __init__(self, embed_dim: int, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = kernel_size - 1
        self.conv = nn.Conv1d(embed_dim, embed_dim * 2, kernel_size=kernel_size)
        self.attn_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor, memory: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # x: [B, T, D]
        residual = x
        x_conv = self.dropout(x.transpose(1, 2))
        x_conv = F.pad(x_conv, (self.padding, 0)) # Causal left padding
        x_conv = self.conv(x_conv)
        a, b = x_conv.chunk(2, dim=1)
        x_glu = (a * torch.sigmoid(b)).transpose(1, 2)
        
        # Cross-Attention over memory
        q = self.attn_proj(x_glu) # [B, T_tgt, D]
        scores = torch.bmm(q, memory.transpose(1, 2)) / math.sqrt(x.size(-1)) # [B, T_tgt, T_src]
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1), -1e9)
        attn_weights = F.softmax(scores, dim=-1)
        context = torch.bmm(attn_weights, memory) # [B, T_tgt, D]
        
        return self.norm(residual + x_glu + context)

class CNNSeq2Seq(BaseSeq2Seq):
    def __init__(self,
                 vocab_src_size: int,
                 vocab_tgt_size: int,
                 embed_dim: int = 256,
                 num_layers: int = 6,
                 kernel_size: int = 3,
                 dropout: float = 0.1,
                 pad_idx: int = 0,
                 sos_idx: int = 1,
                 eos_idx: int = 2):
        super().__init__(pad_idx, sos_idx, eos_idx)
        self.embed_dim = embed_dim
        
        self.src_embedding = nn.Embedding(vocab_src_size, embed_dim, padding_idx=pad_idx)
        self.tgt_embedding = nn.Embedding(vocab_tgt_size, embed_dim, padding_idx=pad_idx)
        self.pos_emb = nn.Embedding(128, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        self.encoder_layers = nn.ModuleList([
            ConvEncoderLayer(embed_dim, kernel_size, dropout) for _ in range(num_layers)
        ])
        
        self.decoder_layers = nn.ModuleList([
            ConvDecoderLayer(embed_dim, kernel_size, dropout) for _ in range(num_layers)
        ])
        
        self.policy_head = nn.Linear(embed_dim, vocab_tgt_size)
        self.critic_head = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1)
        )

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor = None) -> torch.Tensor:
        pos = torch.arange(src.size(1), device=src.device).unsqueeze(0)
        x = self.dropout(self.src_embedding(src) + self.pos_emb(pos))
        for layer in self.encoder_layers:
            x = layer(x)
        return x

    def decode_step(self, tgt_prefix: torch.Tensor, memory: torch.Tensor, src_mask: torch.Tensor = None):
        pos = torch.arange(tgt_prefix.size(1), device=tgt_prefix.device).unsqueeze(0)
        x = self.dropout(self.tgt_embedding(tgt_prefix) + self.pos_emb(pos))
        for layer in self.decoder_layers:
            x = layer(x, memory, mask=src_mask)
            
        last_hidden = x[:, -1, :]
        logits = self.policy_head(last_hidden)
        value = self.critic_head(last_hidden)
        return logits, value

    def forward(self, src: torch.Tensor, tgt: torch.Tensor):
        src_mask = (src == self.pad_idx)
        memory = self.encode(src, src_mask=src_mask)
        
        pos = torch.arange(tgt.size(1), device=tgt.device).unsqueeze(0)
        x = self.dropout(self.tgt_embedding(tgt) + self.pos_emb(pos))
        for layer in self.decoder_layers:
            x = layer(x, memory, mask=src_mask)
            
        logits = self.policy_head(x)
        values = self.critic_head(x)
        return logits, values
