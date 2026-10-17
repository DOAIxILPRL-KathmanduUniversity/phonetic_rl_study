"""Transformer Seq2Seq Architecture matching AI4Bharat IndicXlit specifications.
6 Encoder layers, 6 Decoder layers, d_model=256, d_ff=1024, n_heads=4, Pre-LN GELU.
Includes dual Policy logits head and Critic value head.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseSeq2Seq

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 128):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1), :]

class TransformerSeq2Seq(BaseSeq2Seq):
    def __init__(self,
                 vocab_src_size: int,
                 vocab_tgt_size: int,
                 d_model: int = 256,
                 nhead: int = 4,
                 num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6,
                 dim_feedforward: int = 1024,
                 dropout: float = 0.1,
                 pad_idx: int = 0,
                 sos_idx: int = 1,
                 eos_idx: int = 2):
        super().__init__(pad_idx, sos_idx, eos_idx)
        self.d_model = d_model
        
        self.src_embedding = nn.Embedding(vocab_src_size, d_model, padding_idx=pad_idx)
        self.tgt_embedding = nn.Embedding(vocab_tgt_size, d_model, padding_idx=pad_idx)
        self.pos_encoder = SinusoidalPositionalEncoding(d_model)
        self.dropout = nn.Dropout(dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        
        # Dual Heads: Policy head + Value Critic head
        self.policy_head = nn.Linear(d_model, vocab_tgt_size)
        self.critic_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Linear(128, 1)
        )
        
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor = None) -> torch.Tensor:
        src_emb = self.dropout(self.pos_encoder(self.src_embedding(src) * math.sqrt(self.d_model)))
        return self.encoder(src_emb, src_key_padding_mask=src_mask)

    def decode_step(self, tgt_prefix: torch.Tensor, memory: torch.Tensor, src_mask: torch.Tensor = None):
        tgt_emb = self.dropout(self.pos_encoder(self.tgt_embedding(tgt_prefix) * math.sqrt(self.d_model)))
        t = tgt_prefix.size(1)
        tgt_mask = torch.triu(torch.full((t, t), float('-inf'), device=tgt_prefix.device), diagonal=1)
        
        out = self.decoder(tgt_emb, memory, tgt_mask=tgt_mask, memory_key_padding_mask=src_mask)
        last_hidden = out[:, -1, :]
        logits = self.policy_head(last_hidden)
        value = self.critic_head(last_hidden)
        return logits, value

    def forward(self, src: torch.Tensor, tgt: torch.Tensor):
        src_mask = (src == self.pad_idx)
        tgt_mask = (tgt == self.pad_idx)
        t = tgt.size(1)
        causal_mask = torch.triu(torch.full((t, t), float('-inf'), device=tgt.device), diagonal=1)
        
        memory = self.encode(src, src_mask=src_mask)
        tgt_emb = self.dropout(self.pos_encoder(self.tgt_embedding(tgt) * math.sqrt(self.d_model)))
        out = self.decoder(tgt_emb, memory, tgt_mask=causal_mask,
                           tgt_key_padding_mask=tgt_mask,
                           memory_key_padding_mask=src_mask)
                           
        logits = self.policy_head(out)
        values = self.critic_head(out)
        return logits, values
