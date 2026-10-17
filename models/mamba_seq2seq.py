"""Pure PyTorch Selective State Space (Mamba-Style SSM) Seq2Seq Architecture.
Bidirectional Selective SSM Encoder + Causal SSM Decoder with Cross-Attention.
Includes dual Policy logits head and Critic value head.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseSeq2Seq

class SelectiveSSMBlock(nn.Module):
    """Selective State-Space (S4/Mamba) core block in pure PyTorch."""
    def __init__(self, d_model: int, d_state: int = 16, causal: bool = False):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.causal = causal
        
        # Projections for selective dynamics (B, C, Delta)
        self.x_proj = nn.Linear(d_model, d_state * 2 + d_model, bias=False)
        self.dt_proj = nn.Linear(d_model, d_model, bias=True)
        
        # Fixed state transition matrix A (initialized with HiPPO heuristic)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_model, 1)
        self.A_log = nn.Parameter(torch.log(A)) # [d_model, d_state]
        self.D = nn.Parameter(torch.ones(d_model))
        
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        residual = x
        B, T, D = x.shape
        
        proj = self.x_proj(x) # [B, T, 2*d_state + D]
        B_ssm, C_ssm, dt_in = torch.split(proj, [self.d_state, self.d_state, D], dim=-1)
        
        delta = F.softplus(self.dt_proj(dt_in)) # [B, T, D]
        A = -torch.exp(self.A_log) # [D, d_state]
        
        # Discretize: delta * A
        # Recurrent scan over sequence length
        h = torch.zeros(B, D, self.d_state, device=x.device)
        ys = []
        
        for t in range(T):
            dt = delta[:, t, :].unsqueeze(-1) # [B, D, 1]
            A_bar = torch.exp(dt * A.unsqueeze(0)) # [B, D, d_state]
            B_bar = dt * B_ssm[:, t, :].unsqueeze(1) # [B, 1, d_state] -> [B, D, d_state] via broadcast
            x_t = x[:, t, :].unsqueeze(-1) # [B, D, 1]
            
            h = A_bar * h + B_bar * x_t # [B, D, d_state]
            C_t = C_ssm[:, t, :].unsqueeze(1) # [B, 1, d_state]
            y_t = (h * C_t).sum(dim=-1) # [B, D]
            ys.append(y_t)
            
        y = torch.stack(ys, dim=1) # [B, T, D]
        y = y + x * self.D
        return self.norm(residual + self.out_proj(F.silu(y)))

class MambaSeq2Seq(BaseSeq2Seq):
    def __init__(self,
                 vocab_src_size: int,
                 vocab_tgt_size: int,
                 d_model: int = 256,
                 d_state: int = 16,
                 num_layers: int = 4,
                 dropout: float = 0.1,
                 pad_idx: int = 0,
                 sos_idx: int = 1,
                 eos_idx: int = 2):
        super().__init__(pad_idx, sos_idx, eos_idx)
        self.d_model = d_model
        
        self.src_embedding = nn.Embedding(vocab_src_size, d_model, padding_idx=pad_idx)
        self.tgt_embedding = nn.Embedding(vocab_tgt_size, d_model, padding_idx=pad_idx)
        self.dropout = nn.Dropout(dropout)
        
        self.encoder_layers = nn.ModuleList([
            SelectiveSSMBlock(d_model, d_state=d_state, causal=False) for _ in range(num_layers)
        ])
        
        self.decoder_layers = nn.ModuleList([
            SelectiveSSMBlock(d_model, d_state=d_state, causal=True) for _ in range(num_layers)
        ])
        
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)
        self.cross_norm = nn.LayerNorm(d_model)
        
        self.policy_head = nn.Linear(d_model, vocab_tgt_size)
        self.critic_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Linear(128, 1)
        )

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor = None) -> torch.Tensor:
        x = self.dropout(self.src_embedding(src))
        for layer in self.encoder_layers:
            x = layer(x)
        return x

    def decode_step(self, tgt_prefix: torch.Tensor, memory: torch.Tensor, src_mask: torch.Tensor = None):
        x = self.dropout(self.tgt_embedding(tgt_prefix))
        for layer in self.decoder_layers:
            x = layer(x)
            
        # Cross-Attention to encoder memory
        attn_out, _ = self.cross_attn(x, memory, memory, key_padding_mask=src_mask)
        x = self.cross_norm(x + attn_out)
        
        last_hidden = x[:, -1, :]
        logits = self.policy_head(last_hidden)
        value = self.critic_head(last_hidden)
        return logits, value

    def forward(self, src: torch.Tensor, tgt: torch.Tensor):
        src_mask = (src == self.pad_idx)
        memory = self.encode(src, src_mask=src_mask)
        
        x = self.dropout(self.tgt_embedding(tgt))
        for layer in self.decoder_layers:
            x = layer(x)
            
        attn_out, _ = self.cross_attn(x, memory, memory, key_padding_mask=src_mask)
        x = self.cross_norm(x + attn_out)
        
        logits = self.policy_head(x)
        values = self.critic_head(x)
        return logits, values
