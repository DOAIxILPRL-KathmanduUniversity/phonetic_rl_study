"""BiLSTM Seq2Seq Architecture with Bahdanau Additive Attention.
3-Layer Bidirectional LSTM Encoder + 3-Layer LSTM Decoder.
Stateless and functional: decoder hidden state initialized directly from encoder memory.
Includes dual Policy logits head and Critic value head.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseSeq2Seq

class BahdanauAttention(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.W_s = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_h = nn.Linear(hidden_dim * 2, hidden_dim, bias=False)
        self.v_a = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, dec_hidden: torch.Tensor, enc_outputs: torch.Tensor, mask: torch.Tensor = None):
        # dec_hidden: [B, 1, hidden_dim]
        # enc_outputs: [B, T_src, 2 * hidden_dim]
        score = torch.tanh(self.W_s(dec_hidden) + self.W_h(enc_outputs))
        energy = self.v_a(score).squeeze(-1)
        
        if mask is not None:
            energy = energy.masked_fill(mask, -1e9)
            
        attn_weights = F.softmax(energy, dim=-1).unsqueeze(1)
        context = torch.bmm(attn_weights, enc_outputs)
        return context, attn_weights

class LSTMSeq2Seq(BaseSeq2Seq):
    def __init__(self,
                 vocab_src_size: int,
                 vocab_tgt_size: int,
                 embed_dim: int = 256,
                 hidden_dim: int = 256,
                 num_layers: int = 3,
                 dropout: float = 0.1,
                 pad_idx: int = 0,
                 sos_idx: int = 1,
                 eos_idx: int = 2):
        super().__init__(pad_idx, sos_idx, eos_idx)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.src_embedding = nn.Embedding(vocab_src_size, embed_dim, padding_idx=pad_idx)
        self.tgt_embedding = nn.Embedding(vocab_tgt_size, embed_dim, padding_idx=pad_idx)
        self.dropout = nn.Dropout(dropout)
        
        self.encoder = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.attention = BahdanauAttention(hidden_dim)
        
        self.decoder = nn.LSTM(
            input_size=embed_dim + hidden_dim * 2,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Projections to initialize decoder state from pooled encoder memory
        self.enc_to_dec_h = nn.Linear(hidden_dim * 2, hidden_dim * num_layers)
        self.enc_to_dec_c = nn.Linear(hidden_dim * 2, hidden_dim * num_layers)
        
        # Dual Heads
        self.policy_head = nn.Linear(hidden_dim, vocab_tgt_size)
        self.critic_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1)
        )

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor = None) -> torch.Tensor:
        src_emb = self.dropout(self.src_embedding(src))
        enc_outputs, _ = self.encoder(src_emb)
        return enc_outputs

    def _init_dec_state(self, memory: torch.Tensor, mask: torch.Tensor = None):
        """Initializes decoder states statelessly from pooled encoder memory."""
        B = memory.size(0)
        if mask is not None:
            # Masked mean pool
            valid_mask = (~mask).unsqueeze(-1).float()
            pooled = (memory * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1.0)
        else:
            pooled = memory.mean(dim=1)
            
        h0 = torch.tanh(self.enc_to_dec_h(pooled)).view(B, self.num_layers, self.hidden_dim).transpose(0, 1).contiguous()
        c0 = torch.tanh(self.enc_to_dec_c(pooled)).view(B, self.num_layers, self.hidden_dim).transpose(0, 1).contiguous()
        return h0, c0

    def decode_step(self, tgt_prefix: torch.Tensor, memory: torch.Tensor, src_mask: torch.Tensor = None):
        B = tgt_prefix.size(0)
        tgt_emb = self.dropout(self.tgt_embedding(tgt_prefix))
        dec_h, dec_c = self._init_dec_state(memory, mask=src_mask)
        
        out = None
        for t in range(tgt_prefix.size(1)):
            curr_emb = tgt_emb[:, t:t+1, :]
            top_h = dec_h[-1:].transpose(0, 1)
            context, _ = self.attention(top_h, memory, mask=src_mask)
            dec_in = torch.cat([curr_emb, context], dim=-1)
            out, (dec_h, dec_c) = self.decoder(dec_in, (dec_h, dec_c))
            
        last_hidden = out[:, -1, :]
        logits = self.policy_head(last_hidden)
        value = self.critic_head(last_hidden)
        return logits, value

    def forward(self, src: torch.Tensor, tgt: torch.Tensor):
        src_mask = (src == self.pad_idx)
        memory = self.encode(src, src_mask=src_mask)
        
        B, T_tgt = tgt.size()
        tgt_emb = self.dropout(self.tgt_embedding(tgt))
        dec_h, dec_c = self._init_dec_state(memory, mask=src_mask)
        
        step_outputs = []
        for t in range(T_tgt):
            curr_emb = tgt_emb[:, t:t+1, :]
            top_h = dec_h[-1:].transpose(0, 1)
            context, _ = self.attention(top_h, memory, mask=src_mask)
            dec_in = torch.cat([curr_emb, context], dim=-1)
            out, (dec_h, dec_c) = self.decoder(dec_in, (dec_h, dec_c))
            step_outputs.append(out)
            
        all_hidden = torch.cat(step_outputs, dim=1)
        logits = self.policy_head(all_hidden)
        values = self.critic_head(all_hidden)
        return logits, values
