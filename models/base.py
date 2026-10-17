"""BaseSeq2Seq abstract class providing unified interfaces for all sequence models.
Includes standardized greedy decoding, length-normalized beam search (k=5),
action log-probability extraction for policy gradients, and value critic estimation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import List, Tuple

class BaseSeq2Seq(nn.Module, ABC):
    def __init__(self, pad_idx: int = 0, sos_idx: int = 1, eos_idx: int = 2):
        super().__init__()
        self.pad_idx = pad_idx
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx

    @abstractmethod
    def encode(self, src: torch.Tensor, src_mask: torch.Tensor = None) -> torch.Tensor:
        """Encodes source sequence to memory representations."""
        pass

    @abstractmethod
    def decode_step(self, tgt_prefix: torch.Tensor, memory: torch.Tensor, src_mask: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decodes next character logits and state value given current prefix and memory.
        Returns (logits: [B, vocab_size], value: [B, 1]).
        """
        pass

    @abstractmethod
    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for training. Returns (logits: [B, T_tgt, vocab_tgt], values: [B, T_tgt])."""
        pass

    def generate_greedy(self, src: torch.Tensor, max_len: int = 22) -> torch.Tensor:
        """Standard autoregressive greedy argmax generation."""
        self.eval()
        device = src.device
        batch_size = src.size(0)
        src_mask = (src == self.pad_idx)
        
        with torch.no_grad():
            memory = self.encode(src, src_mask=src_mask)
            generated = torch.full((batch_size, 1), self.sos_idx, dtype=torch.long, device=device)
            finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
            
            for _ in range(max_len):
                logits, _ = self.decode_step(generated, memory, src_mask=src_mask)
                next_tokens = torch.argmax(logits, dim=-1, keepdim=True)
                next_tokens = torch.where(finished.unsqueeze(-1), torch.full_like(next_tokens, self.pad_idx), next_tokens)
                generated = torch.cat([generated, next_tokens], dim=-1)
                
                just_finished = (next_tokens.squeeze(-1) == self.eos_idx)
                finished = finished | just_finished
                if finished.all():
                    break
                    
        return generated

    def generate_beam(self, src: torch.Tensor, beam_size: int = 5, max_len: int = 22, alpha: float = 0.6) -> List[List[Tuple[List[int], float]]]:
        """Vectorized length-normalized batch beam search decoding returning Top-K candidates per batch item."""
        self.eval()
        device = src.device
        B = src.size(0)
        
        with torch.no_grad():
            src_mask = (src == self.pad_idx)
            memory = self.encode(src, src_mask=src_mask)
            
            # Repeat memory and mask for beam_size
            memory_exp = memory.repeat_interleave(beam_size, dim=0)
            src_mask_exp = src_mask.repeat_interleave(beam_size, dim=0) if src_mask is not None else None
            
            # Step 0: Initial prompt with SOS token
            curr_seq = torch.full((B, 1), self.sos_idx, dtype=torch.long, device=device)
            logits, _ = self.decode_step(curr_seq, memory, src_mask=src_mask)
            log_probs = F.log_softmax(logits, dim=-1) # [B, vocab_size]
            topk_log_probs, topk_tokens = torch.topk(log_probs, k=beam_size, dim=-1) # [B, K]
            
            curr_seq = torch.cat([
                curr_seq.repeat_interleave(beam_size, dim=0),
                topk_tokens.view(B * beam_size, 1)
            ], dim=1) # [B * K, 2]
            
            beam_scores = topk_log_probs.view(B * beam_size) # [B * K]
            is_done = (topk_tokens.view(B * beam_size) == self.eos_idx)
            vocab_size = logits.size(-1)
            
            for step in range(2, max_len):
                if is_done.all():
                    break
                logits, _ = self.decode_step(curr_seq, memory_exp, src_mask=src_mask_exp)
                step_log_probs = F.log_softmax(logits, dim=-1) # [B * K, vocab_size]
                
                # Mask finished beams so they only produce eos with 0 additional penalty
                step_log_probs[is_done] = -1e9
                step_log_probs[is_done, self.eos_idx] = 0.0
                
                cand_scores = beam_scores.unsqueeze(-1) + step_log_probs # [B * K, vocab_size]
                cand_scores = cand_scores.view(B, beam_size * vocab_size)
                
                topk_scores, topk_indices = torch.topk(cand_scores, k=beam_size, dim=-1) # [B, K]
                beam_idx = topk_indices // vocab_size # [B, K]
                token_idx = topk_indices % vocab_size # [B, K]
                
                curr_seq = curr_seq.view(B, beam_size, -1)
                idx_exp = beam_idx.unsqueeze(-1).expand(-1, -1, curr_seq.size(-1))
                selected_seqs = torch.gather(curr_seq, 1, idx_exp) # [B, K, step]
                
                curr_seq = torch.cat([selected_seqs, token_idx.unsqueeze(-1)], dim=-1).view(B * beam_size, -1)
                beam_scores = topk_scores.view(B * beam_size)
                prev_done = torch.gather(is_done.view(B, beam_size), 1, beam_idx).view(B * beam_size)
                is_done = prev_done | (token_idx.view(B * beam_size) == self.eos_idx)
                
            curr_seq = curr_seq.view(B, beam_size, -1)
            beam_scores = beam_scores.view(B, beam_size)
            
            results = []
            for b in range(B):
                b_res = []
                for k in range(beam_size):
                    toks = curr_seq[b, k, 1:].tolist()
                    if self.eos_idx in toks:
                        eos_pos = toks.index(self.eos_idx)
                        toks = toks[:eos_pos]
                    score = beam_scores[b, k].item() / (max(1, len(toks)) ** alpha)
                    b_res.append((toks, score))
                b_res.sort(key=lambda x: x[1], reverse=True)
                results.append(b_res)
                
        return results

    def get_action_log_probs(self, src: torch.Tensor, action_seqs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extracts token-level log probabilities and entropies for policy gradient updates."""
        src_mask = (src == self.pad_idx)
        logits, values = self.forward(src, action_seqs[:, :-1])
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Target tokens to evaluate
        targets = action_seqs[:, 1:]
        chosen_log_probs = torch.gather(log_probs, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
        
        # Entropy for regularization
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1)
        
        # Mask out padding
        pad_mask = (targets != self.pad_idx).float()
        chosen_log_probs = chosen_log_probs * pad_mask
        entropy = entropy * pad_mask
        
        return chosen_log_probs, values.squeeze(-1) if values.dim() == 3 else values
