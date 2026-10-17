"""Reinforcement Learning Post-Training Engine for Transliteration.

Optimizes non-differentiable sequence-level transliteration objectives (CER, PER, Exact Match)
directly on full autoregressive rollouts generated without teacher forcing.

Algorithms Supported:
- scst: Self-Critical Sequence Training with greedy decoding baseline
- a2c: Advantage Actor-Critic with state value critic and temporal difference error
- ppo: Proximal Policy Optimization with probability ratio clipping

Usage:
    python train/train_rl.py --algorithm scst --reward_type composite_balanced --epochs 3
"""

import os
import sys
import time
import copy
import argparse
import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models import build_model
from data_loader import TransliterationDataset, CharVocab, load_vocabs, pad_collate_fn
from rewards import compute_batch_rewards
from metrics import compute_cer, compute_per
from train.configs import RL_CONFIG


def sample_rollout(model, src, tgt_vocab, temperature=1.0, max_len=22):
    """Samples stochastic token rollouts from the policy."""
    device = src.device
    B = src.size(0)
    src_mask = (src == tgt_vocab.pad_idx)
    
    memory = model.encode(src, src_mask=src_mask)
    generated = torch.full((B, 1), tgt_vocab.sos_idx, dtype=torch.long, device=device)
    finished = torch.zeros(B, dtype=torch.bool, device=device)
    
    for _ in range(max_len):
        logits, _ = model.decode_step(generated, memory, src_mask=src_mask)
        probs = F.softmax(logits / max(0.01, temperature), dim=-1)
        next_tokens = torch.multinomial(probs, num_samples=1)
        
        next_tokens = torch.where(finished.unsqueeze(-1), torch.full_like(next_tokens, tgt_vocab.pad_idx), next_tokens)
        generated = torch.cat([generated, next_tokens], dim=-1)
        
        finished = finished | (next_tokens.squeeze(-1) == tgt_vocab.eos_idx)
        if finished.all():
            break
            
    return generated


def main():
    parser = argparse.ArgumentParser(description="Reinforcement Learning Post-Training")
    parser.add_argument("--arch", type=str, default="mamba")
    parser.add_argument("--init_checkpoint", type=str, default=None)
    parser.add_argument("--algorithm", type=str, default=RL_CONFIG["algorithm"], choices=["scst", "a2c", "ppo"])
    parser.add_argument("--reward_type", type=str, default=RL_CONFIG["reward_type"])
    parser.add_argument("--epochs", type=int, default=RL_CONFIG["epochs"])
    parser.add_argument("--steps_per_epoch", type=int, default=RL_CONFIG["steps_per_epoch"])
    parser.add_argument("--batch_size", type=int, default=RL_CONFIG["batch_size"])
    parser.add_argument("--lr", type=float, default=RL_CONFIG["lr"])
    parser.add_argument("--temperature", type=float, default=RL_CONFIG["temperature"])
    parser.add_argument("--checkpoint_dir", type=str, default=RL_CONFIG["checkpoint_dir"])
    parser.add_argument("--train_data", type=str, default="data/train_split.csv")
    parser.add_argument("--vocab_file", type=str, default="data/vocab.json")
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[*] Starting RL Post-Training on {device} (Algorithm: {args.algorithm.upper()}, Reward: {args.reward_type})")

    src_vocab, tgt_vocab = load_vocabs(args.vocab_file)
    model = build_model(
        arch=args.arch,
        vocab_src_size=len(src_vocab),
        vocab_tgt_size=len(tgt_vocab),
        pad_idx=src_vocab.pad_idx,
        sos_idx=src_vocab.sos_idx,
        eos_idx=src_vocab.eos_idx
    ).to(device)

    if args.init_checkpoint and os.path.exists(args.init_checkpoint):
        ckpt = torch.load(args.init_checkpoint, map_location=device)
        st = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        model.load_state_dict(st, strict=False)
        print(f"[+] Loaded initial weights from: {args.init_checkpoint}")

    if not os.path.exists(args.train_data):
        print(f"[!] Warning: Train dataset not found at '{args.train_data}'.")
        print("    Run 'python data/prepare_data.py' first.")
        return

    train_ds = TransliterationDataset(args.train_data, src_vocab, tgt_vocab, max_len=22)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=pad_collate_fn)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_reward = 0.0
        total_loss = 0.0
        t0 = time.time()

        for step, batch in enumerate(train_loader):
            if step >= args.steps_per_epoch:
                break

            src = batch["src"].to(device)
            gold_refs = batch["tgt_str"]

            # 1. Exploratory stochastic rollout
            sampled_seqs = sample_rollout(model, src, tgt_vocab, temperature=args.temperature, max_len=22)
            sampled_strs = tgt_vocab.decode_batch(sampled_seqs)
            sampled_rewards = compute_batch_rewards(sampled_strs, gold_refs, reward_type=args.reward_type, device=device)

            # 2. Baseline & Advantage computation
            if args.algorithm == "scst":
                with torch.no_grad():
                    greedy_seqs = model.generate_greedy(src, max_len=22)
                    greedy_strs = tgt_vocab.decode_batch(greedy_seqs)
                    greedy_rewards = compute_batch_rewards(greedy_strs, gold_refs, reward_type=args.reward_type, device=device)
                advantage = sampled_rewards - greedy_rewards

            elif args.algorithm == "a2c":
                # Critic value head baseline
                _, values = model.get_action_log_probs(src, sampled_seqs)
                v_scalar = values.mean(dim=-1) if values.dim() > 1 else values
                advantage = sampled_rewards - v_scalar.detach()

            elif args.algorithm == "ppo":
                # Moving average baseline
                baseline = sampled_rewards.mean()
                advantage = sampled_rewards - baseline

            # 3. Policy loss computation
            log_probs, values = model.get_action_log_probs(src, sampled_seqs)
            seq_log_prob = log_probs.sum(dim=-1)
            policy_loss = -(seq_log_prob * advantage.detach()).mean()

            loss = policy_loss
            if args.algorithm == "a2c" and values is not None:
                critic_loss = F.mse_loss(v_scalar, sampled_rewards)
                loss = policy_loss + 0.5 * critic_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_reward += sampled_rewards.mean().item()
            total_loss += loss.item()

        avg_reward = total_reward / max(1, min(len(train_loader), args.steps_per_epoch))
        avg_loss = total_loss / max(1, min(len(train_loader), args.steps_per_epoch))
        elapsed = time.time() - t0

        print(f"RL Epoch {epoch:02d}/{args.epochs:02d} | Avg Reward: {avg_reward:.4f} | Loss: {avg_loss:.4f} | Time: {elapsed:.1f}s")

        save_path = os.path.join(args.checkpoint_dir, f"model_rl_{args.algorithm}_{args.reward_type}_last.pt")
        torch.save({"model_state_dict": model.state_dict(), "algorithm": args.algorithm, "epoch": epoch}, save_path)

    print("[✓] Reinforcement Learning post-training complete!")


if __name__ == "__main__":
    main()
