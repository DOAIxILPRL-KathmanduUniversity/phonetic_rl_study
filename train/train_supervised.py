"""Supervised Pretraining Engine for Romanized Nepali Transliteration.

Trains sequence models (Mamba, Gated ConvSeq2Seq, IndicXlit Transformer, BiLSTM)
using teacher-forced cross-entropy loss with continuous CER and PER validation.

Usage:
    python train/train_supervised.py --arch mamba --epochs 5 --batch_size 128
"""

import os
import sys
import time
import json
import argparse
import psutil
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models import build_model
from data_loader import TransliterationDataset, CharVocab, load_vocabs, pad_collate_fn
from metrics import compute_cer, compute_per
from train.configs import PRETRAIN_CONFIG


def get_memory_info(device: torch.device) -> Dict[str, float]:
    rss_mb = psutil.Process().memory_info().rss / (1024 * 1024)
    info = {"ram_rss_mb": round(rss_mb, 2)}
    if device.type == "mps" and hasattr(torch.mps, "current_allocated_memory"):
        try:
            mps_mb = torch.mps.current_allocated_memory() / (1024 * 1024)
            info["mps_allocated_mb"] = round(mps_mb, 2)
        except Exception:
            pass
    return info


def evaluate_model(model: nn.Module, 
                   val_loader: DataLoader, 
                   tgt_vocab: CharVocab, 
                   criterion: nn.Module, 
                   device: torch.device,
                   num_eval_batches: int = 50) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_cer = 0.0
    total_per = 0.0
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= num_eval_batches:
                break
            src = batch['src'].to(device)
            tgt = batch['tgt'].to(device)
            refs = batch['tgt_str']
            
            # Forward loss
            logits, _ = model(src, tgt[:, :-1])
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt[:, 1:].reshape(-1))
            total_loss += loss.item()
            
            # Greedy predictions
            preds = model.generate_greedy(src, max_len=22)
            pred_strs = tgt_vocab.decode_batch(preds)
            
            for ref, hyp in zip(refs, pred_strs):
                total_samples += 1
                if ref == hyp:
                    total_correct += 1
                total_cer += compute_cer(ref, hyp)
                total_per += compute_per(ref, hyp)
                
    n = max(1, min(len(val_loader), num_eval_batches))
    s = max(1, total_samples)
    return {
        "val_loss": round(total_loss / n, 4),
        "val_cer": round(total_cer / s, 4),
        "val_per": round(total_per / s, 4),
        "val_top1_acc": round((total_correct / s) * 100.0, 2)
    }


def main():
    parser = argparse.ArgumentParser(description="Supervised Pretraining for Transliteration")
    parser.add_argument("--arch", type=str, default=PRETRAIN_CONFIG["arch"], choices=["mamba", "cnn", "transformer", "lstm"])
    parser.add_argument("--epochs", type=int, default=PRETRAIN_CONFIG["epochs"])
    parser.add_argument("--batch_size", type=int, default=PRETRAIN_CONFIG["batch_size"])
    parser.add_argument("--lr", type=float, default=PRETRAIN_CONFIG["lr"])
    parser.add_argument("--d_model", type=int, default=PRETRAIN_CONFIG["d_model"])
    parser.add_argument("--train_data", type=str, default=PRETRAIN_CONFIG["train_data"])
    parser.add_argument("--val_data", type=str, default=PRETRAIN_CONFIG["val_data"])
    parser.add_argument("--vocab_file", type=str, default=PRETRAIN_CONFIG["vocab_file"])
    parser.add_argument("--checkpoint_dir", type=str, default=PRETRAIN_CONFIG["checkpoint_dir"])
    args = parser.parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[*] Initializing Supervised Pretraining on {device} (Architecture: {args.arch})")

    if not os.path.exists(args.vocab_file):
        raise FileNotFoundError(f"Vocabulary file not found at {args.vocab_file}. Please check data/vocab.json.")
        
    src_vocab, tgt_vocab = load_vocabs(args.vocab_file)
    print(f"[+] Loaded Vocabularies: Source={len(src_vocab)}, Target={len(tgt_vocab)}")

    model = build_model(
        arch=args.arch,
        vocab_src_size=len(src_vocab),
        vocab_tgt_size=len(tgt_vocab),
        d_model=args.d_model,
        pad_idx=src_vocab.pad_idx,
        sos_idx=src_vocab.sos_idx,
        eos_idx=src_vocab.eos_idx
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[+] Model instantiated with {param_count:,} trainable parameters.")

    if not os.path.exists(args.train_data):
        print(f"[!] Warning: Train dataset not found at '{args.train_data}'.")
        print("    Run 'python data/prepare_data.py' first to download and prepare the dataset.")
        return

    train_ds = TransliterationDataset(args.train_data, src_vocab, tgt_vocab, max_len=PRETRAIN_CONFIG["max_len"])
    val_ds = TransliterationDataset(args.val_data, src_vocab, tgt_vocab, max_len=PRETRAIN_CONFIG["max_len"]) if os.path.exists(args.val_data) else None

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=pad_collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=pad_collate_fn) if val_ds else None

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=PRETRAIN_CONFIG["weight_decay"])
    criterion = nn.CrossEntropyLoss(ignore_index=tgt_vocab.pad_idx)

    best_val_cer = 999.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        t0 = time.time()

        for step, batch in enumerate(train_loader):
            src = batch["src"].to(device)
            tgt = batch["tgt"].to(device)

            optimizer.zero_grad()
            logits, _ = model(src, tgt[:, :-1])
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt[:, 1:].reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / max(1, len(train_loader))
        elapsed = time.time() - t0
        mem = get_memory_info(device)

        val_metrics = {}
        if val_loader:
            val_metrics = evaluate_model(model, val_loader, tgt_vocab, criterion, device)
            val_cer = val_metrics["val_cer"]
            print(f"Epoch {epoch:02d}/{args.epochs:02d} | Train Loss: {avg_loss:.4f} | Val Loss: {val_metrics['val_loss']:.4f} | Val CER: {val_cer:.4f} | Val Top-1: {val_metrics['val_top1_acc']}% | Time: {elapsed:.1f}s | RAM: {mem['ram_rss_mb']}MB")
            if val_cer < best_val_cer:
                best_val_cer = val_cer
                save_path = os.path.join(args.checkpoint_dir, f"model_{args.arch}_best.pt")
                torch.save({"model_state_dict": model.state_dict(), "arch": args.arch, "epoch": epoch, "val_metrics": val_metrics}, save_path)
                print(f"  [★] Saved new best checkpoint to {save_path}")
        else:
            print(f"Epoch {epoch:02d}/{args.epochs:02d} | Train Loss: {avg_loss:.4f} | Time: {elapsed:.1f}s | RAM: {mem['ram_rss_mb']}MB")

    print("[✓] Supervised pretraining finished!")


if __name__ == "__main__":
    main()
