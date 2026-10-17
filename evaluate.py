"""Benchmark Evaluation Script for Romanized Nepali Transliteration.

Evaluates checkpoints on test data using both Greedy Argmax and Beam Search (k=5).
Computes:
1. Top-1 Word Accuracy (1 - WER)
2. Top-5 Beam Word Accuracy
3. Character Error Rate (CER)
4. Phonetic Error Rate (PER)
5. Mean Reciprocal Rank (MRR)
6. Character Confusion Substitutions

Usage:
    python evaluate.py --data data/sample_test.csv --beam 5
"""

import os
import sys
import argparse
import time
import pandas as pd
from tqdm import tqdm
import torch

from models import build_model
from data_loader import load_vocabs, CharVocab
from metrics import evaluate_predictions

DEFAULT_CHECKPOINT = "checkpoints/sample_mamba_model.pt"
DEFAULT_VOCAB = "data/vocab.json"
DEFAULT_TEST_CSV = "data/sample_test.csv"


def run_evaluation(checkpoint_path: str,
                   data_path: str,
                   vocab_path: str = DEFAULT_VOCAB,
                   arch: str = "mamba",
                   beam_size: int = 5,
                   max_samples: int = None):
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[*] Running Evaluation on {device}")
    print(f"[*] Checkpoint : {checkpoint_path}")
    print(f"[*] Test Data  : {data_path} (Beam Size: {beam_size})")

    src_vocab, tgt_vocab = load_vocabs(vocab_path)
    model = build_model(
        arch=arch,
        vocab_src_size=len(src_vocab),
        vocab_tgt_size=len(tgt_vocab),
        pad_idx=src_vocab.pad_idx,
        sos_idx=src_vocab.sos_idx,
        eos_idx=src_vocab.eos_idx
    ).to(device)

    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        st = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        model.load_state_dict(st, strict=False)
        print(f"[+] Loaded checkpoint from {checkpoint_path}")
    else:
        print(f"[!] Warning: Checkpoint '{checkpoint_path}' not found! Evaluating randomly initialized model.")

    model.eval()

    df = pd.read_csv(data_path).dropna()
    if max_samples and len(df) > max_samples:
        df = df.iloc[:max_samples]

    references = [str(x).strip() for x in df["nepali"]]
    romans = [str(x).strip().lower() for x in df["roman"]]

    greedy_hypotheses = []
    beam_hypotheses = []

    print(f"[*] Evaluating {len(references)} word pairs...")
    t0 = time.time()

    with torch.no_grad():
        for r_word in tqdm(romans, desc="Evaluating"):
            s_ids = torch.tensor(src_vocab.encode(r_word, add_sos=True, add_eos=True), dtype=torch.long, device=device).unsqueeze(0)
            
            # Greedy
            g_tokens = model.generate_greedy(s_ids, max_len=22)[0]
            greedy_hypotheses.append(tgt_vocab.decode(g_tokens))

            # Beam Top-5
            b_res = model.generate_beam(s_ids, beam_size=beam_size, max_len=22)
            cands = []
            if b_res and len(b_res) > 0:
                for toks, _ in b_res[0]:
                    w = tgt_vocab.decode(toks)
                    if w and w not in cands:
                        cands.append(w)
            if not cands:
                cands = [greedy_hypotheses[-1]]
            beam_hypotheses.append(cands[:beam_size])

    elapsed = time.time() - t0
    metrics = evaluate_predictions(references, greedy_hypotheses, beam_hypotheses)

    print("\n" + "=" * 65)
    print("BENCHMARK EVALUATION RESULTS")
    print("=" * 65)
    print(f"Total Samples Evaluated  : {metrics['num_samples']}")
    print(f"Elapsed Time             : {elapsed:.2f}s ({elapsed / max(1, len(references)) * 1000:.1f} ms/sample)")
    print("-" * 65)
    print(f"Greedy Top-1 Accuracy    : {metrics['top1_accuracy']}%")
    print(f"Beam Top-5 Accuracy      : {metrics['top5_accuracy']}%  (+{metrics['top5_accuracy'] - metrics['top1_accuracy']:.2f}% usability jump)")
    print(f"Mean Reciprocal Rank     : {metrics['mrr']}")
    print(f"Character Error Rate     : {metrics['mean_cer']}")
    print(f"Phonetic Error Rate      : {metrics['mean_per']}  (-{(1 - metrics['mean_per'] / max(0.001, metrics['mean_cer'])) * 100:.1f}% relative reduction)")
    print("=" * 65)

    if metrics.get("top_confusions"):
        print("\nTop Character Confusion Substitutions:")
        for idx, conf in enumerate(metrics["top_confusions"][:8], 1):
            print(f"  {idx}. '{conf['reference']}' -> '{conf['hypothesis']}' : {conf['count']} occurrences ({conf['percentage']}%)")
        print("=" * 65)

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate transliteration model on test data")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT, help="Path to checkpoint")
    parser.add_argument("--data", type=str, default=DEFAULT_TEST_CSV, help="Path to test CSV")
    parser.add_argument("--vocab", type=str, default=DEFAULT_VOCAB, help="Path to vocab.json")
    parser.add_argument("--arch", type=str, default="mamba", choices=["mamba", "cnn", "transformer", "lstm"])
    parser.add_argument("--beam", type=int, default=5, help="Beam size (default: 5)")
    parser.add_argument("--max_samples", type=int, default=None, help="Optional max sample limit")
    args = parser.parse_args()

    run_evaluation(
        checkpoint_path=args.checkpoint,
        data_path=args.data,
        vocab_path=args.vocab,
        arch=args.arch,
        beam_size=args.beam,
        max_samples=args.max_samples
    )


if __name__ == "__main__":
    main()
