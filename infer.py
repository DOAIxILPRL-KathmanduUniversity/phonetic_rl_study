"""Interactive Inference CLI & Python API for Romanized Nepali Transliteration.

Supports:
1. Greedy argmax decoding (Top-1)
2. Length-normalized Beam Search (Top-5 candidate ribbon)
3. Python API (TransliterationEngine) for embedding in web services or scripts

Usage:
    python infer.py --word rama --beam 5
    python infer.py --interactive
"""

import os
import sys
import argparse
from typing import List, Dict, Any, Optional
import torch

from models import build_model
from data_loader import load_vocabs, CharVocab

DEFAULT_CHECKPOINT = "checkpoints/sample_mamba_model.pt"
DEFAULT_VOCAB = "data/vocab.json"


class TransliterationEngine:
    def __init__(self, 
                 checkpoint_path: str = DEFAULT_CHECKPOINT,
                 vocab_path: str = DEFAULT_VOCAB,
                 arch: str = "mamba",
                 device: Optional[torch.device] = None):
        self.device = device or torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.vocab_path = vocab_path
        self.checkpoint_path = checkpoint_path
        self.arch = arch

        if not os.path.exists(self.vocab_path):
            raise FileNotFoundError(f"Vocabulary file not found at: {self.vocab_path}")

        self.src_vocab, self.tgt_vocab = load_vocabs(self.vocab_path)
        self.model = build_model(
            arch=self.arch,
            vocab_src_size=len(self.src_vocab),
            vocab_tgt_size=len(self.tgt_vocab),
            pad_idx=self.src_vocab.pad_idx,
            sos_idx=self.src_vocab.sos_idx,
            eos_idx=self.src_vocab.eos_idx
        ).to(self.device)

        if os.path.exists(self.checkpoint_path):
            ckpt = torch.load(self.checkpoint_path, map_location=self.device)
            st = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
            self.model.load_state_dict(st, strict=False)
            print(f"[+] Loaded weights from {self.checkpoint_path}")
        else:
            print(f"[!] Warning: Checkpoint not found at '{self.checkpoint_path}'. Initialized with random weights.")

        self.model.eval()

    def transliterate(self, text: str, beam_size: int = 5, max_len: int = 22) -> Dict[str, Any]:
        """Transliterates a single Romanized word into Devanagari with beam search candidates."""
        clean_text = text.strip().lower()
        if not clean_text:
            return {"input": text, "top1": "", "candidates": []}

        src_ids = torch.tensor(
            self.src_vocab.encode(clean_text, add_sos=True, add_eos=True),
            dtype=torch.long,
            device=self.device
        ).unsqueeze(0)

        with torch.no_grad():
            # 1. Greedy top-1
            greedy_tokens = self.model.generate_greedy(src_ids, max_len=max_len)[0]
            greedy_str = self.tgt_vocab.decode(greedy_tokens)

            # 2. Beam search top-k
            beam_results = self.model.generate_beam(src_ids, beam_size=beam_size, max_len=max_len)
            candidate_list = []
            if beam_results and len(beam_results) > 0:
                for token_ids, score in beam_results[0]:
                    cand_word = self.tgt_vocab.decode(token_ids)
                    if cand_word and cand_word not in candidate_list:
                        candidate_list.append(cand_word)

            if not candidate_list:
                candidate_list = [greedy_str]

        return {
            "input": clean_text,
            "top1": candidate_list[0],
            "greedy": greedy_str,
            "candidates": candidate_list[:beam_size]
        }

    def transliterate_sentence(self, sentence: str, beam_size: int = 5) -> str:
        """Transliterates a space-separated sentence by mapping word by word."""
        words = sentence.strip().split()
        outputs = []
        for w in words:
            # Preserve punctuation if needed
            res = self.transliterate(w, beam_size=beam_size)
            outputs.append(res["top1"])
        return " ".join(outputs)


def interactive_cli(engine: TransliterationEngine, beam_size: int = 5):
    print("=" * 65)
    print("ROMANIZED NEPALI TRANSLITERATION ENGINE (CLI)")
    print(f"Device: {engine.device} | Architecture: {engine.arch}")
    print("Type any Roman word (e.g., 'rama', 'nepal', 'namaskar', 'kitab')")
    print("Type 'exit' or 'quit' to terminate.")
    print("=" * 65)

    while True:
        try:
            word = input("\nRoman Input > ").strip()
            if not word:
                continue
            if word.lower() in ["exit", "quit", "q"]:
                print("Session terminated. Dhanyabad!")
                break

            res = engine.transliterate(word, beam_size=beam_size)
            print("-" * 50)
            print(f"  Top-1 Output        : {res['top1']}")
            print(f"  Greedy Output       : {res['greedy']}")
            print(f"  Top-{beam_size} IME Ribbon : {' | '.join(res['candidates'])}")
            print("-" * 50)
        except (KeyboardInterrupt, EOFError):
            print("\nSession exited.")
            break


def main():
    parser = argparse.ArgumentParser(description="Inference CLI for Roman to Nepali Transliteration")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT, help="Path to model checkpoint")
    parser.add_argument("--vocab", type=str, default=DEFAULT_VOCAB, help="Path to vocab.json")
    parser.add_argument("--arch", type=str, default="mamba", choices=["mamba", "cnn", "transformer", "lstm"])
    parser.add_argument("--word", type=str, default=None, help="Single word to transliterate")
    parser.add_argument("--sentence", type=str, default=None, help="Sentence to transliterate")
    parser.add_argument("--beam", type=int, default=5, help="Beam size for Top-K candidate generation")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive terminal session")
    args = parser.parse_args()

    engine = TransliterationEngine(checkpoint_path=args.checkpoint, vocab_path=args.vocab, arch=args.arch)

    if args.word:
        res = engine.transliterate(args.word, beam_size=args.beam)
        print(f"Input       : {res['input']}")
        print(f"Top-1       : {res['top1']}")
        print(f"Candidates  : {res['candidates']}")
    elif args.sentence:
        out = engine.transliterate_sentence(args.sentence, beam_size=args.beam)
        print(f"Input   : {args.sentence}")
        print(f"Output  : {out}")
    else:
        interactive_cli(engine, beam_size=args.beam)


if __name__ == "__main__":
    main()
