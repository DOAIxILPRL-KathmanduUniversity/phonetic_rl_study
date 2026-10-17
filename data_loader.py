"""Character-Level Tokenizer & PyTorch DataLoader for Transliteration.

High-Performance In-Memory RAM Caching for Apple Silicon MPS, CUDA, and CPU Throughput.
"""

import json
import os
import sys
import time
from typing import List, Dict, Tuple, Optional, Union, Any
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

PAD_TOKEN = "<PAD>"
SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"
UNK_TOKEN = "<UNK>"

PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3


class CharVocab:
    def __init__(self, chars: Optional[List[str]] = None):
        self.special_tokens = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN]
        self.char2idx: Dict[str, int] = {}
        self.idx2char: Dict[int, str] = {}
        
        self.pad_idx = PAD_IDX
        self.sos_idx = SOS_IDX
        self.eos_idx = EOS_IDX
        self.unk_idx = UNK_IDX
        
        # Add special tokens
        for idx, tok in enumerate(self.special_tokens):
            self.char2idx[tok] = idx
            self.idx2char[idx] = tok
            
        if chars:
            for ch in sorted(list(set(chars))):
                if ch not in self.char2idx:
                    idx = len(self.char2idx)
                    self.char2idx[ch] = idx
                    self.idx2char[idx] = ch

    def __len__(self) -> int:
        return len(self.char2idx)

    def encode(self, text: str, add_sos: bool = True, add_eos: bool = True) -> List[int]:
        tokens = []
        if add_sos:
            tokens.append(SOS_IDX)
        for ch in text:
            tokens.append(self.char2idx.get(ch, UNK_IDX))
        if add_eos:
            tokens.append(EOS_IDX)
        return tokens

    def decode(self, token_ids: Union[List[int], torch.Tensor], remove_special: bool = True) -> str:
        chars = []
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
            
        for tid in token_ids:
            if remove_special and tid in (PAD_IDX, SOS_IDX, EOS_IDX):
                if tid == EOS_IDX:
                    break
                continue
            chars.append(self.idx2char.get(tid, ""))
        return "".join(chars)

    def decode_batch(self, tensor_or_list: Union[torch.Tensor, List[List[int]]], remove_special: bool = True) -> List[str]:
        if isinstance(tensor_or_list, torch.Tensor):
            tensor_or_list = tensor_or_list.tolist()
        return [self.decode(seq, remove_special=remove_special) for seq in tensor_or_list]

    def to_dict(self) -> Dict:
        return {"char2idx": self.char2idx, "idx2char": {str(k): v for k, v in self.idx2char.items()}}

    @classmethod
    def from_dict(cls, data: Dict) -> "CharVocab":
        vocab = cls()
        vocab.char2idx = data["char2idx"]
        vocab.idx2char = {int(k): v for k, v in data["idx2char"].items()}
        return vocab


def build_and_save_vocabs(
    train_csv: str = "data/train_split.csv",
    save_path: str = "data/vocab.json"
) -> Tuple[CharVocab, CharVocab]:
    """Scans dataset to construct and save source and target vocabularies."""
    df = pd.read_csv(train_csv)
    
    src_chars = set()
    tgt_chars = set()
    
    for _, row in df.iterrows():
        r = str(row["roman"]).strip().lower()
        n = str(row["nepali"]).strip()
        src_chars.update(r)
        tgt_chars.update(n)
        
    src_vocab = CharVocab(sorted(list(src_chars)))
    tgt_vocab = CharVocab(sorted(list(tgt_chars)))
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump({
            "src_vocab": src_vocab.to_dict(),
            "tgt_vocab": tgt_vocab.to_dict()
        }, f, ensure_ascii=False, indent=2)
        
    print(f"Saved vocabularies to {save_path}:")
    print(f"  Source (Roman) Vocab Size: {len(src_vocab)}")
    print(f"  Target (Devanagari) Vocab Size: {len(tgt_vocab)}")
    return src_vocab, tgt_vocab


def load_vocabs(vocab_path: str = "data/vocab.json") -> Tuple[CharVocab, CharVocab]:
    with open(vocab_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    src_vocab = CharVocab.from_dict(data["src_vocab"])
    tgt_vocab = CharVocab.from_dict(data["tgt_vocab"])
    return src_vocab, tgt_vocab


class TransliterationDataset(Dataset):
    """In-Memory RAM-Cached Dataset for High-Speed Neural Training."""
    def __init__(self, csv_file: str, src_vocab: CharVocab, tgt_vocab: CharVocab, max_len: int = 22, max_samples: Optional[int] = None):
        t0 = time.time()
        df = pd.read_csv(csv_file).dropna()
        if max_samples and len(df) > max_samples:
            df = df.sample(n=max_samples, random_state=42).reset_index(drop=True)
            
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_len = max_len
        
        # Fast in-RAM tensor pre-allocation
        src_texts = [str(x).strip().lower()[:self.max_len] for x in df["roman"]]
        tgt_texts = [str(x).strip()[:self.max_len] for x in df["nepali"]]
        
        self.samples = []
        for s_txt, t_txt in zip(src_texts, tgt_texts):
            s_ids = torch.tensor(self.src_vocab.encode(s_txt, add_sos=True, add_eos=True), dtype=torch.long)
            t_ids = torch.tensor(self.tgt_vocab.encode(t_txt, add_sos=True, add_eos=True), dtype=torch.long)
            self.samples.append({
                "src_ids": s_ids,
                "tgt_ids": t_ids,
                "src_text": s_txt,
                "tgt_text": t_txt
            })
            
        elapsed = time.time() - t0
        print(f"  ⚡ In-Memory Cached {len(self.samples)} pairs from {os.path.basename(csv_file)} in {elapsed:.2f}s")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]


def pad_collate_fn(batch, pad_idx: int = PAD_IDX) -> Dict[str, Any]:
    src_list = [item["src_ids"] for item in batch]
    tgt_list = [item["tgt_ids"] for item in batch]
    src_strs = [item["src_text"] for item in batch]
    tgt_strs = [item["tgt_text"] for item in batch]
    
    src_padded = pad_sequence(src_list, batch_first=True, padding_value=pad_idx)
    tgt_padded = pad_sequence(tgt_list, batch_first=True, padding_value=pad_idx)
    
    return {
        "src": src_padded,
        "tgt": tgt_padded,
        "src_str": src_strs,
        "tgt_str": tgt_strs
    }


def collate_fn(batch):
    """Backwards compatibility collate function returning tuple."""
    res = pad_collate_fn(batch)
    return res["src"], res["tgt"], res["src_str"], res["tgt_str"]


def get_dataloader(
    csv_file: str,
    src_vocab: CharVocab,
    tgt_vocab: CharVocab,
    batch_size: int = 256,
    shuffle: bool = True,
    max_len: int = 22,
    max_samples: Optional[int] = None,
    num_workers: int = 0
) -> DataLoader:
    dataset = TransliterationDataset(csv_file, src_vocab, tgt_vocab, max_len=max_len, max_samples=max_samples)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda b: pad_collate_fn(b, pad_idx=PAD_IDX),
        num_workers=num_workers
    )
