"""Dataset Download & Preparation Script for Romanized Nepali Transliteration.

Downloads parallel word pairs from HuggingFace (Saugatkafley/Nepali-Roman-Transliteration),
performs Unicode cleaning, removes null/duplicate entries, and generates train/val/test splits.

Usage:
    python data/prepare_data.py --train_size 120000 --output_dir data/
"""

import argparse
import os
import pandas as pd
from datasets import load_dataset


def prepare_dataset(train_size: int = 120000, output_dir: str = "data"):
    os.makedirs(output_dir, exist_ok=True)
    print(f"[*] Loading HuggingFace dataset 'Saugatkafley/Nepali-Roman-Transliteration'...")
    dataset = load_dataset("Saugatkafley/Nepali-Roman-Transliteration")
    print(f"[+] Loaded splits: {list(dataset.keys())}")

    # 1. Held-out Blind Test Split
    print("[*] Processing held-out validation split into blind test set...")
    val_raw = dataset["validation"]
    val_df = pd.DataFrame({
        "id": val_raw["unique_identifier"],
        "roman": [str(w).strip().lower() for w in val_raw["english word"]],
        "nepali": [str(w).strip() for w in val_raw["native word"]]
    })
    val_df = val_df.dropna().drop_duplicates(subset=["roman", "nepali"])
    val_df = val_df[(val_df["roman"].str.len() > 0) & (val_df["nepali"].str.len() > 0)]
    
    test_path = os.path.join(output_dir, "blind_test.csv")
    val_df.to_csv(test_path, index=False)
    print(f"[+] Saved blind test set ({len(val_df)} pairs) to: {test_path}")

    # 2. Training Split
    print(f"[*] Extracting {train_size} training examples...")
    train_raw = dataset["train"]
    total_available = len(train_raw)
    sample_size = min(train_size, total_available)
    train_subset = train_raw.shuffle(seed=42).select(range(sample_size))

    train_df = pd.DataFrame({
        "id": train_subset["unique_identifier"],
        "roman": [str(w).strip().lower() for w in train_subset["english word"]],
        "nepali": [str(w).strip() for w in train_subset["native word"]]
    })
    train_df = train_df.dropna().drop_duplicates(subset=["roman", "nepali"])
    train_df = train_df[(train_df["roman"].str.len() > 0) & (train_df["nepali"].str.len() > 0)]
    train_df = train_df[train_df["roman"].apply(lambda s: all(ord(c) < 128 for c in s))]

    # 80/20 train/val partition
    n_train = int(0.8 * len(train_df))
    df_train = train_df.iloc[:n_train].reset_index(drop=True)
    df_val = train_df.iloc[n_train:].reset_index(drop=True)

    train_path = os.path.join(output_dir, "train_split.csv")
    val_path = os.path.join(output_dir, "val_split.csv")
    df_train.to_csv(train_path, index=False)
    df_val.to_csv(val_path, index=False)

    print(f"[+] Saved training split ({len(df_train)} pairs) to: {train_path}")
    print(f"[+] Saved validation split ({len(df_val)} pairs) to: {val_path}")
    print("[✓] Dataset preparation completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Roman-to-Nepali transliteration dataset")
    parser.add_argument("--train_size", type=int, default=120000, help="Number of training samples to draw")
    parser.add_argument("--output_dir", type=str, default="data", help="Output directory for generated CSV files")
    args = parser.parse_args()

    prepare_dataset(train_size=args.train_size, output_dir=args.output_dir)
