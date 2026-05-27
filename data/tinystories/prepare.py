"""
data/tinystories/prepare.py

Downloads the TinyStories dataset from HuggingFace and tokenises it with
the GPT-2 BPE encoder (tiktoken).  Writes two memory-mapped binary files:
  data/tinystories/train.bin   (~2.1B uint16 tokens)
  data/tinystories/val.bin     (~22M  uint16 tokens)

Usage:
    python data/tinystories/prepare.py

Dependencies:
    pip install tiktoken datasets
"""

import os
import numpy as np
import tiktoken
from datasets import load_dataset

# ── configuration ──────────────────────────────────────────────────────────────
DATASET_NAME  = "roneneldan/TinyStories"
ENCODING_NAME = "gpt2"          # GPT-2 BPE, vocab_size = 50257
OUTPUT_DIR    = os.path.dirname(__file__)   # same directory as this script
BATCH_SIZE    = 1024            # rows processed per iteration (RAM friendly)
# ───────────────────────────────────────────────────────────────────────────────

def main():
    enc = tiktoken.get_encoding(ENCODING_NAME)
    print(f"Using tokenizer: {ENCODING_NAME}  (vocab_size={enc.n_vocab})")

    print(f"Downloading {DATASET_NAME} …")
    raw = load_dataset(DATASET_NAME, num_proc=4)
    # HuggingFace splits: "train" and "validation"
    splits = {
        "train": raw["train"],
        "val":   raw["validation"],
    }

    for split_name, dset in splits.items():
        print(f"\n=== {split_name} ({len(dset):,} stories) ===")

        # ── tokenise ──────────────────────────────────────────────────────────
        def tokenize(example):
            # encode_ordinary skips special tokens; we append EOT manually
            ids = enc.encode_ordinary(example["text"])
            ids.append(enc.eot_token)   # <|endoftext|> separates stories
            return {"ids": ids, "len": len(ids)}

        tokenized = dset.map(
            tokenize,
            remove_columns=["text"],
            num_proc=4,
            desc=f"Tokenising {split_name}",
        )

        total_tokens = sum(tokenized["len"])
        print(f"  Total tokens: {total_tokens:,}")

        # ── write memmap ──────────────────────────────────────────────────────
        out_path = os.path.join(OUTPUT_DIR, f"{split_name}.bin")
        arr = np.memmap(out_path, dtype=np.uint16, mode="w+", shape=(total_tokens,))
        idx = 0
        for batch in tokenized.iter(batch_size=BATCH_SIZE):
            for ids in batch["ids"]:
                arr[idx : idx + len(ids)] = ids
                idx += len(ids)
        arr.flush()
        print(f"  Written → {out_path}  ({os.path.getsize(out_path) / 1e9:.2f} GB)")

    print("\nDone.  Files ready for training.")
    print(f"  train.bin : {os.path.join(OUTPUT_DIR, 'train.bin')}")
    print(f"  val.bin   : {os.path.join(OUTPUT_DIR, 'val.bin')}")
    print(f"  vocab_size: {enc.n_vocab} (use this in GPTConfig)")


if __name__ == "__main__":
    main()
