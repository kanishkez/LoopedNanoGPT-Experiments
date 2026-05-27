"""
data/fineweb/prepare.py

Streams FineWeb-Edu (sample-10BT split) and writes the first ~100M tokens
to train.bin + val.bin using GPT-2 BPE via tiktoken.

Streaming avoids downloading the full 100GB dataset.
Target: 100M train tokens + 2M val tokens.

Usage:
    python3 data/fineweb/prepare.py
"""
import os, sys, struct, time
import numpy as np
import tiktoken
from datasets import load_dataset

# ── config ────────────────────────────────────────────────────────────────────
OUT_DIR         = os.path.dirname(os.path.abspath(__file__))
TRAIN_TOKENS    = 100_000_000   # 100M train tokens
VAL_TOKENS      =   2_000_000   # 2M val tokens
DATASET_NAME    = "HuggingFaceFW/fineweb-edu"
DATASET_SPLIT   = "train"       # FineWeb only has train; we carve val ourselves
DATASET_CONFIG  = "sample-10BT"
TEXT_FIELD      = "text"
# ──────────────────────────────────────────────────────────────────────────────

enc = tiktoken.get_encoding("gpt2")
EOT = enc.encode_single_token("<|endoftext|>")   # document separator

def tokenise(text: str) -> list[int]:
    return enc.encode_ordinary(text)

def write_bin(path: str, tokens: list[int]):
    arr = np.array(tokens, dtype=np.uint16)
    arr.tofile(path)
    gb = arr.nbytes / 1e9
    print(f"  Written → {path}  ({len(tokens):,} tokens, {gb:.3f} GB)")

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"Streaming {DATASET_NAME} ({DATASET_CONFIG}) …")
    ds = load_dataset(DATASET_NAME, name=DATASET_CONFIG,
                      split=DATASET_SPLIT, streaming=True,
                      trust_remote_code=True)

    train_toks, val_toks = [], []
    n_train = n_val = 0
    t0 = time.time()

    for i, sample in enumerate(ds):
        toks = tokenise(sample[TEXT_FIELD]) + [EOT]

        if n_val < VAL_TOKENS:
            val_toks.extend(toks)
            n_val += len(toks)
        elif n_train < TRAIN_TOKENS:
            train_toks.extend(toks)
            n_train += len(toks)
        else:
            break

        if i % 5000 == 0:
            elapsed = time.time() - t0
            print(f"  doc {i:>8,} | train {n_train/1e6:.1f}M | "
                  f"val {n_val/1e6:.1f}M | {elapsed:.0f}s")

    print(f"\nTokenisation complete in {time.time()-t0:.0f}s")
    write_bin(os.path.join(OUT_DIR, "train.bin"), train_toks)
    write_bin(os.path.join(OUT_DIR, "val.bin"),   val_toks)
    print(f"\nvocab_size: 50257  (GPT-2 BPE)")
    print("Done.")

if __name__ == "__main__":
    main()
