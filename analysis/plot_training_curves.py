"""
analysis/plot_training_curves.py

Reads metrics.csv from both experiment out-dirs and plots:
  1. Train loss vs iterations
  2. Val loss vs iterations
  3. Val perplexity vs iterations
  4. Tokens/sec over time

Usage:
    python analysis/plot_training_curves.py \
        --standard out-tinystories-standard/metrics.csv \
        --looped   out-tinystories-looped/metrics.csv \
        --looped_ds out-tinystories-looped-ds/metrics.csv \
        --out      analysis/figures
"""

import argparse, os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

plt.style.use('seaborn-v0_8-paper')
COLORS = {'standard': '#2196F3', 'looped': '#FF5722', 'looped_ds': '#4CAF50'}
LABELS = {'standard': 'Standard (6 blocks)', 'looped': 'Looped (1×6)',
          'looped_ds': 'Looped+DeepSup (1×6)'}


def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.replace('', float('nan'))
    for c in ['train_loss','val_loss','val_ppl','lr','tokens_per_sec','mfu']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df


def savefig(fig, out_dir: str, name: str):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  saved → {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--standard',  default=None)
    parser.add_argument('--looped',    default=None)
    parser.add_argument('--looped_ds', default=None)
    parser.add_argument('--out',       default='analysis/figures')
    args = parser.parse_args()

    datasets = {}
    for key, path in [('standard',args.standard),
                      ('looped',args.looped),
                      ('looped_ds',args.looped_ds)]:
        if path and os.path.exists(path):
            datasets[key] = load(path)
            print(f"Loaded {key}: {len(datasets[key])} rows from {path}")

    if not datasets:
        print("No CSV files found. Run training first.")
        return

    # ── 1. Train loss ──────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8,5))
    for key, df in datasets.items():
        sub = df.dropna(subset=['train_loss'])
        ax.plot(sub['iter'], sub['train_loss'],
                color=COLORS[key], label=LABELS[key], alpha=0.85)
    ax.set_xlabel("Iteration"); ax.set_ylabel("Train Loss")
    ax.set_title("Training Loss Comparison"); ax.legend(); ax.grid(True, alpha=0.3)
    savefig(fig, args.out, 'train_loss.png'); plt.close(fig)

    # ── 2. Val loss ────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8,5))
    for key, df in datasets.items():
        sub = df.dropna(subset=['val_loss'])
        ax.plot(sub['iter'], sub['val_loss'],
                color=COLORS[key], label=LABELS[key], marker='o', ms=4)
    ax.set_xlabel("Iteration"); ax.set_ylabel("Validation Loss")
    ax.set_title("Validation Loss Comparison"); ax.legend(); ax.grid(True, alpha=0.3)
    savefig(fig, args.out, 'val_loss.png'); plt.close(fig)

    # ── 3. Val perplexity ─────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8,5))
    for key, df in datasets.items():
        sub = df.dropna(subset=['val_ppl'])
        ax.plot(sub['iter'], sub['val_ppl'],
                color=COLORS[key], label=LABELS[key], marker='s', ms=4)
    ax.set_xlabel("Iteration"); ax.set_ylabel("Validation Perplexity")
    ax.set_title("Validation Perplexity Comparison"); ax.legend(); ax.grid(True, alpha=0.3)
    savefig(fig, args.out, 'val_ppl.png'); plt.close(fig)

    # ── 4. Tokens/sec ─────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8,5))
    for key, df in datasets.items():
        sub = df.dropna(subset=['tokens_per_sec'])
        ax.plot(sub['iter'], sub['tokens_per_sec'],
                color=COLORS[key], label=LABELS[key], alpha=0.7)
    ax.set_xlabel("Iteration"); ax.set_ylabel("Tokens / sec")
    ax.set_title("Training Throughput"); ax.legend(); ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'{x:,.0f}'))
    savefig(fig, args.out, 'tokens_per_sec.png'); plt.close(fig)

    # ── 5. Val loss vs params (summary bar) ───────────────────────────────
    print("\nFinal validation losses:")
    for key, df in datasets.items():
        sub = df.dropna(subset=['val_loss'])
        if not sub.empty:
            print(f"  {key:12s}: val_loss={sub['val_loss'].iloc[-1]:.4f}  "
                  f"val_ppl={sub['val_ppl'].iloc[-1]:.2f}")

    print("\nDone. Figures saved to:", args.out)


if __name__ == '__main__':
    main()
