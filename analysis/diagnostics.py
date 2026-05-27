"""
analysis/diagnostics.py

Plots the recurrence health diagnostics logged to metrics.csv:
  1. Hidden-state norm per loop across training
  2. Hidden-state delta per loop (fixed-point collapse detector)
  3. Attention entropy per loop (attention collapse detector)
  4. Gradient norm over training (instability detector)
  5. Per-step losses over training (gradient starvation detector)

These answer the most critical scientific question:
  "Is later-loop computation meaningful, or has the recurrence collapsed?"

Usage:
    python analysis/diagnostics.py \
        --looped   out-tinystories-looped/metrics.csv \
        --looped_ds out-tinystories-looped-ds/metrics.csv \
        --out      analysis/figures
"""

import argparse, os, sys, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

plt.style.use('seaborn-v0_8-paper')
COLORS = ['#2196F3','#FF5722','#4CAF50','#9C27B0','#FF9800','#00BCD4']


def load(path):
    df = pd.read_csv(path)
    df = df.replace('', float('nan'))
    for c in df.columns:
        if c not in ('step_losses',):
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df


def savefig(fig, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, name)
    fig.savefig(p, dpi=150, bbox_inches='tight')
    print(f"  saved → {p}")
    plt.close(fig)


def plot_grad_norm(dfs: dict, out_dir: str):
    """Gradient norm over training — detect instability."""
    fig, ax = plt.subplots(figsize=(9,4))
    for (label, df), color in zip(dfs.items(), COLORS):
        sub = df.dropna(subset=['grad_norm'])
        if sub.empty: continue
        ax.semilogy(sub['iter'], sub['grad_norm'], color=color,
                    label=label, alpha=0.7, linewidth=0.8)
    ax.set_xlabel("Iteration"); ax.set_ylabel("Gradient Norm (log scale)")
    ax.set_title("Gradient Norm Over Training\n"
                 "(spikes > 1.0 → instability; clip threshold = 1.0)")
    ax.axhline(1.0, color='red', linestyle='--', alpha=0.5, label='clip threshold')
    ax.legend(); ax.grid(True, alpha=0.3)
    savefig(fig, out_dir, 'grad_norm.png')


def plot_hidden_state_health(df: pd.DataFrame, label: str, out_dir: str):
    """
    Plot hidden-state norm and delta per training checkpoint.
    These are scalar summaries logged at diag_interval.

    Fixed-point collapse signature:
      h_delta_min → 0 at an early loop step
      → later loops doing no meaningful computation
    """
    diag = df.dropna(subset=['h_norm_mean'])
    if diag.empty:
        print(f"  No diagnostic rows found in {label} CSV. "
              f"Set diag_interval > 0 in training config.")
        return

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    # Hidden state norm
    axes[0].plot(diag['iter'], diag['h_norm_mean'], color='#2196F3',
                 marker='o', ms=4, label='h_norm_mean')
    axes[0].set_ylabel("Hidden State Norm / sqrt(d)")
    axes[0].set_title(f"Recurrence Health — {label}\n"
                      "norm: explosion>10, collapse<0.1")
    axes[0].axhline(1.0, color='gray', linestyle='--', alpha=0.4, label='expected ≈ 1')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    # Hidden state delta
    axes[1].plot(diag['iter'], diag['h_delta_mean'], color='#FF5722',
                 marker='s', ms=4, label='h_delta_mean')
    if 'h_delta_min' in diag.columns:
        axes[1].fill_between(diag['iter'], diag['h_delta_min'],
                             diag['h_delta_mean'], alpha=0.2,
                             color='#FF5722', label='min–mean range')
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("||h_i − h_{i-1}|| mean")
    axes[1].set_title("Fixed-Point Collapse Detector\n"
                      "delta→0 early ⇒ later loops doing nothing")
    axes[1].axhline(1e-3, color='red', linestyle='--', alpha=0.5,
                    label='collapse threshold (1e-3)')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    savefig(fig, out_dir, f'hidden_state_health_{label.replace(" ","_")}.png')


def plot_attn_entropy(df: pd.DataFrame, label: str, out_dir: str):
    """Attention entropy — detect attention collapse (head focusing on single token)."""
    diag = df.dropna(subset=['attn_ent_mean'])
    if diag.empty: return
    fig, ax = plt.subplots(figsize=(9,4))
    ax.plot(diag['iter'], diag['attn_ent_mean'], color='#9C27B0',
            marker='o', ms=4)
    ax.set_xlabel("Iteration"); ax.set_ylabel("Mean Attention Entropy (nats)")
    ax.set_title(f"Attention Entropy — {label}\n"
                 "entropy→0 ⇒ attention collapse (heads spiking to single token)")
    ax.grid(True, alpha=0.3)
    savefig(fig, out_dir, f'attn_entropy_{label.replace(" ","_")}.png')


def plot_step_losses(df: pd.DataFrame, label: str, n_loop: int, out_dir: str):
    """
    Plot per-loop-step CE losses over training.
    Tests for:
      Gradient starvation: loss[0] >> loss[-1] and barely decreasing
      Useful refinement:   all losses decreasing, with gap between loop 1 and loop N
    """
    step_df = df.dropna(subset=['step_losses'])
    if step_df.empty:
        print(f"  No step_losses data in {label}. Only available for looped_ds model.")
        return

    # Parse the pipe-separated step loss strings
    parsed = []
    for row in step_df.itertuples():
        try:
            vals = [float(v) for v in str(row.step_losses).split('|')]
            if len(vals) == n_loop:
                parsed.append((row.iter, vals))
        except Exception:
            continue

    if not parsed:
        print(f"  Could not parse step_losses for {label}.")
        return

    iters      = [p[0] for p in parsed]
    step_mat   = np.array([p[1] for p in parsed])   # shape (T, n_loop)

    fig, ax = plt.subplots(figsize=(10,5))
    for i in range(n_loop):
        ax.plot(iters, step_mat[:, i], color=COLORS[i % len(COLORS)],
                label=f'Loop step {i+1}', alpha=0.85)
    ax.set_xlabel("Iteration"); ax.set_ylabel("Cross-Entropy Loss")
    ax.set_title(f"Per-Loop-Step CE Loss — {label}\n"
                 "All steps should decrease; early steps should converge to later steps")
    ax.legend(ncol=2); ax.grid(True, alpha=0.3)
    savefig(fig, out_dir, f'step_losses_{label.replace(" ","_")}.png')

    # Also print final convergence summary
    final = step_mat[-1]
    print(f"\n  [{label}] Final per-step losses (loop 1 → loop {n_loop}):")
    for i, v in enumerate(final):
        delta = final[0] - v
        bar   = '▓' * int(delta / max(delta+1e-8, 1.0) * 20)
        print(f"    Loop {i+1}: {v:.4f}  improvement over step 1: {delta:+.4f}  {bar}")


def print_collapse_summary(df: pd.DataFrame, label: str):
    """Print a summary table of fixed-point collapse risk."""
    diag = df.dropna(subset=['h_delta_min'])
    if diag.empty: return
    last = diag.iloc[-1]
    print(f"\n  [{label}] Final recurrence health (last diagnostic checkpoint):")
    print(f"    h_norm_mean  = {last.get('h_norm_mean', 'N/A'):.4f}  "
          f"(healthy: 0.5 – 3.0)")
    print(f"    h_delta_mean = {last.get('h_delta_mean','N/A'):.5f}  "
          f"(collapse threshold: < 0.001)")
    print(f"    h_delta_min  = {last.get('h_delta_min','N/A'):.5f}  "
          f"← most sensitive collapse indicator")
    print(f"    attn_ent     = {last.get('attn_ent_mean','N/A'):.4f}  "
          f"(collapse: < 0.5 nats)")
    if last.get('h_delta_min', 1.0) < 1e-3:
        print(f"    ⚠️  FIXED-POINT COLLAPSE LIKELY")
    else:
        print(f"    ✓  No collapse detected")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--looped',    default=None)
    parser.add_argument('--looped_ds', default=None)
    parser.add_argument('--standard',  default=None)
    parser.add_argument('--n_loop',    type=int, default=6)
    parser.add_argument('--out',       default='analysis/figures')
    args = parser.parse_args()

    dfs = {}
    for key, path in [('standard',args.standard),
                      ('looped',args.looped),
                      ('looped_ds',args.looped_ds)]:
        if path and os.path.exists(path):
            dfs[key] = load(path)
            print(f"Loaded {key}: {path}")

    if not dfs:
        print("No CSV files found. Run training first.")
        return

    # Gradient norm — all models
    plot_grad_norm(dfs, args.out)

    # Per-model recurrence health
    for key, df in dfs.items():
        if key == 'standard': continue
        plot_hidden_state_health(df, key, args.out)
        plot_attn_entropy(df, key, args.out)
        print_collapse_summary(df, key)

    # Per-step losses — looped_ds only
    if 'looped_ds' in dfs:
        plot_step_losses(dfs['looped_ds'], 'looped_ds', args.n_loop, args.out)

    print(f"\nDone. Diagnostic figures → {args.out}")


if __name__ == '__main__':
    main()
