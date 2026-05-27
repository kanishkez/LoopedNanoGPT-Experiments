"""
results/plot_tinystories.py

Local plotting script for TinyStories POC experiment results.
Run this on your Mac — no SSH needed.

Usage:
    cd "Loop Experiments"
    python3 results/plot_tinystories.py
"""
import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

# ── style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':      'sans-serif',
    'font.sans-serif':  ['SF Pro Display', 'Helvetica Neue', 'Arial'],
    'font.size':        11,
    'axes.spines.top':  False,
    'axes.spines.right':False,
    'axes.grid':        True,
    'grid.alpha':       0.25,
    'grid.linestyle':   '--',
    'figure.dpi':       150,
})

COLORS = {
    'standard':  '#2196F3',   # blue
    'looped':    '#FF5722',   # orange-red
    'looped_ds': '#4CAF50',   # green
}
LABELS = {
    'standard':  'Standard GPT (6 layers, ~0.31M params)',
    'looped':    'Looped GPT (1×6, ~0.07M block params)',
    'looped_ds': 'Looped GPT + Deep Supervision',
}

BASE = os.path.dirname(os.path.abspath(__file__))
CSV = {
    'standard':  os.path.join(BASE, 'tinystories', 'standard_metrics.csv'),
    'looped':    os.path.join(BASE, 'tinystories', 'looped_metrics.csv'),
    'looped_ds': os.path.join(BASE, 'tinystories', 'looped_ds_metrics.csv'),
}
OUT = os.path.join(BASE, 'tinystories', 'figures')
os.makedirs(OUT, exist_ok=True)

# ── load ──────────────────────────────────────────────────────────────────────
def load(path):
    df = pd.read_csv(path)
    df = df.replace('', float('nan'))
    for c in df.columns:
        if c != 'step_losses':
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df

dfs = {k: load(p) for k, p in CSV.items() if os.path.exists(p)}
print(f"Loaded: {list(dfs.keys())}")

def val_rows(df):
    return df.dropna(subset=['val_loss']).copy()

def train_rows(df):
    return df[df['train_loss'].notna() & df['val_loss'].isna()].copy()

def savefig(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"  → {p}")
    plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Val PPL vs Training Steps  +  Val Loss vs Tokens Seen
# ═══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for key, df in dfs.items():
    vr = val_rows(df)
    if vr.empty: continue
    c = COLORS[key]; lbl = LABELS[key]
    axes[0].plot(vr['iter'], vr['val_ppl'], color=c, lw=2, label=lbl, marker='o', ms=4)
    axes[1].plot(vr['tokens_seen']/1e6, vr['val_loss'], color=c, lw=2, label=lbl, marker='o', ms=4)

axes[0].set_xlabel("Training Iteration")
axes[0].set_ylabel("Validation Perplexity")
axes[0].set_title("Val PPL vs Training Steps\nTinyStories POC")
axes[0].legend(fontsize=8)

axes[1].set_xlabel("Tokens Seen (M)")
axes[1].set_ylabel("Validation Loss")
axes[1].set_title("Val Loss vs Tokens Seen\n(fair token-budget comparison)")
axes[1].legend(fontsize=8)

fig.tight_layout()
savefig(fig, '1_val_ppl_loss.png')

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Train Loss curves (smoothed)
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 5))

for key, df in dfs.items():
    tr = train_rows(df)
    if tr.empty: continue
    # smooth with rolling window
    smoothed = tr['train_loss'].rolling(window=20, min_periods=1).mean()
    ax.plot(tr['iter'], smoothed, color=COLORS[key], lw=1.5,
            label=LABELS[key], alpha=0.9)
    ax.plot(tr['iter'], tr['train_loss'], color=COLORS[key],
            lw=0.4, alpha=0.2)

ax.set_xlabel("Iteration")
ax.set_ylabel("Train Loss (CE)")
ax.set_title("Training Loss Curves — TinyStories\n(faint = raw, solid = 20-iter smoothed)")
ax.legend(fontsize=9)
fig.tight_layout()
savefig(fig, '2_train_loss.png')

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Tokens/sec throughput comparison
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 4))

for key, df in dfs.items():
    tr = df[df['tokens_per_sec'].notna() & (df['tokens_per_sec'] > 0)]
    if tr.empty: continue
    # Rolling mean, skip first 10 iters (warmup noise)
    tr = tr.iloc[10:]
    sm = tr['tokens_per_sec'].rolling(30, min_periods=1).mean()
    ax.plot(tr['iter'], sm/1e3, color=COLORS[key], lw=1.8,
            label=LABELS[key])

ax.set_xlabel("Iteration")
ax.set_ylabel("Throughput (K tok/s)")
ax.set_title("Training Throughput\n(Standard uses torch.compile; Looped does not)")
ax.legend(fontsize=9)
fig.tight_layout()
savefig(fig, '3_throughput.png')

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — Recurrence Diagnostics (Looped model)
# ═══════════════════════════════════════════════════════════════════════════════
for key in ['looped', 'looped_ds']:
    if key not in dfs: continue
    df   = dfs[key]
    diag = df.dropna(subset=['h_norm_mean'])
    if diag.empty:
        print(f"  No diagnostic rows for {key}")
        continue

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(diag['iter'], diag['h_norm_mean'], 'o-',
                 color=COLORS[key], lw=2, ms=5)
    axes[0].axhline(1.0, color='gray', linestyle='--', alpha=0.5, label='expected ≈ 1')
    axes[0].set_xlabel("Iteration"); axes[0].set_ylabel("Hidden State Norm / √d")
    axes[0].set_title(f"Hidden State Norm\n{LABELS[key]}")
    axes[0].legend(fontsize=8)

    axes[1].plot(diag['iter'], diag['h_delta_mean'], 's-',
                 color=COLORS[key], lw=2, ms=5, label='mean delta')
    if 'h_delta_min' in diag.columns:
        axes[1].fill_between(diag['iter'], diag['h_delta_min'],
                             diag['h_delta_mean'], alpha=0.2, color=COLORS[key])
        axes[1].plot(diag['iter'], diag['h_delta_min'], '--',
                     color=COLORS[key], lw=1, alpha=0.7, label='min delta')
    axes[1].axhline(1e-3, color='red', linestyle='--', alpha=0.6,
                    label='collapse (1e-3)')
    axes[1].set_xlabel("Iteration"); axes[1].set_ylabel("||h_i − h_{i−1}||")
    axes[1].set_title(f"Fixed-Point Collapse Detector\n{'No collapse detected ✓' if diag['h_delta_min'].min() > 1e-3 else '⚠️ Collapse detected'}")
    axes[1].legend(fontsize=8)
    axes[1].set_yscale('log')

    axes[2].plot(diag['iter'], diag['attn_ent_mean'], '^-',
                 color=COLORS[key], lw=2, ms=5)
    axes[2].set_xlabel("Iteration"); axes[2].set_ylabel("Attention Entropy (nats)")
    axes[2].set_title(f"Attention Entropy\n(→0 = attention collapse)")

    fig.suptitle(f"Recurrence Health Diagnostics — {key}", fontweight='bold')
    fig.tight_layout()
    savefig(fig, f'4_diagnostics_{key}.png')

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 5 — Parameter Efficiency: Val PPL vs Non-Embedding Params
# ═══════════════════════════════════════════════════════════════════════════════
# Final val PPL + approx non-embedding params for each model
PARAMS = {
    'standard':  312_128,   # 6 blocks × ~52k each
    'looped':     65_728,   # 1 block
    'looped_ds':  65_728,
}

fig, ax = plt.subplots(figsize=(8, 5))
for key, df in dfs.items():
    vr = val_rows(df)
    if vr.empty: continue
    final_ppl = vr['val_ppl'].dropna().iloc[-1]
    p = PARAMS[key] / 1e3  # in K
    ax.scatter(p, final_ppl, color=COLORS[key], s=150, zorder=5,
               label=f"{LABELS[key]}\nPPL={final_ppl:.2f}")
    ax.annotate(f"PPL={final_ppl:.2f}", (p, final_ppl),
                textcoords="offset points", xytext=(8, 4),
                fontsize=8, color=COLORS[key])

ax.set_xlabel("Non-Embedding Block Parameters (K)")
ax.set_ylabel("Final Validation Perplexity")
ax.set_title("Parameter Efficiency\nFinal Val PPL vs Non-Embedding Params")
ax.legend(fontsize=8, loc='upper right')
ax.set_xscale('log')
fig.tight_layout()
savefig(fig, '5_param_efficiency.png')

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 6 — Deep Supervision: Per-step losses over training
# ═══════════════════════════════════════════════════════════════════════════════
if 'looped_ds' in dfs:
    df = dfs['looped_ds']
    step_df = df.dropna(subset=['step_losses'])

    parsed = []
    for _, row in step_df.iterrows():
        try:
            vals = [float(v) for v in str(row['step_losses']).split('|')]
            if len(vals) == 6:
                parsed.append((row['iter'], vals))
        except Exception:
            continue

    if parsed:
        iters    = [p[0] for p in parsed]
        step_mat = np.array([p[1] for p in parsed])

        fig, ax = plt.subplots(figsize=(10, 5))
        step_colors = plt.cm.plasma(np.linspace(0.1, 0.9, 6))
        for i in range(6):
            ax.plot(iters, step_mat[:, i], color=step_colors[i],
                    lw=1.5, label=f'Loop step {i+1}', alpha=0.85)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Cross-Entropy Loss")
        ax.set_title("Per-Loop-Step CE Loss — Deep Supervision Model\n"
                     "All steps converging = healthy gradient flow")
        ax.legend(ncol=2, fontsize=8)
        fig.tight_layout()
        savefig(fig, '6_ds_step_losses.png')

        # Also: gap between step 1 and step 6
        fig, ax = plt.subplots(figsize=(8, 4))
        gap = step_mat[:, -1] - step_mat[:, 0]
        ax.plot(iters, gap, color='#9C27B0', lw=2)
        ax.axhline(0, color='gray', linestyle='--', alpha=0.4)
        ax.fill_between(iters, gap, 0, where=(gap < 0),
                        alpha=0.15, color='green', label='step 6 better than step 1')
        ax.fill_between(iters, gap, 0, where=(gap > 0),
                        alpha=0.15, color='red', label='step 1 better (unexpected)')
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Loss(step 6) − Loss(step 1)")
        ax.set_title("Iterative Refinement: Is Loop 6 Better Than Loop 1?")
        ax.legend(fontsize=8)
        fig.tight_layout()
        savefig(fig, '6b_ds_refinement_gap.png')

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 7 — Summary comparison at matched token budgets
# ═══════════════════════════════════════════════════════════════════════════════
TOKEN_CHECKPOINTS = [8.2, 16.4, 24.6, 32.8, 41.0, 49.2, 57.3, 65.5, 73.7, 81.9]

fig, ax = plt.subplots(figsize=(10, 5))

for key, df in dfs.items():
    vr = val_rows(df).dropna(subset=['tokens_seen', 'val_ppl'])
    if vr.empty: continue
    vr = vr.sort_values('tokens_seen')
    ax.plot(vr['tokens_seen']/1e6, vr['val_ppl'],
            color=COLORS[key], lw=2.5, marker='o', ms=5,
            label=LABELS[key])

# Annotate final points
for key, df in dfs.items():
    vr = val_rows(df).dropna(subset=['tokens_seen', 'val_ppl'])
    if vr.empty: continue
    last = vr.sort_values('tokens_seen').iloc[-1]
    ax.annotate(f"  {last['val_ppl']:.2f}",
                (last['tokens_seen']/1e6, last['val_ppl']),
                fontsize=8, color=COLORS[key], va='center')

ax.set_xlabel("Tokens Seen (M)  ← fair comparison axis")
ax.set_ylabel("Validation Perplexity")
ax.set_title("Val PPL vs Tokens Seen — All Models\nTinyStories POC (vocab_size=50257, n_embd=256, depth=6)")
ax.legend(fontsize=8)
fig.tight_layout()
savefig(fig, '7_all_models_token_budget.png')

print(f"\n✓ All figures saved to: {OUT}")
print("\nFinal Results Summary:")
print(f"{'Model':<30} {'Final Val PPL':>14} {'Block Params':>14} {'Tok/s':>10}")
print("-"*70)
for key, df in dfs.items():
    vr = val_rows(df).dropna(subset=['val_ppl'])
    if vr.empty: continue
    ppl = vr['val_ppl'].iloc[-1]
    tr  = df[df['tokens_per_sec'].notna() & (df['tokens_per_sec'] > 100)]
    tps = tr['tokens_per_sec'].median() if not tr.empty else 0
    print(f"{LABELS[key][:30]:<30} {ppl:>14.2f} {PARAMS[key]:>14,} {tps:>10,.0f}")
