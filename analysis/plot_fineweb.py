"""
results/plot_fineweb.py — Local plots for the FineWeb scaling experiment.
Run on your Mac: python3 results/plot_fineweb.py
"""
import os, math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Helvetica Neue', 'Arial'],
    'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.alpha': 0.25, 'grid.linestyle': '--', 'figure.dpi': 150,
})

COLORS  = {'standard': '#2196F3', 'looped': '#FF5722'}
LABELS  = {'standard': 'Standard GPT (12L, ~56M block params)',
           'looped':   'Looped GPT (1x12, ~4.7M block params)'}
PARAMS  = {'standard': 56_000_000, 'looped': 4_700_000}

BASE = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.join(BASE, 'fineweb', 'figures')
os.makedirs(OUT, exist_ok=True)

def load(path):
    df = pd.read_csv(path).replace('', float('nan'))
    for c in df.columns:
        if c != 'step_losses':
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df

dfs = {}
for k in ['standard', 'looped']:
    p = os.path.join(BASE, 'fineweb', f'{k}_metrics.csv')
    if os.path.exists(p):
        dfs[k] = load(p)
        print(f"Loaded {k}: {len(dfs[k])} rows")

def val_rows(df):  return df.dropna(subset=['val_loss']).copy()
def train_rows(df): return df[df['train_loss'].notna() & df['val_loss'].isna()].copy()

def savefig(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"  -> {p}")
    plt.close(fig)

# ── 1. Val PPL vs Tokens ────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for k, df in dfs.items():
    vr = val_rows(df).dropna(subset=['tokens_seen','val_ppl'])
    axes[0].plot(vr['tokens_seen']/1e6, vr['val_ppl'],
                 color=COLORS[k], lw=2.5, marker='o', ms=5, label=LABELS[k])
    axes[1].plot(vr['iter'], vr['val_ppl'],
                 color=COLORS[k], lw=2.5, marker='o', ms=5, label=LABELS[k])

for ax, xl in zip(axes, ['Tokens Seen (M)', 'Training Iteration']):
    ax.set_xlabel(xl); ax.set_ylabel('Validation Perplexity')
    ax.legend(fontsize=8)

axes[0].set_title('Val PPL vs Tokens\n(fair compute comparison)')
axes[1].set_title('Val PPL vs Iteration')
fig.suptitle('FineWeb Scaling Experiment — Standard vs Looped GPT', fontweight='bold')
fig.tight_layout()
savefig(fig, '1_val_ppl.png')

# ── 2. Train Loss (smoothed) ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
for k, df in dfs.items():
    tr = train_rows(df)
    sm = tr['train_loss'].rolling(30, min_periods=1).mean()
    ax.plot(tr['tokens_seen']/1e6, sm, color=COLORS[k], lw=2, label=LABELS[k])
    ax.plot(tr['tokens_seen']/1e6, tr['train_loss'],
            color=COLORS[k], lw=0.4, alpha=0.15)
ax.set_xlabel('Tokens Seen (M)'); ax.set_ylabel('Train Loss (CE)')
ax.set_title('Training Loss — FineWeb\n(faint=raw, solid=30-step smoothed)')
ax.legend(fontsize=9); fig.tight_layout()
savefig(fig, '2_train_loss.png')

# ── 3. PPL gap over training ─────────────────────────────────────────────────
std_vr = val_rows(dfs.get('standard', pd.DataFrame())).dropna(subset=['tokens_seen','val_ppl']).set_index('iter')
lp_vr  = val_rows(dfs.get('looped',   pd.DataFrame())).dropna(subset=['tokens_seen','val_ppl']).set_index('iter')
common  = std_vr.index.intersection(lp_vr.index)

if len(common) > 1:
    gap = lp_vr.loc[common, 'val_ppl'].values - std_vr.loc[common, 'val_ppl'].values
    toks = std_vr.loc[common, 'tokens_seen'].values / 1e6
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(toks, gap, 'o-', color='#9C27B0', lw=2.5, ms=6)
    ax.fill_between(toks, 0, gap, alpha=0.12, color='#9C27B0')
    ax.axhline(0, color='gray', linestyle='--', alpha=0.4)
    ax.set_xlabel('Tokens Seen (M)'); ax.set_ylabel('PPL gap (Looped - Standard)')
    ax.set_title('Parameter Efficiency Gap Over Training\nNarrowing gap = looped catching up')
    fig.tight_layout(); savefig(fig, '3_ppl_gap.png')

# ── 4. Parameter efficiency scatter ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
for k, df in dfs.items():
    vr = val_rows(df).dropna(subset=['val_ppl'])
    ppl = vr['val_ppl'].iloc[-1]
    p   = PARAMS[k] / 1e6
    ax.scatter(p, ppl, color=COLORS[k], s=200, zorder=5)
    ax.annotate(f"  {LABELS[k].split('(')[0].strip()}\n  PPL={ppl:.1f}",
                (p, ppl), fontsize=8, color=COLORS[k], va='center')
ax.set_xscale('log')
ax.set_xlabel('Block Parameters (M, log scale)')
ax.set_ylabel('Final Validation PPL')
ax.set_title('Parameter Efficiency — FineWeb\n12x fewer block params, competitive PPL')
fig.tight_layout(); savefig(fig, '4_param_efficiency.png')

# ── 5. Recurrence diagnostics over training ──────────────────────────────────
if 'looped' in dfs:
    df   = dfs['looped']
    diag = df.dropna(subset=['h_norm_mean'])
    if not diag.empty:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        axes[0].plot(diag['iter'], diag['h_norm_mean'], 'o-', color=COLORS['looped'], lw=2, ms=5)
        axes[0].set_title('Hidden State Norm / sqrt(d)'); axes[0].set_xlabel('Iteration')
        axes[1].plot(diag['iter'], diag['h_delta_mean'], 's-', color=COLORS['looped'], lw=2, ms=5, label='mean')
        axes[1].plot(diag['iter'], diag['h_delta_min'],  '--', color=COLORS['looped'], lw=1.5, ms=4, alpha=0.7, label='min')
        axes[1].axhline(1e-3, color='red', linestyle=':', alpha=0.5, label='collapse threshold')
        axes[1].set_yscale('log'); axes[1].legend(fontsize=8)
        axes[1].set_title('h_delta (Fixed-Point Collapse Detector)\nGROWING = actively using recurrence')
        axes[1].set_xlabel('Iteration')
        axes[2].plot(diag['iter'], diag['attn_ent_mean'], '^-', color=COLORS['looped'], lw=2, ms=5)
        axes[2].set_title('Attention Entropy (nats)\nDecreasing = learning to focus')
        axes[2].set_xlabel('Iteration')
        fig.suptitle('Looped GPT Recurrence Health — FineWeb', fontweight='bold')
        fig.tight_layout(); savefig(fig, '5_diagnostics.png')

# ── 6. Per-loop delta profile at final checkpoint ────────────────────────────
# From the last DIAG log line parsed manually:
final_deltas = [14.93, 12.75, 5.59, 4.81, 3.69, 3.54, 3.60, 3.90, 4.50, 5.62, 7.23, 8.72]
final_norms  = [0.541, 0.505, 0.422, 0.387, 0.407, 0.452, 0.510, 0.581, 0.666, 0.777, 0.934, 1.147]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
x = list(range(1, 13))
axes[0].bar(x, final_deltas, color=COLORS['looped'], alpha=0.8, edgecolor='white')
axes[0].axhline(1e-3, color='red', linestyle='--', alpha=0.4, label='collapse threshold')
axes[0].set_xlabel('Loop Step'); axes[0].set_ylabel('||h_i - h_{i-1}||')
axes[0].set_title('Per-Loop Hidden State Delta\n(Final checkpoint, iter 8000)\nU-shape = early+late loops most active')
axes[0].legend(fontsize=8)

axes[1].plot(x, final_norms, 'o-', color='#4CAF50', lw=2.5, ms=7)
axes[1].set_xlabel('Loop Step'); axes[1].set_ylabel('Hidden State Norm / sqrt(d)')
axes[1].set_title('Per-Loop Hidden State Norm\n(Final checkpoint)\nGrowing norm = representations expanding')

fig.suptitle('Looped GPT — Per-Loop Analysis at End of Training', fontweight='bold')
fig.tight_layout(); savefig(fig, '6_per_loop_profile.png')

# ── Summary ──────────────────────────────────────────────────────────────────
print(f"\nFinal Results Summary:")
print(f"{'Model':<30} {'Val PPL':>10} {'Block Params':>14}")
print("-"*56)
for k, df in dfs.items():
    vr = val_rows(df).dropna(subset=['val_ppl'])
    if not vr.empty:
        print(f"{LABELS[k][:30]:<30} {vr['val_ppl'].iloc[-1]:>10.2f} {PARAMS[k]:>14,}")
print(f"\nAll figures -> {OUT}")
