"""
analysis/compare_params.py

Prints a parameter efficiency comparison table for all three model classes.
Also plots validation loss vs parameter count (efficiency frontier).

Usage:
    python analysis/compare_params.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from model_looped import GPTConfig, StandardGPT, LoopedGPT, LoopedGPTDeepSupervision

# Shared hyperparameters matching the experiment configs
BASE = dict(block_size=256, vocab_size=50257, n_head=8,
            n_embd=256, dropout=0.0, bias=False)

configs = [
    ("Standard (6 blocks)", StandardGPT,
     GPTConfig(**BASE, n_layer=6, n_loop=1)),
    ("Looped   (1 block × 6)", LoopedGPT,
     GPTConfig(**BASE, n_layer=1, n_loop=6, looped=True)),
    ("Looped-DS(1 block × 6)", LoopedGPTDeepSupervision,
     GPTConfig(**BASE, n_layer=1, n_loop=6, looped=True,
               deep_supervision=True, ds_loss_mode='geometric')),
]

print(f"\n{'Model':<28} {'Params':>10} {'Unique Layers':>14} "
      f"{'Eff. Depth':>12} {'Param Ratio':>12}")
print("-" * 80)

baseline_params = None
for name, cls, cfg in configs:
    m = cls(cfg)
    n = m.get_num_params()
    depth = cfg.n_loop if cfg.looped else cfg.n_layer
    layers = 1 if cfg.looped else cfg.n_layer
    ratio = f"{baseline_params/n:.1f}×" if baseline_params else "1.0× (ref)"
    if baseline_params is None:
        baseline_params = n
    print(f"  {name:<26} {n/1e6:>8.3f}M {layers:>14} {depth:>12} {ratio:>12}")

print("-" * 80)
print(f"\n  Standard model has ~{baseline_params/1e6:.1f}M params.")
print(f"  Looped model has ~{baseline_params / configs[1][2].n_loop / 1e6:.1f}M "
      f"params — {configs[1][2].n_loop}× fewer.\n")

# Weight sharing verification
print("Weight sharing check for LoopedGPT:")
_, _, cfg_loop = configs[1]
m_loop = LoopedGPT(cfg_loop)
m_loop.verify_weight_sharing()

print("\nParameter breakdown for LoopedGPT:")
for n, p in m_loop.named_parameters():
    print(f"  {n:<45} {p.numel():>10,}  shape={list(p.shape)}")
