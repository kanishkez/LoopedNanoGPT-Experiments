"""
analysis/intermediate_predictions.py

Visualises how predictions are refined across loop steps for LoopedGPTDeepSupervision.

For a batch of validation examples:
  - Computes perplexity at each of the n_loop steps
  - Plots per-step perplexity (line plot)
  - Plots a heatmap: sequence position × loop step → token probability (final token)

Usage:
    python analysis/intermediate_predictions.py \
        --checkpoint out-tinystories-looped-ds/ckpt.pt \
        --data_dir   data/tinystories \
        --n_samples  8 \
        --out        analysis/figures
"""

import argparse, os, sys, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from model_looped import GPTConfig, LoopedGPTDeepSupervision

plt.style.use('seaborn-v0_8-paper')


def load_model(ckpt_path: str, device: str):
    ckpt   = torch.load(ckpt_path, map_location=device)
    args   = ckpt['model_args']
    config = GPTConfig(**args)
    model  = LoopedGPTDeepSupervision(config)
    state  = {k.replace('_orig_mod.',''):v for k,v in ckpt['model'].items()}
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    print(f"Loaded checkpoint: {ckpt_path}  (iter {ckpt['iter_num']})")
    return model, config


def get_samples(data_dir: str, block_size: int, n: int, device: str):
    val  = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    idxs = torch.randint(len(val)-block_size, (n,))
    x = torch.stack([torch.from_numpy(val[i:i+block_size].astype(np.int64)) for i in idxs])
    y = torch.stack([torch.from_numpy(val[i+1:i+1+block_size].astype(np.int64)) for i in idxs])
    return x.to(device), y.to(device)


def compute_step_ppl(model, x, y):
    """Return list of perplexities, one per loop step."""
    with torch.no_grad():
        model(x, y)   # populates model.last_loop_logits
    ppls = []
    for step_logits in model.last_loop_logits:
        loss = F.cross_entropy(
            step_logits.view(-1, step_logits.size(-1)),
            y.view(-1), ignore_index=-1
        )
        ppls.append(math.exp(loss.item()))
    return ppls


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data_dir',   default='data/tinystories')
    parser.add_argument('--n_samples',  type=int, default=16)
    parser.add_argument('--out',        default='analysis/figures')
    parser.add_argument('--device',     default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    model, config = load_model(args.checkpoint, args.device)

    x, y = get_samples(args.data_dir, config.block_size, args.n_samples, args.device)
    ppls = compute_step_ppl(model, x, y)

    n_loop = config.n_loop
    steps  = list(range(1, n_loop+1))

    # ── Plot 1: per-step perplexity ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7,4))
    ax.plot(steps, ppls, marker='o', color='#FF5722', linewidth=2)
    ax.set_xlabel("Loop Step"); ax.set_ylabel("Perplexity")
    ax.set_title("Prediction Refinement Across Loop Steps\n"
                 "(lower = better, each step reuses the same block)")
    ax.set_xticks(steps)
    ax.grid(True, alpha=0.3)
    for s, p in zip(steps, ppls):
        ax.annotate(f"{p:.1f}", (s,p), textcoords="offset points",
                    xytext=(0,6), ha='center', fontsize=9)
    path = os.path.join(args.out, 'step_perplexity.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f"Saved → {path}"); plt.close(fig)

    # ── Plot 2: heatmap token probability at each loop step (first sample) ─
    with torch.no_grad():
        model(x[:1], y[:1])

    n_show = min(32, config.block_size)   # show first 32 positions
    probs_matrix = np.zeros((n_loop, n_show))

    for step_idx, step_logits in enumerate(model.last_loop_logits):
        # step_logits shape: (1, T, vocab)
        log_probs = F.log_softmax(step_logits[0], dim=-1)   # (T, vocab)
        targets   = y[0]                                      # (T,)
        for pos in range(n_show):
            tok = targets[pos].item()
            probs_matrix[step_idx, pos] = log_probs[pos, tok].exp().item()

    fig, ax = plt.subplots(figsize=(14, 4))
    im = ax.imshow(probs_matrix, aspect='auto', cmap='YlOrRd',
                   vmin=0.0, vmax=probs_matrix.max())
    ax.set_xlabel("Sequence Position")
    ax.set_ylabel("Loop Step")
    ax.set_yticks(range(n_loop))
    ax.set_yticklabels([f"Step {i+1}" for i in range(n_loop)])
    ax.set_title("P(correct token | loop step, position)\n"
                 "Brighter = more confident at that step")
    plt.colorbar(im, ax=ax, label="P(correct token)")
    path = os.path.join(args.out, 'token_prob_heatmap.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f"Saved → {path}"); plt.close(fig)

    print("\nPer-step perplexities:")
    for s, p in zip(steps, ppls):
        bar = '█' * int(p / max(ppls) * 30)
        print(f"  Step {s}: {p:7.2f}  {bar}")


if __name__ == '__main__':
    main()
