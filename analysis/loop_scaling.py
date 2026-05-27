"""
analysis/loop_scaling.py

CRITICAL EVALUATION: Test-time loop scaling.

Loads a trained LoopedGPT checkpoint and evaluates val PPL
at increasing inference-time loop counts beyond the training setting.

Tests whether iterative recurrent compute scales at inference time:
  train loops=12 → eval at 12, 16, 20, 24, 32

Usage:
    python3 analysis/loop_scaling.py \
        --ckpt out-fineweb-looped/ckpt.pt \
        --data data/fineweb/val.bin \
        --loops 12 16 20 24 32 \
        --out   analysis/figures
"""
import argparse, math, os, sys, time
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model_looped import LoopedGPT, GPTConfig

plt.style.use('seaborn-v0_8-paper')


def load_model(ckpt_path: str, device: str):
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    args = ckpt['model_args']
    cfg  = GPTConfig(**{k: v for k, v in args.items()
                        if hasattr(GPTConfig, k)})
    model = LoopedGPT(cfg)
    # Strip _orig_mod prefix if saved from compiled model
    sd = {k.replace('_orig_mod.', ''): v
          for k, v in ckpt['model'].items()}
    model.load_state_dict(sd)
    model.to(device)
    model.eval()

    train_loops = cfg.n_loop
    print(f"  Trained with n_loop={train_loops}, n_embd={cfg.n_embd}")
    print(f"  Val loss at save: {ckpt.get('best_val_loss', 'N/A')}")
    return model, cfg, train_loops


def get_batch(data: np.ndarray, block_size: int,
              batch_size: int, device: str):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x  = torch.stack([torch.from_numpy(data[i:i+block_size].astype(np.int64))
                      for i in ix]).to(device)
    y  = torch.stack([torch.from_numpy(data[i+1:i+1+block_size].astype(np.int64))
                      for i in ix]).to(device)
    return x, y


@torch.no_grad()
def evaluate(model: LoopedGPT, data: np.ndarray, n_loop_override: int,
             block_size: int, batch_size: int, n_batches: int,
             device: str, dtype):
    """Evaluate with a different n_loop than what the model was trained with."""
    original_n_loop = model.config.n_loop
    model.config.n_loop = n_loop_override   # temporarily override

    losses, all_diag = [], []
    for _ in range(n_batches):
        x, y = get_batch(data, block_size, batch_size, device)
        with torch.amp.autocast(device_type='cuda', dtype=dtype):
            logits, loss, diag = model(x, y, diagnostics=True)
        losses.append(loss.item())
        all_diag.append(diag)

    model.config.n_loop = original_n_loop   # restore

    mean_loss = float(np.mean(losses))
    ppl       = math.exp(mean_loss)

    # Aggregate diagnostics across batches
    n = n_loop_override
    h_norms  = [np.mean([d['h_norms'][i]   for d in all_diag]) for i in range(n)]
    h_deltas = [np.mean([d['h_deltas'][i]  for d in all_diag]) for i in range(n)]
    attn_ent = [np.mean([d['attn_ents'][i] for d in all_diag]) for i in range(n)]

    # Cosine similarity between successive hidden states can't be computed
    # from diagnostics alone (only scalars are stored); we compute it separately
    return {'loss': mean_loss, 'ppl': ppl,
            'h_norms': h_norms, 'h_deltas': h_deltas, 'attn_ent': attn_ent}


@torch.no_grad()
def compute_kl_cosine(model: LoopedGPT, data: np.ndarray,
                      n_loop: int, block_size: int,
                      batch_size: int, device: str, dtype):
    """
    Run a single forward pass collecting full hidden states per loop to
    compute:
      - cosine similarity cos(h_i, h_{i+1})
      - KL divergence KL(p_i || p_{i+1}) where p_i = softmax(lm_head(h_i))
    """
    model.config.n_loop = n_loop
    x, y = get_batch(data, block_size, batch_size, device)

    # Monkey-patch forward to collect per-loop hidden states and logits
    hidden_states = []
    loop_logits   = []

    original_forward = model.transformer.h.forward

    def hooked_block(inp, **kw):
        out = original_forward(inp, **kw)
        hidden_states.append(out.detach().cpu() if not isinstance(out, tuple)
                              else out[0].detach().cpu())
        # get logits from this hidden state
        with torch.no_grad():
            xn = model.transformer.ln_f(
                out if not isinstance(out, tuple) else out[0])
            lgs = model.lm_head(xn)
            loop_logits.append(lgs.detach().cpu())
        return out

    model.transformer.h.forward = hooked_block

    with torch.amp.autocast(device_type='cuda', dtype=dtype):
        model(x, y)

    model.transformer.h.forward = original_forward  # restore

    # Compute cosine similarity between successive hidden states (mean over B, T)
    cos_sims = []
    for i in range(len(hidden_states) - 1):
        h1 = hidden_states[i].float()    # (B, T, C)
        h2 = hidden_states[i+1].float()
        cos = torch.nn.functional.cosine_similarity(h1, h2, dim=-1).mean().item()
        cos_sims.append(cos)

    # KL(p_i || p_{i+1})
    kl_divs = []
    for i in range(len(loop_logits) - 1):
        p = torch.softmax(loop_logits[i].float(), dim=-1)
        q = torch.softmax(loop_logits[i+1].float(), dim=-1)
        # KL(p||q) = sum p * log(p/q)
        kl = (p * (p.clamp(1e-9).log() - q.clamp(1e-9).log())).sum(dim=-1).mean().item()
        kl_divs.append(kl)

    return cos_sims, kl_divs


def plot_results(results: dict, out_dir: str, train_loops: int):
    os.makedirs(out_dir, exist_ok=True)
    loop_counts = sorted(results.keys())
    ppls        = [results[n]['ppl'] for n in loop_counts]

    fig = plt.figure(figsize=(16, 12))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    # ── 1. PPL vs loop count ──────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(loop_counts, ppls, 'o-', color='#2196F3', lw=2, ms=8)
    ax1.axvline(train_loops, color='red', linestyle='--', alpha=0.6,
                label=f'Training depth ({train_loops})')
    ax1.fill_betweenx([min(ppls)*0.98, max(ppls)*1.02],
                      train_loops, max(loop_counts),
                      alpha=0.08, color='green', label='Extra-depth regime')
    ax1.set_xlabel("Inference Loop Count")
    ax1.set_ylabel("Validation Perplexity")
    ax1.set_title("Test-Time Loop Scaling\nDoes extra recurrence improve PPL?")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # ── 2. h_delta per loop at different depths ───────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(loop_counts)))
    for n, c in zip(loop_counts, colors):
        deltas = results[n]['h_deltas']
        ax2.plot(range(1, len(deltas)+1), deltas, 'o-',
                 color=c, label=f'loops={n}', alpha=0.85)
    ax2.axhline(1e-3, color='red', linestyle='--', alpha=0.5,
                label='Collapse threshold')
    ax2.set_xlabel("Loop Step Index")
    ax2.set_ylabel("||h_i - h_{i-1}||")
    ax2.set_title("Hidden State Delta per Loop\n(collapse if → 0)")
    ax2.legend(fontsize=7, ncol=2)
    ax2.grid(True, alpha=0.3)

    # ── 3. Attention entropy per loop ─────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    for n, c in zip(loop_counts, colors):
        ent = results[n]['attn_ent']
        ax3.plot(range(1, len(ent)+1), ent, 'o-', color=c,
                 label=f'loops={n}', alpha=0.85)
    ax3.set_xlabel("Loop Step Index")
    ax3.set_ylabel("Mean Attention Entropy (nats)")
    ax3.set_title("Attention Entropy per Loop\n(collapse if → 0)")
    ax3.legend(fontsize=7, ncol=2)
    ax3.grid(True, alpha=0.3)

    # ── 4. h_norm per loop ────────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    for n, c in zip(loop_counts, colors):
        norms = results[n]['h_norms']
        ax4.plot(range(1, len(norms)+1), norms, 'o-', color=c,
                 label=f'loops={n}', alpha=0.85)
    ax4.set_xlabel("Loop Step Index")
    ax4.set_ylabel("Hidden State Norm / sqrt(d)")
    ax4.set_title("Hidden State Norm per Loop")
    ax4.legend(fontsize=7, ncol=2)
    ax4.grid(True, alpha=0.3)

    plt.suptitle("Test-Time Loop Scaling Evaluation", fontsize=14, fontweight='bold')
    path = os.path.join(out_dir, 'loop_scaling.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  saved → {path}")
    plt.close(fig)


def print_summary(results: dict, train_loops: int):
    print(f"\n{'='*55}")
    print(f"  Test-Time Loop Scaling Summary")
    print(f"  Trained with {train_loops} loops")
    print(f"{'='*55}")
    print(f"  {'Loops':<8} {'Val Loss':<12} {'PPL':<12} {'vs train PPL'}")
    print(f"  {'-'*50}")
    train_ppl = results[train_loops]['ppl']
    for n in sorted(results.keys()):
        r = results[n]
        delta = r['ppl'] - train_ppl
        marker = '← train' if n == train_loops else ''
        print(f"  {n:<8} {r['loss']:<12.4f} {r['ppl']:<12.2f} "
              f"{delta:+.2f}  {marker}")
    print(f"{'='*55}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt',   default='out-fineweb-looped/ckpt.pt')
    parser.add_argument('--data',   default='data/fineweb/val.bin')
    parser.add_argument('--loops',  nargs='+', type=int,
                        default=[12, 16, 20, 24, 32])
    parser.add_argument('--batch',  type=int, default=8)
    parser.add_argument('--nbatch', type=int, default=50)
    parser.add_argument('--out',    default='analysis/figures')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype  = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    model, cfg, train_loops = load_model(args.ckpt, device)
    data = np.fromfile(args.data, dtype=np.uint16)
    print(f"Val data: {len(data):,} tokens")

    results = {}
    for n in args.loops:
        print(f"\nEvaluating n_loop={n} …")
        t0 = time.time()
        r  = evaluate(model, data, n, cfg.block_size,
                      args.batch, args.nbatch, device, dtype)
        dt = time.time() - t0
        print(f"  PPL={r['ppl']:.3f}  loss={r['loss']:.4f}  ({dt:.1f}s)")
        results[n] = r

    print_summary(results, train_loops)
    plot_results(results, args.out, train_loops)

    # KL + cosine at train depth
    print(f"\nComputing KL divergence + cosine similarity (n_loop={train_loops}) …")
    cos_sims, kl_divs = compute_kl_cosine(model, data, train_loops,
                                           cfg.block_size, args.batch, device, dtype)
    print(f"  cos_sim per step: {[f'{v:.4f}' for v in cos_sims]}")
    print(f"  KL div  per step: {[f'{v:.4f}' for v in kl_divs]}")

    # Plot KL and cosine
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(range(1, len(cos_sims)+1), cos_sims, 'o-', color='#9C27B0', lw=2)
    ax1.axhline(1.0, color='red', linestyle='--', alpha=0.4, label='identity (1.0)')
    ax1.set_ylim(0, 1.05)
    ax1.set_xlabel("Loop transition i → i+1")
    ax1.set_ylabel("Cosine Similarity")
    ax1.set_title("cos(h_i, h_{i+1})\n1.0 = loops are identical (collapse)")
    ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(range(1, len(kl_divs)+1), kl_divs, 'o-', color='#FF5722', lw=2)
    ax2.set_xlabel("Loop transition i → i+1")
    ax2.set_ylabel("KL Divergence (nats)")
    ax2.set_title("KL(logits_i || logits_{i+1})\n0 = prediction unchanged (useless loop)")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(args.out, 'kl_cosine.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  saved → {path}")
    plt.close(fig)


if __name__ == '__main__':
    main()
