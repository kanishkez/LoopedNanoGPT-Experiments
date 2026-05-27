"""
model_looped.py - Standard vs Looped Transformer for research experiment.

SCIENTIFIC FIXES applied (see critique):
  - get_param_breakdown(): exact per-component param counts (embeddings ≠ block params)
  - CausalSelfAttention: optional attention entropy return (for collapse detection)
  - LoopedGPT.forward(): optional diagnostic mode returning per-loop stats:
      * hidden state L2 norm
      * hidden state delta ||h_{i} - h_{i-1}||   (fixed-point collapse detector)
      * mean attention entropy per head
  - LoopedGPTDeepSupervision: logs per-step loss values (detached) for gradient starvation analysis
  - Correct MFU estimate uses actual depth (n_loop), not param count
  - Residual init std uses n_loop for looped models
"""
import math, inspect
from dataclasses import dataclass
from typing import Optional
import torch, torch.nn as nn
from torch.nn import functional as F


# ═══════════════════════════════════════════════════════════════════════════════
# Building blocks
# ═══════════════════════════════════════════════════════════════════════════════

class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias   = nn.Parameter(torch.zeros(ndim)) if bias else None
    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn  = nn.Linear(config.n_embd, 3*config.n_embd, bias=config.bias)
        self.c_proj  = nn.Linear(config.n_embd, config.n_embd,   bias=config.bias)
        self.attn_dropout  = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head  = config.n_head
        self.n_embd  = config.n_embd
        self.dropout = config.dropout
        self.flash   = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            self.register_buffer("bias",
                torch.tril(torch.ones(config.block_size, config.block_size))
                     .view(1,1,config.block_size,config.block_size))

    def forward(self, x, return_entropy: bool = False):
        """
        Args:
            return_entropy: if True, also return mean attention entropy across
                            heads and batch — used to detect attention collapse.
        """
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B,T,self.n_head,C//self.n_head).transpose(1,2)
        q = q.view(B,T,self.n_head,C//self.n_head).transpose(1,2)
        v = v.view(B,T,self.n_head,C//self.n_head).transpose(1,2)

        if self.flash and not return_entropy:
            # Fast path: flash attention, no entropy available
            y = F.scaled_dot_product_attention(q,k,v,attn_mask=None,
                dropout_p=self.dropout if self.training else 0, is_causal=True)
            entropy = None
        else:
            # Slow path: manual attention — always used when entropy is needed
            att = (q @ k.transpose(-2,-1)) * (1.0/math.sqrt(k.size(-1)))
            att = att.masked_fill(
                torch.tril(torch.ones(T,T,device=x.device)).view(1,1,T,T)==0,
                float('-inf'))
            att_softmax = F.softmax(att, dim=-1)
            if return_entropy:
                # Shannon entropy H = -sum(p*log(p)), averaged over batch & heads
                # clamp to avoid log(0)
                p = att_softmax.clamp(min=1e-9)
                entropy = -(p * p.log()).sum(dim=-1).mean().item()
            else:
                entropy = None
            att_softmax = self.attn_dropout(att_softmax)
            y = att_softmax @ v

        y = y.transpose(1,2).contiguous().view(B,T,C)
        out = self.resid_dropout(self.c_proj(y))
        return (out, entropy) if return_entropy else out


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4*config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4*config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
    def forward(self, x):
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


class Block(nn.Module):
    """Pre-norm transformer block."""
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp  = MLP(config)

    def forward(self, x, return_entropy: bool = False):
        if return_entropy:
            attn_out, ent = self.attn(self.ln_1(x), return_entropy=True)
            x = x + attn_out
        else:
            x   = x + self.attn(self.ln_1(x))
            ent = None
        x = x + self.mlp(self.ln_2(x))
        return (x, ent) if return_entropy else x


# ═══════════════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class GPTConfig:
    block_size: int   = 256
    vocab_size: int   = 50257
    n_layer:    int   = 6
    n_head:     int   = 8
    n_embd:     int   = 256
    dropout:    float = 0.0
    bias:       bool  = False
    # looped model
    looped:           bool = False
    n_loop:           int  = 1
    # deep supervision
    deep_supervision: bool = False
    ds_loss_mode:     str  = "geometric"   # "uniform"|"geometric"|"final_only"


# ═══════════════════════════════════════════════════════════════════════════════
# Base class — shared utilities
# ═══════════════════════════════════════════════════════════════════════════════

class _GPTBase(nn.Module):

    # ── Parameter counting ─────────────────────────────────────────────────────

    def get_num_params(self, non_embedding: bool = True) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.transformer.wpe.weight.numel()
        return n

    def get_param_breakdown(self) -> dict:
        """
        SCIENTIFIC FIX #1: Return exact per-component parameter counts.

        At small scale (n_embd=256, vocab=50257) embeddings dominate:
          tok_emb ≈ 50257*256 = 12.9M   >> block params ≈ 1.3M per block

        This means a "6× smaller" claim may only be "2× smaller" in practice.
        Always log this breakdown before comparing models.

        Returns dict with keys:
          tok_emb, pos_emb, attention, ffn, layer_norm, lm_head, total, non_emb_total
        """
        breakdown = {'tok_emb': 0, 'pos_emb': 0, 'attention': 0,
                     'ffn': 0, 'layer_norm': 0, 'lm_head': 0}

        for name, p in self.named_parameters():
            n = p.numel()
            if 'wte' in name:
                breakdown['tok_emb'] += n
            elif 'wpe' in name:
                breakdown['pos_emb'] += n
            elif 'c_attn' in name or ('c_proj' in name and 'mlp' not in name and '.h.' in name):
                breakdown['attention'] += n
            elif 'c_fc' in name or ('c_proj' in name and 'mlp' in name):
                breakdown['ffn'] += n
            elif 'ln_' in name or 'ln_f' in name:
                breakdown['layer_norm'] += n
            elif 'lm_head' in name:
                # lm_head shares weights with tok_emb — count separately for clarity
                breakdown['lm_head'] += n
            else:
                breakdown['attention'] += n  # catch-all for attn proj

        breakdown['total']       = sum(p.numel() for p in self.parameters())
        breakdown['non_emb_total'] = breakdown['total'] - breakdown['pos_emb']
        return breakdown

    def print_param_breakdown(self):
        bd = self.get_param_breakdown()
        print(f"\n  {'Component':<20} {'Params':>12}  {'% of total':>10}")
        print(f"  {'-'*46}")
        total = bd['total']
        for k in ['tok_emb','pos_emb','attention','ffn','layer_norm','lm_head']:
            pct = 100*bd[k]/total if total>0 else 0
            print(f"  {k:<20} {bd[k]:>12,}  {pct:>9.1f}%")
        print(f"  {'-'*46}")
        print(f"  {'total':<20} {total:>12,}  100.0%")
        print(f"  {'non_emb_total':<20} {bd['non_emb_total']:>12,}\n")
        return bd

    # ── Initialisation ─────────────────────────────────────────────────────────

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _scaled_residual_init(self):
        """GPT-2 scaled residual init: std = 0.02 / sqrt(2 * effective_depth)."""
        depth = getattr(self.config, 'n_loop', self.config.n_layer)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2*depth))

    # ── Optimiser ──────────────────────────────────────────────────────────────

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict   = {n:p for n,p in self.named_parameters() if p.requires_grad}
        decay_params = [p for n,p in param_dict.items() if p.dim() >= 2]
        nodecay      = [p for n,p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay,      'weight_decay': 0.0},
        ]
        fused_ok  = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_ok and device_type == 'cuda'
        opt = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas,
                                **({'fused':True} if use_fused else {}))
        print(f"  fused AdamW: {use_fused}")
        return opt

    # ── MFU ───────────────────────────────────────────────────────────────────

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """
        MFU estimate. NOTE: theoretical FLOPs ≠ wall-clock efficiency for
        looped models (cache locality, sequential depth). Always cross-check
        with measured tokens/sec.
        """
        N    = self.get_num_params()
        cfg  = self.config
        depth = getattr(cfg, 'n_loop', cfg.n_layer)
        L, H, Q, T = depth, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops = (6*N + 12*L*H*Q*T) * T * fwdbwd_per_iter
        return (flops / dt) / 312e12

    # ── Generation ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_c = (idx if idx.size(1) <= self.config.block_size
                     else idx[:, -self.config.block_size:])
            logits, _ = self(idx_c)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-Inf')
            idx_next = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


# ═══════════════════════════════════════════════════════════════════════════════
# Model 1 — StandardGPT
# ═══════════════════════════════════════════════════════════════════════════════

class StandardGPT(_GPTBase):
    """Standard transformer: n_layer independent blocks. Research baseline."""
    model_type = "standard"

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte  = nn.Embedding(config.vocab_size, config.n_embd),
            wpe  = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h    = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)
        self._scaled_residual_init()
        print(f"[StandardGPT] {self.get_num_params()/1e6:.3f}M params | "
              f"{config.n_layer} unique blocks")
        self.print_param_breakdown()

    def forward(self, idx, targets=None):
        b, t = idx.size()
        assert t <= self.config.block_size
        pos = torch.arange(0, t, dtype=torch.long, device=idx.device)
        x   = self.transformer.drop(self.transformer.wte(idx) +
                                    self.transformer.wpe(pos))
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        if targets is not None:
            logits = self.lm_head(x)
            loss   = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                     targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss   = None
        return logits, loss


# ═══════════════════════════════════════════════════════════════════════════════
# Model 2 — LoopedGPT
# ═══════════════════════════════════════════════════════════════════════════════

class LoopedGPT(_GPTBase):
    """
    Looped / weight-shared transformer.

    SCIENTIFIC FIXES:
    - forward() has `diagnostics=True` mode that returns per-loop stats:
        * h_norm:   ||h_i||_2 / sqrt(d)      — detect hidden state explosion
        * h_delta:  ||h_i - h_{i-1}||_2      — detect fixed-point collapse
                    (if h_delta → 0, later loops are doing nothing useful)
        * attn_ent: mean attention entropy    — detect attention collapse
    - These are logged to train_experiment.py at `diag_interval` steps.

    ARCHITECTURE NOTES:
    - Positional embeddings injected ONCE before loop 0 (purer recurrence).
      The model must rely on positional geometry surviving repeated transforms.
    - Each loop step applies INDEPENDENT dropout masks (different noise each step).
      This acts like stochastic depth and helps prevent fixed-point collapse.
    - Gradient flows through all n_loop unrolled steps (full BPTT).
    """
    model_type = "looped"

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_loop >= 1
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte  = nn.Embedding(config.vocab_size, config.n_embd),
            wpe  = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h    = Block(config),      # ← SINGLE shared block (not ModuleList)
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)
        self._scaled_residual_init()
        print(f"[LoopedGPT]   {self.get_num_params()/1e6:.3f}M params | "
              f"1 shared block × {config.n_loop} loops")
        self.print_param_breakdown()

    def forward(self, idx, targets=None, diagnostics: bool = False):
        """
        Args:
            diagnostics: if True, return (logits, loss, diag_dict) where
                diag_dict = {
                    'h_norms':   [float] * n_loop,   # L2 norm of hidden state
                    'h_deltas':  [float] * n_loop,   # change from previous step
                    'attn_ents': [float] * n_loop,   # mean attention entropy
                }
        """
        b, t = idx.size()
        assert t <= self.config.block_size
        pos = torch.arange(0, t, dtype=torch.long, device=idx.device)
        x   = self.transformer.drop(self.transformer.wte(idx) +
                                    self.transformer.wpe(pos))

        h_norms, h_deltas, attn_ents = [], [], []
        x_prev = x

        for i in range(self.config.n_loop):
            if diagnostics:
                x, ent = self.transformer.h(x, return_entropy=True)
                with torch.no_grad():
                    # Normalise by sqrt(d) so scale is independent of n_embd
                    norm  = x.norm(dim=-1).mean().item() / math.sqrt(self.config.n_embd)
                    delta = (x - x_prev).norm(dim=-1).mean().item()
                    h_norms.append(norm)
                    h_deltas.append(delta)
                    attn_ents.append(ent if ent is not None else float('nan'))
                x_prev = x.detach().clone()
            else:
                x = self.transformer.h(x)

        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss   = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                     targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss   = None

        if diagnostics:
            diag = {'h_norms': h_norms, 'h_deltas': h_deltas,
                    'attn_ents': attn_ents}
            return logits, loss, diag

        return logits, loss

    def verify_weight_sharing(self):
        block_keys = [k for k in self.state_dict() if k.startswith('transformer.h')]
        indexed    = [k for k in block_keys if '.h.0.' in k or '.h.1.' in k]
        ok = not indexed
        print(f"[Weight Sharing] {len(block_keys)} block keys — "
              f"{'✓ shared' if ok else '✗ BROKEN'}")
        return ok


# ═══════════════════════════════════════════════════════════════════════════════
# Model 3 — LoopedGPTDeepSupervision
# ═══════════════════════════════════════════════════════════════════════════════

class LoopedGPTDeepSupervision(_GPTBase):
    """
    Looped transformer with deep supervision (auxiliary loss at each step).

    SCIENTIFIC FIXES:
    - last_step_losses: list of per-step CE loss values (detached floats).
      Used to detect gradient starvation:
        if loss[0] >> loss[-1] but loss[0] barely decreases → early loops starved.
    - Loss weights control gradient flow to early loops.
      'geometric': w_i = 2^i/sum  — strongly favours later steps.
      'uniform':   w_i = 1/n_loop — equal gradient to all steps.
      Recommend starting with 'uniform' to avoid loop-1 starvation,
      then switching to 'geometric' if refinement is confirmed.
    """
    model_type = "looped_ds"

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_loop >= 1
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte  = nn.Embedding(config.vocab_size, config.n_embd),
            wpe  = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h    = Block(config),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        self._loss_weights  = self._make_weights(config.ds_loss_mode, config.n_loop)
        self.last_loop_logits: list = []
        self.last_step_losses: list = []   # per-step CE loss (detached floats)
        self.apply(self._init_weights)
        self._scaled_residual_init()
        print(f"[LoopedGPT-DS] {self.get_num_params()/1e6:.3f}M params | "
              f"1 block × {config.n_loop} loops | mode={config.ds_loss_mode}")
        print(f"  Loss weights: {[f'{w:.3f}' for w in self._loss_weights]}")
        self.print_param_breakdown()

    @staticmethod
    def _make_weights(mode, n):
        if mode == "uniform":
            return [1.0/n]*n
        elif mode == "geometric":
            raw = [2.0**i for i in range(n)]
            s   = sum(raw); return [w/s for w in raw]
        elif mode == "final_only":
            return [0.0]*(n-1) + [1.0]
        raise ValueError(f"Unknown ds_loss_mode: {mode!r}")

    def forward(self, idx, targets=None):
        b, t = idx.size()
        assert t <= self.config.block_size
        pos = torch.arange(0, t, dtype=torch.long, device=idx.device)
        x   = self.transformer.drop(self.transformer.wte(idx) +
                                    self.transformer.wpe(pos))
        loop_logits = []
        for _ in range(self.config.n_loop):
            x           = self.transformer.h(x)
            x_n         = self.transformer.ln_f(x)
            step_logits = self.lm_head(x_n)
            loop_logits.append(step_logits)

        self.last_loop_logits = [l.detach() for l in loop_logits]

        if targets is not None:
            step_losses = [
                F.cross_entropy(l.view(-1, l.size(-1)),
                                targets.view(-1), ignore_index=-1)
                for l in loop_logits
            ]
            # Store per-step losses for gradient starvation analysis
            self.last_step_losses = [sl.item() for sl in step_losses]
            loss   = sum(w * sl for w, sl in zip(self._loss_weights, step_losses))
            logits = loop_logits[-1]
        else:
            self.last_step_losses = []
            logits = loop_logits[-1][:, [-1], :]
            loss   = None

        return logits, loss


# ═══════════════════════════════════════════════════════════════════════════════
# Factory
# ═══════════════════════════════════════════════════════════════════════════════

def build_model(model_class: str, config: GPTConfig):
    dispatch = {
        "standard":  StandardGPT,
        "looped":    LoopedGPT,
        "looped_ds": LoopedGPTDeepSupervision,
    }
    if model_class not in dispatch:
        raise ValueError(f"Unknown model_class {model_class!r}. "
                         f"Choose: {list(dispatch)}")
    return dispatch[model_class](config)
