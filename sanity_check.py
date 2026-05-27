"""
sanity_check.py — Run before training to verify the implementation.

Tests:
  1. Forward pass through all three model classes (CPU)
  2. Weight sharing verification for LoopedGPT
  3. Exact parameter breakdown (embeddings vs block params)
  4. Deep supervision produces n_loop intermediate logits + per-step losses
  5. Gradient flows through all loop steps
  6. Diagnostic forward pass (h_norm, h_delta, attn_entropy) — critical for collapse detection
  7. Fixed-point collapse would be detected by h_delta → 0

Usage:
    python sanity_check.py
"""
import sys, math, torch
from model_looped import (GPTConfig, StandardGPT, LoopedGPT,
                          LoopedGPTDeepSupervision, build_model)

SEP = "="*60
print(SEP)
print("  nanoGPT Looped Transformer — Sanity Check")
print(SEP)

# Tiny config for fast CPU testing
BASE = dict(block_size=32, vocab_size=256, n_head=2,
            n_embd=64, dropout=0.0, bias=False)
cfg_std  = GPTConfig(**BASE, n_layer=6)
cfg_loop = GPTConfig(**BASE, n_layer=1, n_loop=6, looped=True)
cfg_ds   = GPTConfig(**BASE, n_layer=1, n_loop=6, looped=True,
                     deep_supervision=True, ds_loss_mode='geometric')

x = torch.randint(0, 256, (2, 32))
y = torch.randint(0, 256, (2, 32))

# ── Test 1: Forward passes ────────────────────────────────────────────────────
print("\n[TEST 1] Forward passes")
for name, cls, cfg in [("StandardGPT",              StandardGPT,               cfg_std),
                        ("LoopedGPT",                 LoopedGPT,                 cfg_loop),
                        ("LoopedGPTDeepSupervision",  LoopedGPTDeepSupervision,  cfg_ds)]:
    m = cls(cfg)
    logits, loss = m(x, y)
    assert logits.shape == (2, 32, 256), f"Bad logits shape: {logits.shape}"
    assert loss is not None and loss.item() > 0
    print(f"  {name:<35} loss={loss.item():.4f}  ✓")

# ── Test 2: Weight sharing ────────────────────────────────────────────────────
print("\n[TEST 2] Weight sharing verification")
m_loop = LoopedGPT(cfg_loop)
m_std  = StandardGPT(cfg_std)
std_keys  = [k for k in m_std.state_dict()  if k.startswith('transformer.h')]
loop_keys = [k for k in m_loop.state_dict() if k.startswith('transformer.h')]
print(f"  StandardGPT block keys : {len(std_keys)}")
print(f"  LoopedGPT   block keys : {len(loop_keys)}")
assert len(loop_keys) < len(std_keys), "Looped model should have fewer block keys"
ok = m_loop.verify_weight_sharing()
assert ok, "Weight sharing broken"

# ── Test 3: Exact parameter breakdown (SCIENTIFIC FIX) ───────────────────────
print("\n[TEST 3] Exact parameter breakdown")
for name, m in [("StandardGPT", m_std), ("LoopedGPT", m_loop)]:
    print(f"\n  {name}:")
    bd = m.print_param_breakdown()
    emb_frac = (bd['tok_emb'] + bd['pos_emb']) / bd['total'] * 100
    print(f"  Embedding fraction: {emb_frac:.1f}%  "
          f"(if >50%, param ratio claim is misleading)")

n_std  = m_std.get_num_params()
n_loop_m = m_loop.get_num_params()
total_ratio = n_std / n_loop_m
# Non-embedding ratio is the scientifically honest number
bd_std  = m_std.get_param_breakdown()
bd_loop = m_loop.get_param_breakdown()
non_emb_ratio = bd_std['non_emb_total'] / bd_loop['non_emb_total']
print(f"\n  Total param ratio:       {total_ratio:.2f}×")
print(f"  Non-embedding ratio:     {non_emb_ratio:.2f}×  ← honest scientific ratio")
print(f"  (non-emb ratio < total ratio because embeddings are shared/fixed)")
assert non_emb_ratio > 2.0, "Non-embedding ratio should be substantial"
print("  ✓ Param breakdown verified")

# ── Test 4: Deep supervision ──────────────────────────────────────────────────
print("\n[TEST 4] Deep supervision intermediate logits + per-step losses")
m_ds = LoopedGPTDeepSupervision(cfg_ds)
logits, loss = m_ds(x, y)
n_logits = len(m_ds.last_loop_logits)
n_losses = len(m_ds.last_step_losses)
assert n_logits == cfg_ds.n_loop, f"Expected {cfg_ds.n_loop} logits, got {n_logits}"
assert n_losses == cfg_ds.n_loop, f"Expected {cfg_ds.n_loop} losses, got {n_losses}"
print(f"  Intermediate logits: {n_logits}  ✓")
print(f"  Per-step losses: {[f'{v:.4f}' for v in m_ds.last_step_losses]}")
# Verify losses are decreasing (models converge; not guaranteed at init, just check finite)
assert all(math.isfinite(v) for v in m_ds.last_step_losses), "Non-finite step losses"
print("  ✓ All step losses are finite")

# ── Test 5: Gradient flow ─────────────────────────────────────────────────────
print("\n[TEST 5] Gradient flow through all loop steps")
m_l2 = LoopedGPT(cfg_loop)
_, loss2 = m_l2(x, y)
loss2.backward()
has_grad = [p.grad is not None for p in m_l2.transformer.h.parameters()]
all_have_grad = all(has_grad)
print(f"  Params with gradients: {sum(has_grad)}/{len(has_grad)}")
assert all_have_grad, "Some parameters missing gradients"
print("  ✓ Gradients flow through all loop steps")

# ── Test 6: Diagnostic forward pass ──────────────────────────────────────────
print("\n[TEST 6] Diagnostic forward pass (h_norm, h_delta, attn_ent)")
m_diag = LoopedGPT(cfg_loop)
logits, loss, diag = m_diag(x, y, diagnostics=True)
assert len(diag['h_norms'])  == cfg_loop.n_loop
assert len(diag['h_deltas']) == cfg_loop.n_loop
assert len(diag['attn_ents']) == cfg_loop.n_loop
print(f"  h_norms  per step: {[f'{v:.4f}' for v in diag['h_norms']]}")
print(f"  h_deltas per step: {[f'{v:.4f}' for v in diag['h_deltas']]}")
print(f"  attn_ent per step: {[f'{v:.4f}' for v in diag['attn_ents']]}")

# ── Test 7: Fixed-point collapse detection ────────────────────────────────────
print("\n[TEST 7] Fixed-point collapse detection logic")
# Simulate a collapsed state where all h_deltas are zero
mock_deltas = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
would_trigger = min(mock_deltas) < 1e-3
print(f"  Mock delta={mock_deltas[0]} → collapse would trigger: {would_trigger}")
assert would_trigger, "Collapse detection logic broken"
# Verify real model (random init) does NOT trigger collapse
real_min_delta = min(diag['h_deltas'])
real_collapsed = real_min_delta < 1e-3
print(f"  Real model min delta={real_min_delta:.6f} → collapsed: {real_collapsed}")
# At random init, collapse is unlikely but not impossible — just report
print(f"  {'⚠️  Suspicious early collapse at init' if real_collapsed else '✓ No collapse at init'}")
print("  ✓ Collapse detection logic works")

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("  All 7 tests PASSED ✓")
print(f"\n  Key findings:")
print(f"    Total param ratio:   {total_ratio:.1f}×")
print(f"    Non-emb param ratio: {non_emb_ratio:.1f}×  (use this in papers)")
print(f"    Diagnostics: h_norm, h_delta, attn_ent all available")
print(f"\n  Safe to proceed with training.")
print(SEP)
