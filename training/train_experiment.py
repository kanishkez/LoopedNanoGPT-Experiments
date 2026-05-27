"""
train_experiment.py

Modified nanoGPT train.py with scientific validity fixes:
  1. Routes to StandardGPT / LoopedGPT / LoopedGPTDeepSupervision via model_class
  2. Logs metrics to {out_dir}/metrics.csv  (no W&B required)
  3. Prints EXACT per-component param breakdown at startup (fix misleading ratios)
  4. Logs grad norms every step (detect explosion/instability)
  5. Logs recurrence diagnostics at diag_interval:
       - hidden state norm per loop   (detect drift)
       - hidden state delta per loop  (detect fixed-point collapse)
       - attention entropy per loop   (detect attention collapse)
  6. Logs per-step losses for LoopedGPTDeepSupervision (detect gradient starvation)
  7. Tracks total tokens seen (compare by tokens, not just iterations)

Usage:
    python train_experiment.py config/train_tinystories_standard.py
    python train_experiment.py config/train_tinystories_looped.py
    python train_experiment.py config/train_tinystories_looped_ds.py
"""

import os, time, math, pickle, csv
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model_looped import GPTConfig, build_model

# ── I/O ───────────────────────────────────────────────────────────────────────
out_dir           = 'out'
eval_interval     = 500
log_interval      = 10
eval_iters        = 100
eval_only         = False
always_save_checkpoint = True
init_from         = 'scratch'   # 'scratch' or 'resume'

# ── wandb ─────────────────────────────────────────────────────────────────────
wandb_log      = False
wandb_project  = 'looped-transformer'
wandb_run_name = 'run'

# ── data ──────────────────────────────────────────────────────────────────────
dataset                   = 'tinystories'
gradient_accumulation_steps = 1
batch_size                = 64
block_size                = 256

# ── model ─────────────────────────────────────────────────────────────────────
model_class = 'standard'   # 'standard' | 'looped' | 'looped_ds'
n_layer     = 6
n_head      = 8
n_embd      = 256
dropout     = 0.0
bias        = False
n_loop      = 1            # only used when model_class != 'standard'
ds_loss_mode = 'geometric' # only used when model_class == 'looped_ds'

# ── diagnostic settings ───────────────────────────────────────────────────────
# diag_interval: how often to run the expensive recurrence diagnostic forward
# pass (hidden norms, deltas, attention entropy). Set 0 to disable.
diag_interval = 500        # every N iters; 0 = disabled

# ── optimizer ─────────────────────────────────────────────────────────────────
learning_rate = 3e-4
max_iters     = 20000
weight_decay  = 1e-1
beta1         = 0.9
beta2         = 0.95
grad_clip     = 1.0

# ── lr schedule ───────────────────────────────────────────────────────────────
decay_lr      = True
warmup_iters  = 200
lr_decay_iters = 20000
min_lr        = 3e-5

# ── system ────────────────────────────────────────────────────────────────────
backend = 'nccl'
device  = 'cuda'
dtype   = ('bfloat16' if torch.cuda.is_available() and
            torch.cuda.is_bf16_supported() else 'float16')
compile = True

# ── config override (reads from config/*.py file via CLI arg) ─────────────────
config_keys = [k for k,v in globals().items()
               if not k.startswith('_') and isinstance(v, (int,float,bool,str))]
exec(open('configurator.py').read())
config = {k: globals()[k] for k in config_keys}

# ── DDP setup ─────────────────────────────────────────────────────────────────
ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    init_process_group(backend=backend)
    ddp_rank       = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
    seed_offset    = ddp_rank
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    master_process = True
    seed_offset    = 0
    ddp_world_size = 1

tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens/iter: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)

torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32,
           'bfloat16': torch.bfloat16,
           'float16':  torch.float16}[dtype]
ctx = (nullcontext() if device_type == 'cpu'
       else torch.amp.autocast(device_type=device_type, dtype=ptdtype))

# ── data loader ───────────────────────────────────────────────────────────────
data_dir = os.path.join('data', dataset)

def get_batch(split):
    fname = 'train.bin' if split == 'train' else 'val.bin'
    data  = np.memmap(os.path.join(data_dir, fname), dtype=np.uint16, mode='r')
    ix    = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i+block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i+1:i+1+block_size].astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), \
               y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# ── model init ────────────────────────────────────────────────────────────────
iter_num      = 0
best_val_loss = 1e9

meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']

model_args = dict(
    n_layer   = n_layer,
    n_head    = n_head,
    n_embd    = n_embd,
    block_size = block_size,
    bias      = bias,
    vocab_size = meta_vocab_size if meta_vocab_size is not None else 50257,
    dropout   = dropout,
    n_loop    = n_loop,
    ds_loss_mode = ds_loss_mode,
)

if init_from == 'scratch':
    print(f"Initialising {model_class} model from scratch")
    gptconf = GPTConfig(**model_args)
    model   = build_model(model_class, gptconf)
elif init_from == 'resume':
    ckpt_path  = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    for k in ['n_layer','n_head','n_embd','block_size','bias','vocab_size']:
        model_args[k] = checkpoint['model_args'][k]
    gptconf  = GPTConfig(**model_args)
    model    = build_model(checkpoint['model_class'], gptconf)
    model.load_state_dict({k.replace('_orig_mod.',''):v
                           for k,v in checkpoint['model'].items()})
    iter_num      = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']

# ── report params ─────────────────────────────────────────────────────────────
num_params = model.get_num_params()
print(f"\n{'='*50}")
print(f"  Model class : {model_class}")
print(f"  Parameters  : {num_params/1e6:.3f}M")
print(f"  n_layer     : {n_layer}")
print(f"  n_loop      : {n_loop}")
print(f"  n_embd      : {n_embd}  n_head: {n_head}")
print(f"  block_size  : {block_size}")
print(f"{'='*50}\n")

model.to(device)
scaler    = torch.cuda.amp.GradScaler(enabled=(dtype=='float16'))
optimizer = model.configure_optimizers(weight_decay, learning_rate,
                                       (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])

if compile:
    print("Compiling model with torch.compile …")
    model = torch.compile(model)

if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

raw_model = model.module if ddp else model

# ── CSV logger ────────────────────────────────────────────────────────────────
csv_path   = os.path.join(out_dir, 'metrics.csv')
csv_fields = ['iter','tokens_seen','train_loss','val_loss','val_ppl',
              'lr','tokens_per_sec','mfu','grad_norm',
              # recurrence diagnostics (looped models only)
              'h_norm_mean','h_delta_mean','h_delta_min','attn_ent_mean',
              # per-step losses (looped_ds only, comma-separated)
              'step_losses']

if master_process:
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields,
                                extrasaction='ignore')
        writer.writeheader()

def log_csv(row: dict):
    """Write a row; unknown keys are silently ignored (extrasaction='ignore')."""
    if master_process:
        with open(csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields,
                                    extrasaction='ignore')
            writer.writerow(row)

# ── wandb ─────────────────────────────────────────────────────────────────────
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)
    wandb.run.summary['num_params'] = num_params

# ── print param breakdown at startup ─────────────────────────────────────────
if master_process:
    print(f"\n{'='*55}")
    print(f"  Model : {model_class}")
    print(f"  n_loop={n_loop}  n_layer={n_layer}  n_embd={n_embd}")
    raw_model.print_param_breakdown()
    print(f"{'='*55}\n")

# ── estimate loss ─────────────────────────────────────────────────────────────
@torch.no_grad()
def estimate_loss():
    model.eval()
    out = {}
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

@torch.no_grad()
def run_diagnostics(X, Y):
    """
    Run one diagnostic forward pass for looped models.
    Returns dict of recurrence health metrics, or None for standard model.

    Metrics:
      h_norm_mean  — mean hidden-state L2 norm across loops (normalised by sqrt(d))
                     Watch for explosion (>10) or collapse (<0.1)
      h_delta_mean — mean ||h_i - h_{i-1}|| across loops
                     If this → 0 early: FIXED-POINT COLLAPSE (loops doing nothing)
      h_delta_min  — minimum delta (worst-case collapse)
      attn_ent_mean — mean attention entropy
                     If → 0: ATTENTION COLLAPSE (head points to single token)
    """
    if model_class not in ('looped', 'looped_ds'):
        return None
    model.eval()
    try:
        _, _, diag = raw_model.forward(X[:4], Y[:4], diagnostics=True)  # small batch
        result = {
            'h_norm_mean':  sum(diag['h_norms'])  / len(diag['h_norms']),
            'h_delta_mean': sum(diag['h_deltas']) / len(diag['h_deltas']),
            'h_delta_min':  min(diag['h_deltas']),
            'attn_ent_mean': sum(diag['attn_ents']) / len(diag['attn_ents']),
            'h_norms':  diag['h_norms'],
            'h_deltas': diag['h_deltas'],
        }
    except TypeError:
        # looped_ds doesn't expose diagnostics= argument; skip
        result = None
    model.train()
    return result

# ── lr schedule ───────────────────────────────────────────────────────────────
def get_lr(it):
    if it < warmup_iters:
        return learning_rate * (it+1) / (warmup_iters+1)
    if it > lr_decay_iters:
        return min_lr
    ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    return min_lr + 0.5*(1.0+math.cos(math.pi*ratio))*(learning_rate-min_lr)

# ── training loop ─────────────────────────────────────────────────────────────
X, Y = get_batch('train')
t0   = time.time()
local_iter_num = 0
running_mfu    = -1.0
tokens_seen    = 0          # cumulative tokens processed (compare by tokens not iters)

while True:
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for pg in optimizer.param_groups:
        pg['lr'] = lr

    # ── eval & checkpoint ─────────────────────────────────────────────────
    if iter_num % eval_interval == 0 and master_process:
        losses  = estimate_loss()
        val_ppl = math.exp(losses['val'])
        print(f"step {iter_num:5d} | tok {tokens_seen/1e6:.1f}M | "
              f"train {losses['train']:.4f} | val {losses['val']:.4f} | "
              f"ppl {val_ppl:.2f}")

        csv_row = {
            'iter':           iter_num,
            'tokens_seen':    tokens_seen,
            'train_loss':     f"{losses['train']:.4f}",
            'val_loss':       f"{losses['val']:.4f}",
            'val_ppl':        f"{val_ppl:.2f}",
            'lr':             f"{lr:.6f}",
            'tokens_per_sec': '',
            'mfu':            f"{running_mfu*100:.2f}",
            'grad_norm':      '',
        }
        log_csv(csv_row)

        if wandb_log:
            import wandb
            wandb.log({"iter":iter_num, "tokens_seen":tokens_seen,
                       "train/loss":losses['train'], "val/loss":losses['val'],
                       "val/ppl":val_ppl, "lr":lr, "mfu":running_mfu*100})

        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                ckpt = {
                    'model':       raw_model.state_dict(),
                    'optimizer':   optimizer.state_dict(),
                    'model_args':  model_args,
                    'model_class': model_class,
                    'iter_num':    iter_num,
                    'tokens_seen': tokens_seen,
                    'best_val_loss': best_val_loss,
                    'config':      config,
                    'num_params':  num_params,
                }
                torch.save(ckpt, os.path.join(out_dir, 'ckpt.pt'))
                print(f"  checkpoint saved → {out_dir}/ckpt.pt")

    if iter_num == 0 and eval_only:
        break

    # ── forward / backward ────────────────────────────────────────────────
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            model.require_backward_grad_sync = \
                (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps
        X, Y = get_batch('train')
        scaler.scale(loss).backward()

    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    # ── timing & logging ──────────────────────────────────────────────────
    t1  = time.time()
    dt  = t1 - t0
    t0  = t1
    tokens_seen += tokens_per_iter

    # ── grad norm (every step) ────────────────────────────────────────────
    grad_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            grad_norm += p.grad.data.norm(2).item() ** 2
    grad_norm = grad_norm ** 0.5

    if iter_num % log_interval == 0 and master_process:
        lossf    = loss.item() * gradient_accumulation_steps
        toks_sec = tokens_per_iter / dt
        if local_iter_num >= 5:
            mfu = raw_model.estimate_mfu(batch_size*gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu+0.1*mfu

        # ── per-step losses (looped_ds) ───────────────────────────────────
        step_losses_str = ''
        if model_class == 'looped_ds' and hasattr(raw_model, 'last_step_losses'):
            sl = raw_model.last_step_losses
            step_losses_str = '|'.join(f'{v.item() if hasattr(v,"item") else float(v):.4f}' for v in sl)
            if iter_num % (log_interval * 10) == 0:
                print(f"  step losses: {' → '.join(f'{v:.4f}' for v in sl)}")

        print(f"iter {iter_num:5d} | tok {tokens_seen/1e6:.1f}M | "
              f"loss {lossf:.4f} | {dt*1000:.1f}ms | "
              f"{toks_sec:,.0f} tok/s | gnorm {grad_norm:.3f}")

        log_csv({
            'iter':          iter_num,
            'tokens_seen':   tokens_seen,
            'train_loss':    f"{lossf:.4f}",
            'val_loss':      '', 'val_ppl': '',
            'lr':            f"{lr:.6f}",
            'tokens_per_sec': f"{toks_sec:.0f}",
            'mfu':           f"{running_mfu*100:.2f}",
            'grad_norm':     f"{grad_norm:.4f}",
            'step_losses':   step_losses_str,
        })

    # ── recurrence diagnostics ────────────────────────────────────────────
    if (diag_interval > 0 and iter_num % diag_interval == 0
            and master_process and model_class in ('looped', 'looped_ds')):
        Xd, Yd = get_batch('val')
        diag   = run_diagnostics(Xd, Yd)
        if diag is not None:
            print(f"  [DIAG iter {iter_num}] "
                  f"h_norm={diag['h_norm_mean']:.3f} "
                  f"h_delta_mean={diag['h_delta_mean']:.4f} "
                  f"h_delta_min={diag['h_delta_min']:.4f} "
                  f"attn_ent={diag['attn_ent_mean']:.3f}")
            print(f"    norms  per loop: {[f'{v:.3f}' for v in diag['h_norms']]}")
            print(f"    deltas per loop: {[f'{v:.4f}' for v in diag['h_deltas']]}")

            # Fixed-point collapse warning
            if diag['h_delta_min'] < 1e-3:
                print(f"  ⚠️  FIXED-POINT COLLAPSE DETECTED: "
                      f"min h_delta={diag['h_delta_min']:.6f} — "
                      f"later loops doing near-zero work")

            # Hidden state explosion warning
            if diag['h_norm_mean'] > 10.0:
                print(f"  ⚠️  HIDDEN STATE EXPLOSION: norm={diag['h_norm_mean']:.2f}")

            if wandb_log:
                import wandb
                wandb.log({"diag/h_norm":diag['h_norm_mean'],
                           "diag/h_delta":diag['h_delta_mean'],
                           "diag/attn_ent":diag['attn_ent_mean'],
                           "iter":iter_num})
            log_csv({
                'iter':          iter_num,
                'tokens_seen':   tokens_seen,
                'h_norm_mean':   f"{diag['h_norm_mean']:.4f}",
                'h_delta_mean':  f"{diag['h_delta_mean']:.5f}",
                'h_delta_min':   f"{diag['h_delta_min']:.5f}",
                'attn_ent_mean': f"{diag['attn_ent_mean']:.4f}",
            })

    iter_num += 1
    local_iter_num += 1
    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()
