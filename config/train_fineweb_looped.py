# config/train_fineweb_looped.py
# Looped GPT — 1 shared block × 12 loops, ~28M total / ~4.7M block params
# Same effective depth as standard (12), 12× fewer block params

dataset   = 'fineweb'
out_dir   = 'out-fineweb-looped'
wandb_run_name = 'fineweb-looped-1x12-768d'

model_class = 'looped'
n_layer  = 1
n_loop   = 12
n_head   = 12
n_embd   = 768
block_size = 512
bias     = False
dropout  = 0.0

# Same effective batch as standard for fair comparison
batch_size                = 16
gradient_accumulation_steps = 8
max_iters    = 8000
eval_interval = 500
eval_iters   = 100
log_interval = 10

# Recurrence diagnostics every 1000 iters (expensive: runs slow path attn)
diag_interval = 1000

learning_rate  = 3e-4
weight_decay   = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
decay_lr  = True
warmup_iters   = 500
lr_decay_iters = 8000
min_lr = 3e-5

dtype   = 'bfloat16'
compile = False    # diagnostics need .item() — incompatible with compile
