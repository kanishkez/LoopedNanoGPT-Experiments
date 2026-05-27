# config/train_fineweb_standard.py
# Standard GPT — 12 layers, ~85M params — FineWeb-Edu scaling run

dataset   = 'fineweb'
out_dir   = 'out-fineweb-standard'
wandb_run_name = 'fineweb-standard-12L-768d'

model_class = 'standard'
n_layer  = 12
n_head   = 12
n_embd   = 768
block_size = 512
bias     = False
dropout  = 0.0

# Effective batch = 16 * 8 * 512 = 65,536 tokens/step
batch_size                = 16
gradient_accumulation_steps = 8
max_iters    = 8000
eval_interval = 500
eval_iters   = 100
log_interval = 10
diag_interval = 0    # standard model has no recurrence diagnostics

learning_rate  = 3e-4
weight_decay   = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
decay_lr  = True
warmup_iters   = 500
lr_decay_iters = 8000
min_lr = 3e-5

# bf16 + compile for maximum speed
dtype   = 'bfloat16'
compile = True
