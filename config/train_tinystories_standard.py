# config/train_tinystories_standard.py
# Standard 6-layer GPT baseline on TinyStories

dataset   = 'tinystories'
out_dir   = 'out-tinystories-standard'
wandb_run_name = 'standard-6L-256d'

model_class = 'standard'
n_layer  = 6
n_loop   = 1       # unused for standard, documents intent
n_head   = 8
n_embd   = 256
block_size = 256
bias     = False
dropout  = 0.0

batch_size                = 64
gradient_accumulation_steps = 1
max_iters    = 20000
eval_interval = 500
eval_iters   = 100
log_interval = 10

learning_rate  = 3e-4
weight_decay   = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
decay_lr  = True
warmup_iters   = 200
lr_decay_iters = 20000
min_lr = 3e-5

compile = True
