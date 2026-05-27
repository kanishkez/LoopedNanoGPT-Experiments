# config/train_tinystories_looped_ds.py
# Looped GPT with deep supervision (aux loss at each loop step)

dataset   = 'tinystories'
out_dir   = 'out-tinystories-looped-ds'
wandb_run_name = 'looped-ds-geometric-6x'

model_class   = 'looped_ds'
n_layer  = 1
n_loop   = 6
n_head   = 8
n_embd   = 256
block_size = 256
bias     = False
dropout  = 0.0

# Deep supervision loss weighting: "geometric" | "uniform" | "final_only"
ds_loss_mode = 'geometric'

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


compile = False
