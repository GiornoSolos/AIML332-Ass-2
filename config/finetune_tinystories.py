out_dir = 'out-tinystories'
eval_interval = 100
eval_iters = 20
log_interval = 10
wandb_log = False

init_from = 'gpt2'

max_iters = 1000 # reduced for quicker testing
learning_rate = 5e-5
batch_size = 1  # Reduced to 1 to avoid OOM errors
gradient_accumulation_steps = 32  # Increased to maintain effective batch size
block_size = 256  # Further reduced context window to save memory

dataset = 'tinystories'
compile = False
