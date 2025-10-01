out_dir = 'out-tinystories'
eval_interval = 100
eval_iters = 20
log_interval = 10
wandb_log = False

init_from = 'gpt2'

max_iters = 1000 # total number of training iterations
learning_rate = 5e-5
batch_size = 1 # this is the micro-batch size
gradient_accumulation_steps = 32 # used to simulate larger batch sizes
block_size = 256  # Context length

dataset = 'tinystories'
compile = False
