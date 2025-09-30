"""
Sample from a trained model
"""
import os
import pickle
from contextlib import nullcontext
import torch
import tiktoken
from model import GPTConfig, GPT
import matplotlib.pyplot as plt
import torch.nn.functional as F
import time

def show_token_probabilities(logits, selected_token_idx, tokenizer, top_k=10):
    """
    Display a bar chart of top-k token probabilities and save to file.
    """
    probs = F.softmax(logits, dim=-1)
    top_probs, top_indices = torch.topk(probs, top_k)
    top_probs = top_probs.cpu().numpy()
    top_indices = top_indices.cpu().numpy()
    
    token_strings = []
    for idx in top_indices:
        token_str = tokenizer.decode([idx])
        token_str = token_str.replace('\n', '\\n').replace('\t', '\\t')
        token_strings.append(token_str[:20])
    
    # FIX: Convert both to int for proper comparison
    colors = ['green' if int(idx) == int(selected_token_idx) else 'blue' 
              for idx in top_indices]
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(top_k), top_probs, color=colors)
    plt.xlabel('Token', fontsize=12)
    plt.ylabel('Probability', fontsize=12)
    plt.title('Top 10 Token Probabilities (Green = Selected)', fontsize=14)
    plt.xticks(range(top_k), token_strings, rotation=45, ha='right')
    plt.tight_layout()
    
    for i, (bar, prob) in enumerate(zip(bars, top_probs)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{prob:.4f}', ha='center', va='bottom', fontsize=9)
    
    # Save to file
    filename = f'token_probs_{int(time.time()*1000)}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\nChart saved to: {filename}")
    plt.close()
    
    print("\n=== Top 10 Token Probabilities ===")
    for i, (token_str, prob, idx) in enumerate(zip(token_strings, top_probs, top_indices)):
        marker = " <-- SELECTED" if int(idx) == int(selected_token_idx) else ""
        print(f"{i+1}. '{token_str}' (idx={idx}): {prob:.6f}{marker}")
        
# -----------------------------------------------------------------------------
init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = 'out' # ignored if init_from is not 'resume'
start = "\n" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 10 # number of samples to draw
max_new_tokens = 500 # number of tokens generated in each sample
temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1337
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster
show_probs= False #Display token probability Distribution
use_beam_search = False  # use Beam Search
beam_width = 5 # beam width for Beam Search
exec(open('configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# model
if init_from == 'resume':
    # init from a model saved in a specific directory
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
elif init_from.startswith('gpt2'):
    # init from a given GPT-2 model
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))

model.eval()
model.to(device)
if compile:
    model = torch.compile(model) # requires PyTorch 2.0 (optional)

# look for the meta pickle in case it is available in the dataset folder
load_meta = False
if init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']: # older checkpoints might not have these...
    meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
    load_meta = os.path.exists(meta_path)
if load_meta:
    print(f"Loading meta from {meta_path}...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    # TODO want to make this more general to arbitrary encoder/decoder schemes
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
else:
    # ok let's assume gpt-2 encodings by default
    print("No meta.pkl found, assuming GPT-2 encodings...")
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

# encode the beginning of the prompt
if start.startswith('FILE:'):
    with open(start[5:], 'r', encoding='utf-8') as f:
        start = f.read()
start_ids = encode(start)
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

# run generation
with torch.no_grad():
    with ctx:
        for k in range(num_samples):
          if show_probs:
                # New Generation loop with visualization
                y = x
                for _ in range(max_new_tokens):
                    # Crop context if needed
                    idx_cond = y if y.size(1) <= model.config.block_size else y[:, -model.config.block_size:]
                    
                    # Forward pass
                    logits, _ = model(idx_cond)
                    logits = logits[:, -1, :] / temperature
                    
                    # Apply top-k filtering if specified
                    if top_k is not None:
                        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                        logits[logits < v[:, [-1]]] = -float('Inf')
                    
                    # Sample from distribution
                    probs = F.softmax(logits, dim=-1)
                    idx_next = torch.multinomial(probs, num_samples=1)
                    
                    # Show probabilities for this step
                    show_token_probabilities(logits[0], idx_next[0].item(), enc, top_k=10)
                    
                    # Append to sequence
                    y = torch.cat((y, idx_next), dim=1)
                
                print(decode(y[0].tolist()))
                print('---------------')

          else:
            #Generation that returns probability 
            y, prob = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k,
                        use_beam_search=use_beam_search, beam_width=beam_width)
            print(decode(y[0].tolist()))
            print(f'Sequence probability: {prob:.6e}')
            print('---------------')
