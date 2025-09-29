"""
Evaluate a trained model on prompt-response pairs
"""
import os
import pickle
import json
from contextlib import nullcontext
import torch
import tiktoken
from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = 'out' # ignored if init_from is not 'resume'
eval_data_file = 'eval_data.json' # path to evaluation data
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
seed = 1337
exec(open('configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# Load model
if init_from == 'resume':
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
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))

model.eval()
model.to(device)

# Load tokenizer
print("Loading tokenizer...")
enc = tiktoken.get_encoding("gpt2")
encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
decode = lambda l: enc.decode(l)

def eval(eval_data_file, model, encode, decode, device):
    """
    Evaluate model on prompt-response pairs from JSON file.
    
    Args:
        eval_data_file: Path to JSON file containing prompt-response pairs
        model: GPT model to evaluate
        encode: Encoding function
        decode: Decoding function
        device: Device to run on
    
    Returns:
        Total summed probability across all examples
    """
    # Load evaluation data
    print(f"Loading evaluation data from {eval_data_file}...")
    with open(eval_data_file, 'r', encoding='utf-8') as f:
        eval_data = json.load(f)
    
    print(f"Found {len(eval_data)} evaluation examples\n")
    
    total_prob = 0.0
    results = []
    
    print("="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    
    for i, example in enumerate(eval_data, 1):
        prompt = example['prompt']
        expected_response = example['response']
        
        # Encode prompt and response
        prompt_ids = encode(prompt)
        response_ids = encode(expected_response)
        
        # Response IDs are already properly encoded
        # (The response comes from JSON, not from model output)
        
        # Create input tensor
        x = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)
        
        # Generate with fixed response to get probability
        with torch.no_grad():
            with ctx:
                output, prob = model.generate(
                    x, 
                    max_new_tokens=len(response_ids),
                    fixed_response=response_ids,
                    temperature=1.0
                )
        
        # Decode the full sequence
        full_text = decode(output[0].tolist())
        
        # Store results
        results.append({
            'prompt': prompt,
            'expected_response': expected_response,
            'probability': prob
        })
        
        total_prob += prob
        
        # Print results
        print(f"\nExample {i}:")
        print(f"  Prompt: '{prompt}'")
        print(f"  Expected response: '{expected_response}'")
        print(f"  Full text: '{full_text}'")
        print(f"  Probability: {prob:.10f}")
        print(f"  Log probability: {torch.log(torch.tensor(prob)).item():.4f}")
    
    print("\n" + "="*70)
    print(f"SUMMARY")
    print("="*70)
    print(f"Total examples: {len(eval_data)}")
    print(f"Sum of probabilities: {total_prob:.10f}")
    print(f"Average probability: {total_prob / len(eval_data):.10f}")
    print("="*70)
    
    return total_prob, results

# Run evaluation
if __name__ == '__main__':
    total_prob, results = eval(eval_data_file, model, encode, decode, device)
    
    # Optionally save results
    results_file = eval_data_file.replace('.json', '_results.json')
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump({
            'total_probability': total_prob,
            'average_probability': total_prob / len(results),
            'results': results
        }, f, indent=2)
    print(f"\nResults saved to {results_file}")