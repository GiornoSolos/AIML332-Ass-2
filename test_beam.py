import tiktoken        
from model import GPT
import torch

# Test script for beam search
def test_beam_search():
    """
    Compare greedy search vs beam search
    """
    print("Loading GPT-2...")
    model = GPT.from_pretrained('gpt2', dict(dropout=0.0))
    model.eval()    
    enc = tiktoken.get_encoding("gpt2")
    
    prompts = [          
        "The meaning of life is",
        "Once upon a time",
        "In the year 2050,",
        "The best way to learn programming is"    
    ]

    for prompt in prompts:
        print("\n" + "="*70)
        print(f"PROMPT: '{prompt}'")
        print("="*70)
        
        prompt_ids = torch.tensor(enc.encode(prompt)).unsqueeze(0)
        
        # Greedy search (baseline)
        print("\nGREEDY SEARCH (temperature=0.001):")
        output, prob = model.generate(prompt_ids, max_new_tokens=15, temperature=0.001)
        greedy_text = enc.decode(output[0].tolist())
        print(f"  Text: {greedy_text}")
        print(f"  Probability: {prob:.10f}")           
        print(f"  Log probability: {torch.log(torch.tensor(prob)).item():.4f}")
        
        # Beam search with different beam widths
        for beam_width in [3, 5]:
            print(f"\nBEAM SEARCH (width={beam_width}):")
            results = model.beam_search(prompt_ids, max_new_tokens=5, beam_width=beam_width, temperature=1.0)          
            # Show top 3 results
            for rank, (seq, prob, log_score) in enumerate(results[:3], 1):
                text = enc.decode(seq.tolist())
                print(f"\n  Rank {rank}:")
                print(f"    Text: {text}")
                print(f"    Probability: {prob:.10f}")
                print(f"    Log probability: {log_score:.4f}")

def compare_beam_widths():
    """
    Show how beam width affects results
    """
    print("Loading GPT-2...")
    model = GPT.from_pretrained('gpt2', dict(dropout=0.0))
    model.eval()
    enc = tiktoken.get_encoding("gpt2")
    prompt = "The capital of France is"
    prompt_ids = torch.tensor(enc.encode(prompt)).unsqueeze(0)
    print("\n" + "="*70)
    print(f"PROMPT: '{prompt}'")
    print("Comparing different beam widths (max_new_tokens=5)")
    print("="*70)
    for beam_width in [1, 3, 5, 10]:
        print(f"\nBeam width = {beam_width}:")
        if beam_width == 1:
            # Beam width 1 is equivalent to greedy
            output, prob = model.generate(prompt_ids, max_new_tokens=5, temperature=0.001)
            text = enc.decode(output[0].tolist())
            print(f"  Best: {text}")
            print(f"  Probability: {prob:.10f}")
        else:
            results = model.beam_search(prompt_ids, max_new_tokens=5, beam_width=beam_width, temperature=1.0)
            best_seq, best_prob, log_score = results[0]
            text = enc.decode(best_seq.tolist())
            print(f"  Best: {text}")
            print(f"  Probability: {best_prob:.10f}")
            print(f"  Total hypotheses explored: {beam_width ** 5}")

if __name__ == '__main__':
    print("Test 1: Comparing Greedy vs Beam Search")
    test_beam_search()
    print("\n\n")
    print("Test 2: Effect of Beam Width")
    compare_beam_widths()
