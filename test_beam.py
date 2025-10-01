"""
Test script for beam search implementation.

This script compares beam search with greedy search to demonstrate
how beam search can find higher-probability sequences.
"""
import tiktoken
from model import GPT
import torch

def test_beam_search():
    """
    Compare greedy search vs beam search on various prompts.

    This test demonstrates that beam search can find sequences with higher
    overall probability by exploring multiple hypotheses, whereas greedy search
    only follows the single most likely token at each step.
    """
    # Load pre-trained GPT-2 model
    print("Loading GPT-2...")
    model = GPT.from_pretrained('gpt2', dict(dropout=0.0))
    model.eval()
    enc = tiktoken.get_encoding("gpt2")

    # Test prompts to compare greedy vs beam search
    prompts = [
        "The meaning of life is",
        "Once upon a time",
        "In the year 2050,",
        "The best way to learn programming is"
    ]

    for prompt in prompts:
        print(f"Prompt: '{prompt}'")

        # Encode prompt to token IDs
        prompt_ids = torch.tensor(enc.encode(prompt)).unsqueeze(0)
        
        # Baseline: Greedy search (always picks the most likely token at each step)
        print("\nGreedy Search (temperature=0.001):")
        output, prob = model.generate(prompt_ids, max_new_tokens=15, temperature=0.001)
        greedy_text = enc.decode(output[0].tolist())
        print(f"  Text: {greedy_text}")
        print(f"  Probability: {prob:.10f}")
        print(f"  Log probability: {torch.log(torch.tensor(prob)).item():.4f}")

        # Beam search with different beam widths
        # Larger beam width explores more hypotheses but is more computationally expensive
        for beam_width in [3, 5]:
            print(f"\nBeam Search (width={beam_width}):")
            results = model.beam_search(prompt_ids, max_new_tokens=5, beam_width=beam_width, temperature=0.001)
            # Show top 3 results (best sequences found by beam search)
            for rank, (seq, prob, log_score) in enumerate(results[:3], 1):
                text = enc.decode(seq.tolist())
                print(f"\n  Rank {rank}:")
                print(f"    Text: {text}")
                print(f"    Probability: {prob:.10f}")
                print(f"    Log probability: {log_score:.4f}")

def compare_beam_widths():
    """
    Demonstrate how beam width affects results and computational cost.

    This test shows that larger beam widths can find better sequences
    but require exploring exponentially more hypotheses.
    """
    # Load pre-trained GPT-2 model
    print("Loading GPT-2...")
    model = GPT.from_pretrained('gpt2', dict(dropout=0.0))
    model.eval()
    enc = tiktoken.get_encoding("gpt2")

    # Simple factual prompt for testing
    prompt = "The capital of France is"
    prompt_ids = torch.tensor(enc.encode(prompt)).unsqueeze(0)

    print(f"Prompt: '{prompt}'")
    print("Comparing different beam widths (max_new_tokens=5)")

    # Test various beam widths
    for beam_width in [1, 3, 5, 10]:
        print(f"\nBeam width = {beam_width}:")
        if beam_width == 1:
            # Beam width 1 is equivalent to greedy search
            output, prob = model.generate(prompt_ids, max_new_tokens=5, temperature=0.001)
            text = enc.decode(output[0].tolist())
            print(f"  Best: {text}")
            print(f"  Probability: {prob:.10f}")
        else:
            # Use beam search for beam_width > 1
            results = model.beam_search(prompt_ids, max_new_tokens=5, beam_width=beam_width, temperature=0.001)
            best_seq, best_prob, log_score = results[0]
            text = enc.decode(best_seq.tolist())
            print(f"  Best: {text}")
            print(f"  Probability: {best_prob:.10f}")
            # Show computational cost (number of hypotheses considered)
            print(f"  Total hypotheses explored: {beam_width ** 5}")

if __name__ == '__main__':
    # Run test 1: Compare greedy search with beam search
    print("Test 1: Comparing Greedy vs Beam Search")
    test_beam_search()
    print("\n\n")

    # Run test 2: Show effect of different beam widths
    print("Test 2: Effect of Beam Width")
    compare_beam_widths()
