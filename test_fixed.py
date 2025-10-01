"""
Test script for GPT-2 probability calculations with fixed continuations.

This script tests the model's ability to calculate probabilities for specific
continuations given different contexts, demonstrating:
1. Comparative probabilities for different word choices
2. How probability changes with sequence length
3. Context-dependent probability variation
"""

import torch
import tiktoken
from model import GPT

# Initialize the pre-trained GPT-2 model with dropout disabled for deterministic behavior
print("Loading GPT-2...")
model = GPT.from_pretrained('gpt2', dict(dropout=0.0))
model.eval()  # Set to evaluation mode (disables dropout, batch norm training behavior)

# Initialize the tokenizer for GPT-2 encoding/decoding
enc = tiktoken.get_encoding("gpt2")

# Test 1: Compare different continuations
# This test evaluates which city name the model assigns the highest probability
# when completing the sentence about France's capital. Expected: Paris > others

print("TEST 1: Comparing Different Continuations")
context = "The capital of France is"
# Convert context to token IDs and add batch dimension (unsqueeze creates shape [1, seq_len])
context_ids = torch.tensor(enc.encode(context)).unsqueeze(0)

# Different possible continuations to test
continuations = [" Paris", " London", " Berlin", " Rome", " Madrid"]

print(f"\nContext: '{context}'")
print("\nComparing probabilities of different continuations:\n")

# Calculate probability for each continuation
for cont in continuations:
    cont_ids = enc.encode(cont)
    # Generate with fixed_response to force specific continuation and get its probability
    output, prob = model.generate(
        context_ids,
        max_new_tokens=len(cont_ids),  # Generate exactly as many tokens as the continuation
        fixed_response=cont_ids,        # Force this specific continuation
        temperature=1.0                  # Use unmodified probabilities
    )
    full_text = enc.decode(output[0].tolist())
    print(f"  '{cont}': {prob:.8f}")

print("\nObservation: 'Paris' should have the highest probability.")

# Test 2: Effect of sequence length
# This test demonstrates how joint probability decreases as sequences get longer.
# Since P(A and B) = P(A) * P(B|A), longer sequences have compounding probabilities
# that multiply together, resulting in exponentially smaller values.

print("TEST 2: Effect of Sequence Length")
context = "Once upon a time"
context_ids = torch.tensor(enc.encode(context)).unsqueeze(0)

# Progressively longer continuations to show probability decay
sequences = [
    " there",
    " there was",
    " there was a",
    " there was a princess",
    " there was a princess who lived"
]

print(f"\nContext: '{context}'")
print("\nHow probability changes with sequence length:\n")
print(f"{'Length':<8} {'Probability':<20} {'Sequence'}")

# Calculate joint probability for sequences of increasing length
for seq in sequences:
    seq_ids = enc.encode(seq)
    output, prob = model.generate(
        context_ids,
        max_new_tokens=len(seq_ids),
        fixed_response=seq_ids,
        temperature=1.0
    )
    # Display token count, joint probability, and the actual sequence
    print(f"{len(seq_ids):<8} {prob:.12f}      {seq}")

print("\nObservation: Probability decreases exponentially with length.")

# Test 3: Different starting contexts
print("TEST 3: Same Continuation, Different Contexts")

contexts_and_cont = [
    ("I think the answer is", " yes"),
    ("The answer to your question is", " yes"),
    ("My response is definitely", " yes")
]

continuation = " yes"
cont_ids = enc.encode(continuation)

print(f"\nContinuation: '{continuation}'")
print("\nHow context affects probability of the same continuation:\n")

for context, cont in contexts_and_cont:
    context_ids = torch.tensor(enc.encode(context)).unsqueeze(0)
    output, prob = model.generate(
        context_ids,
        max_new_tokens=len(cont_ids),
        fixed_response=cont_ids,
        temperature=1.0
    )
    print(f"  Context: '{context}'")
    print(f"  Probability: {prob:.8f}\n")

print("\nObservation: Same word has different probabilities in different contexts.")

print("All tests complete!")