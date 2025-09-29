import torch
import tiktoken
from model import GPT

print("Loading GPT-2...")
model = GPT.from_pretrained('gpt2', dict(dropout=0.0))
model.eval()

enc = tiktoken.get_encoding("gpt2")

# Test 1: Compare different continuations
print("\n" + "="*70)
print("TEST 1: Comparing Different Continuations")
print("="*70)
context = "The capital of France is"
context_ids = torch.tensor(enc.encode(context)).unsqueeze(0)

continuations = [" Paris", " London", " Berlin", " Rome", " Madrid"]

print(f"\nContext: '{context}'")
print("\nComparing probabilities of different continuations:\n")

for cont in continuations:
    cont_ids = enc.encode(cont)
    output, prob = model.generate(
        context_ids,
        max_new_tokens=len(cont_ids),
        fixed_response=cont_ids,
        temperature=1.0
    )
    full_text = enc.decode(output[0].tolist())
    print(f"  '{cont}': {prob:.8f}")

print("\nObservation: 'Paris' should have the highest probability.")

# Test 2: Effect of sequence length
print("\n" + "="*70)
print("TEST 2: Effect of Sequence Length")
print("="*70)
context = "Once upon a time"
context_ids = torch.tensor(enc.encode(context)).unsqueeze(0)

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
print("-" * 70)

for seq in sequences:
    seq_ids = enc.encode(seq)
    output, prob = model.generate(
        context_ids,
        max_new_tokens=len(seq_ids),
        fixed_response=seq_ids,
        temperature=1.0
    )
    print(f"{len(seq_ids):<8} {prob:.12f}      {seq}")

print("\nObservation: Probability decreases exponentially with length.")

# Test 3: Different starting contexts
print("\n" + "="*70)
print("TEST 3: Same Continuation, Different Contexts")
print("="*70)

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

print("\n" + "="*70)
print("All tests complete!")
print("="*70)