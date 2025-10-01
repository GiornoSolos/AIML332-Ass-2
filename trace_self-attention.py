"""
Trace through self-attention mechanism for Task 3.1 report
Run this to get concrete examples for your report
"""

import torch
import torch.nn.functional as F
from model import GPT
import tiktoken

print("Tracing Self-Attention Mechanism")

# Load model
model = GPT.from_pretrained('gpt2', dict(dropout=0.0))
model.eval()

# Get first attention layer
attn_layer = model.transformer.h[0].attn
print(f"\nModel config:")
print(f"  Embedding dimension (n_embd): {model.config.n_embd}")
print(f"  Number of heads (n_head): {model.config.n_head}")
print(f"  Dimension per head: {model.config.n_embd // model.config.n_head}")

# Create sample input
B, T, C = 1, 5, 768  # Batch=1, Sequence=5, Embedding=768
x = torch.randn(B, T, C)

print("Step-by-step Self-trace:")

print(f"\n1. Input shape: {x.shape}")
print(f"   (Batch={B}, Sequence_Length={T}, Embedding_Dim={C})")

# Step 1: Compute Q, K, V
qkv = attn_layer.c_attn(x)
print(f"\n2. After c_attn (single linear layer for Q, K, V):")
print(f"   Shape: {qkv.shape}")
print(f"   (Produces 3 × embedding_dim = {3 * C} dimensions)")

# Split into Q, K, V
q, k, v = qkv.split(C, dim=2)
print(f"\n3. Split into Q, K, V:")
print(f"   Q shape: {q.shape}")
print(f"   K shape: {k.shape}")
print(f"   V shape: {v.shape}")

# Reshape for multi-head
n_head = model.config.n_head
head_dim = C // n_head

k = k.view(B, T, n_head, head_dim).transpose(1, 2)
q = q.view(B, T, n_head, head_dim).transpose(1, 2)
v = v.view(B, T, n_head, head_dim).transpose(1, 2)

print(f"\n4. Reshape for {n_head} heads:")
print(f"   Q shape: {q.shape}")
print(f"   K shape: {k.shape}")
print(f"   V shape: {v.shape}")
print(f"   (Batch, Heads, Sequence, Dim_per_head)")

# Compute attention scores
att = (q @ k.transpose(-2, -1)) * (1.0 / (head_dim ** 0.5))
print(f"\n5. Compute attention scores (Q @ K^T):")
print(f"   Shape: {att.shape}")
print(f"   (Batch, Heads, Sequence, Sequence)")
print(f"   Each position attends to every other position")
print(f"   Scaled by 1/√{head_dim} = {1.0 / (head_dim ** 0.5):.4f}")

# Show actual values for one head
print(f"\n   Attention scores for head 0 (before masking):")
head_0_scores = att[0, 0].detach()
print("   " + " ".join([f"Pos{i}" for i in range(T)]))
for i in range(T):
    row = "   " + " ".join([f"{head_0_scores[i,j]:6.2f}" for j in range(T)])
    print(f"Pos{i}: {row}")

# Apply causal mask
mask = torch.tril(torch.ones(T, T)).view(1, 1, T, T)
att = att.masked_fill(mask == 0, float('-inf'))

print(f"\n6. Apply causal mask (prevent attending to future):")
print(f"   Attention scores for head 0 (after masking):")
head_0_masked = att[0, 0].detach()
for i in range(T):
    row = "   " + " ".join([f"{head_0_masked[i,j]:6.1f}" if head_0_masked[i,j] != float('-inf') else "  -∞  " for j in range(T)])
    print(f"Pos{i}: {row}")

# Apply softmax
att = F.softmax(att, dim=-1)
print(f"\n7. Apply softmax (convert to probabilities):")
print(f"   Attention weights for head 0:")
head_0_weights = att[0, 0].detach()
print("   " + " ".join([f"Pos{i}" for i in range(T)]))
for i in range(T):
    row = "   " + " ".join([f"{head_0_weights[i,j]:5.3f}" for j in range(T)])
    print(f"Pos{i}: {row}")
    print(f"        Sum: {head_0_weights[i].sum():.3f} (should be 1.0)")

# Apply attention to values
y = att @ v
print(f"\n8. Apply attention to values (attention @ V):")
print(f"   Shape: {y.shape}")
print(f"   Each output is weighted combination of value vectors")

# Concatenate heads
y = y.transpose(1, 2).contiguous().view(B, T, C)
print(f"\n9. Concatenate all heads:")
print(f"   Shape: {y.shape}")
print(f"   (Merged {n_head} heads of {head_dim} dims = {C} dims)")

# Output projection
y_final = attn_layer.c_proj(y)
print(f"\n10. Output projection:")
print(f"    Shape: {y_final.shape}")
print(f"    Final transformation before residual connection")

print("\nSUMMARY")
print(f"""
The self-attention mechanism:
1. Projects input to Q, K, V using single linear layer
2. Splits into {n_head} parallel heads (each {head_dim} dims)
3. Computes attention scores (Q @ K^T / √d_k)
4. Applies causal mask (prevents attending to future)
5. Applies softmax (converts to probabilities)
6. Applies to values (attention @ V)
7. Concatenates heads back together
8. Final output projection

Key property: Each position can attend to all previous positions,
creating direct connections for long-range dependencies.
""")

print("\n" + "="*70)
print("Real example with text:")

enc = tiktoken.get_encoding("gpt2")
text = "The cat sat on the mat"
tokens = enc.encode(text)
token_strs = [enc.decode([t]) for t in tokens]

print(f"\nInput text: '{text}'")
print(f"Tokens: {token_strs}")
print(f"Token IDs: {tokens}")

# Get embeddings and run through first attention layer
input_ids = torch.tensor(tokens).unsqueeze(0)
with torch.no_grad():
    # Get token embeddings
    tok_emb = model.transformer.wte(input_ids)
    # Get position embeddings
    pos = torch.arange(0, len(tokens), dtype=torch.long)
    pos_emb = model.transformer.wpe(pos)
    # Combine
    x = tok_emb + pos_emb
    
    # Run through first attention layer
    attn_out = attn_layer(x)
    
    # Get attention weights from forward pass (need to modify forward to return them)
    # For now, just show the pattern
    
print(f"\nAttention pattern (conceptual):")
print("Each token attends to previous tokens:")
for i, token in enumerate(token_strs):
    can_attend_to = token_strs[:i+1]
    print(f"  '{token}' can attend to: {can_attend_to}")
