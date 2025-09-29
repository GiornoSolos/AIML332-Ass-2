from datasets import load_dataset

print("Downloading TinyStories dataset...")
# Load a subset (the full dataset is huge)
dataset = load_dataset("roneneldan/TinyStories", split="train[:5000]")  # First 5000 stories

print(f"Loaded {len(dataset)} stories")

# Create data directory
import os
os.makedirs('data/tinystories', exist_ok=True)

# Save training text
output_file = 'data/tinystories/input.txt'
print(f"Writing to {output_file}...")

with open(output_file, 'w', encoding='utf-8') as f:
    for i, item in enumerate(dataset):
        f.write(item['text'] + '\n\n')
        if (i + 1) % 1000 == 0:
            print(f"Processed {i + 1} stories...")

print(f"Done! Saved {len(dataset)} stories to {output_file}")

# Show statistics
with open(output_file, 'r', encoding='utf-8') as f:
    content = f.read()
    print(f"\nFile size: {len(content) / 1024 / 1024:.2f} MB")
    print(f"Total characters: {len(content):,}")