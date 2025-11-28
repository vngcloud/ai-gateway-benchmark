#!/usr/bin/env python3
"""
Extract prompts from ShareGPT_V3 dataset for benchmarking.
Extracts the first human message from each conversation.
"""

import json
import random
import argparse
from pathlib import Path


def extract_prompts(input_file: str, output_file: str, num_prompts: int = 1000,
                    min_length: int = 20, max_length: int = 2000, seed: int = 42):
    """
    Extract prompts from ShareGPT dataset.

    Args:
        input_file: Path to ShareGPT JSON file
        output_file: Path to output JSONL file
        num_prompts: Number of prompts to extract
        min_length: Minimum prompt length (characters)
        max_length: Maximum prompt length (characters)
        seed: Random seed for reproducibility
    """
    print(f"Loading dataset from {input_file}...")

    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Loaded {len(data)} conversations")

    # Extract first human message from each conversation
    prompts = []
    for item in data:
        conversations = item.get('conversations', [])
        for conv in conversations:
            if conv.get('from') == 'human':
                prompt = conv.get('value', '').strip()
                # Filter by length
                if min_length <= len(prompt) <= max_length:
                    prompts.append(prompt)
                break  # Only take first human message per conversation

    print(f"Extracted {len(prompts)} valid prompts (length {min_length}-{max_length} chars)")

    # Randomly sample if we have more than needed
    random.seed(seed)
    if len(prompts) > num_prompts:
        prompts = random.sample(prompts, num_prompts)

    print(f"Selected {len(prompts)} prompts")

    # Write to JSONL format (only prompt, let benchmark.py use --max-tokens)
    with open(output_file, 'w', encoding='utf-8') as f:
        for prompt in prompts:
            entry = {"prompt": prompt}
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    print(f"Saved to {output_file}")

    # Print some statistics
    lengths = [len(p) for p in prompts]
    print(f"\nDataset Statistics:")
    print(f"  Total prompts: {len(prompts)}")
    print(f"  Avg length: {sum(lengths) / len(lengths):.0f} chars")
    print(f"  Min length: {min(lengths)} chars")
    print(f"  Max length: {max(lengths)} chars")

    # Show a few examples
    print(f"\nSample prompts:")
    for i, p in enumerate(prompts[:3]):
        preview = p[:100] + "..." if len(p) > 100 else p
        print(f"  {i+1}. {preview}")


def main():
    parser = argparse.ArgumentParser(description="Extract prompts from ShareGPT dataset")
    parser.add_argument("--input", "-i",
                        default="datasets/ShareGPT_V3_unfiltered_cleaned_split.json",
                        help="Input ShareGPT JSON file")
    parser.add_argument("--output", "-o",
                        default="benchmark_dataset.jsonl",
                        help="Output JSONL file")
    parser.add_argument("--num-prompts", "-n", type=int, default=1000,
                        help="Number of prompts to extract (default: 1000)")
    parser.add_argument("--min-length", type=int, default=20,
                        help="Minimum prompt length in chars (default: 20)")
    parser.add_argument("--max-length", type=int, default=2000,
                        help="Maximum prompt length in chars (default: 2000)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")

    args = parser.parse_args()

    extract_prompts(
        input_file=args.input,
        output_file=args.output,
        num_prompts=args.num_prompts,
        min_length=args.min_length,
        max_length=args.max_length,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
