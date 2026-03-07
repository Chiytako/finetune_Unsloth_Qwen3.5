"""
Test script to verify HuggingFace dataset access
"""

from datasets import load_dataset

print("=" * 50)
print("Testing HuggingFace Dataset Access")
print("=" * 50)

# Test 1: niconico_sft dataset
print("\n[1/2] Loading ChiTako/niconico_sft...")
try:
    sft_dataset = load_dataset(
        "ChiTako/niconico_sft",
        split="train",
        token=True,
    )
    print(f"  ✓ Success! Loaded {len(sft_dataset):,} rows")
    print(f"  Columns: {sft_dataset.column_names}")
    print(f"  Sample: {sft_dataset[0]}")
except Exception as e:
    print(f"  ✗ Error: {e}")

# Test 2: Qwen3.5-27b-ja dataset
print("\n[2/2] Loading ChiTako/Qwen3.5-27b-ja...")
try:
    reasoning_raw = load_dataset(
        "ChiTako/Qwen3.5-27b-ja",
        split="train",
        token=True,
    )
    print(f"  ✓ Success! Loaded {len(reasoning_raw):,} rows")
    print(f"  Columns: {reasoning_raw.column_names}")
    
    # Filter passed quality
    reasoning_filtered = reasoning_raw.filter(lambda x: x["quality"]["passed"])
    print(f"  After filter (passed): {len(reasoning_filtered):,} rows")
except Exception as e:
    print(f"  ✗ Error: {e}")

print("\n" + "=" * 50)
print("Test completed!")
print("=" * 50)
