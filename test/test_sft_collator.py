import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import requests
from PIL import Image

from transformers import AutoProcessor

# Test the LRVInstructForSFTImageCollator
from src.common.collator import LRVInstructForSFTImageCollator

model_id = "llava-hf/llava-1.5-7b-hf"

processor = AutoProcessor.from_pretrained(model_id)

# Load test images
image_file1 = "http://images.cocodataset.org/val2017/000000039769.jpg"
image_file2 = "http://images.cocodataset.org/train2017/000000391895.jpg"
raw_image1 = Image.open(requests.get(image_file1, stream=True).raw)
raw_image2 = Image.open(requests.get(image_file2, stream=True).raw)

# Create test dataset
features = [
    {
        "images": raw_image1,
        "prompt": "What are these?",
        "completion": "These are cats sleeping on a couch.",

    },
    {
        "images": raw_image2,
        "prompt": "What do you see?",
        "completion": "I see a stop sign.",
    }
]

# Initialize collator
collator = LRVInstructForSFTImageCollator(
    processor=processor,
    padding="longest",
    max_length=2048,
    truncation=False,
    return_tensors="pt"
)

# Test collation with debugging
print("=" * 80)
print("Testing LRVInstructForSFTImageCollator")
print("=" * 80)

# Add debugging to collator

images = [f["images"] for f in collator._convert_images(features)]
questions = [f["prompt"] for f in features]
answers = [f["completion"] for f in features]

batch = collator(features)

print("\nBatch keys:", batch.keys())
print("\nShapes:")
print(f"  input_ids: {batch['input_ids'].shape}")
print(f"  attention_mask: {batch['attention_mask'].shape}")
print(f"  pixel_values: {batch['pixel_values'].shape}")
print(f"  labels: {batch['labels'].shape}")

print("\n" + "=" * 80)
print("Sample 1:")
print("=" * 80)

# Decode first sample
decoded_input = processor.decode(batch['input_ids'][0], skip_special_tokens=False)
print("\nDecoded input_ids:")
print(decoded_input)

# Check labels
print("\nLabels (first 100 tokens):")
print(batch['labels'][0][:100])

# Count masked tokens
masked_count = (batch['labels'][0] == -100).sum().item()
total_count = (batch['attention_mask'][0] == 1).sum().item()
print(f"\nMasked tokens: {masked_count} / {total_count}")

# Decode only the non-masked part (the answer)
answer_tokens = batch['input_ids'][0][batch['labels'][0] != -100]
decoded_answer = processor.decode(answer_tokens, skip_special_tokens=False)
print("\nDecoded answer (non-masked part):")
print(decoded_answer)

# Decode the masked part (question)
masked_tokens = batch['input_ids'][0][batch['labels'][0] == -100]
decoded_masked = processor.decode(masked_tokens, skip_special_tokens=False)
print("\nDecoded masked part (question):")
print(decoded_masked)

print("\n" + "=" * 80)
print("Sample 2:")
print("=" * 80)

# Decode second sample
decoded_input = processor.decode(batch['input_ids'][1], skip_special_tokens=False)
print("\nDecoded input_ids:")
print(decoded_input)

# Check labels
print("\nLabels (first 100 tokens):")
print(batch['labels'][1][:100])

# Count masked tokens
masked_count = (batch['labels'][1] == -100).sum().item()
total_count = (batch['attention_mask'][1] == 1).sum().item()
print(f"\nMasked tokens: {masked_count} / {total_count}")

# Decode only the non-masked part (the answer)
answer_tokens = batch['input_ids'][1][batch['labels'][1] != -100]
decoded_answer = processor.decode(answer_tokens, skip_special_tokens=False)
print("\nDecoded answer (non-masked part):")
print(decoded_answer)

# Decode the masked part (question)
masked_tokens = batch['input_ids'][1][batch['labels'][1] == -100]
decoded_masked = processor.decode(masked_tokens, skip_special_tokens=False)
print("\nDecoded masked part (question):")
print(decoded_masked)

print("\n" + "=" * 80)
print("Testing complete!")
print("=" * 80)
