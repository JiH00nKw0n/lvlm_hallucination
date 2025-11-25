import math
import os
import random
from typing import List, Tuple

from datasets import load_dataset
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    pipeline,
)


def choose_mask_config() -> Tuple[int, int]:
    """Pick a mask percentage and span length."""
    percent = random.choice([5, 10, 15, 20, 25, 30, 35, 40, 45, 50])
    span = random.choice([1, 3, 5])
    return percent, span


def mask_tokens(
    input_ids: List[int],
    mask_id: int,
    percent: int,
    span: int,
) -> Tuple[List[int], List[int]]:
    """Mask roughly `percent` of tokens in contiguous spans of length `span`."""
    total = len(input_ids)
    target = max(1, math.ceil(total * (percent / 100)))
    masked_ids = list(input_ids)
    masked_positions = set()
    attempts = 0

    while len(masked_positions) < target and attempts < total * 5:
        start = random.randint(0, total - 1)
        for idx in range(start, min(start + span, total)):
            if len(masked_positions) >= target:
                break
            if idx in masked_positions:
                continue
            masked_ids[idx] = mask_id
            masked_positions.add(idx)
        attempts += 1

    return masked_ids, sorted(masked_positions)


def fill_masks(masked_text: str, mask_token: str, fill_pipe) -> str:
    """
    Iteratively fill mask tokens using the provided pipeline.

    The fill-mask pipeline handles one mask at a time, so we loop until none remain.
    """
    current = masked_text
    while mask_token in current:
        result = fill_pipe(current, top_k=1)
        if isinstance(result, list):
            result = result[0]
        replacement = result["token_str"]
        current = current.replace(mask_token, replacement, 1)
    return current


def main() -> None:
    random.seed()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    dataset = load_dataset("Lin-Chen/ShareGPT4V", split="train")
    sample = dataset[0]
    conversations = sample["conversations"]
    original = conversations[1]["value"]

    tokenizer = AutoTokenizer.from_pretrained("GSAI-ML/LLaDA-8B-Base")
    if tokenizer.mask_token is None or tokenizer.mask_token_id is None:
        raise ValueError("Tokenizer does not define a mask token.")

    percent, span = choose_mask_config()
    input_ids = tokenizer.encode(original, add_special_tokens=False)
    masked_ids, masked_positions = mask_tokens(
        input_ids,
        tokenizer.mask_token_id,
        percent,
        span,
    )

    masked_text = tokenizer.decode(masked_ids, skip_special_tokens=True)

    model = AutoModelForMaskedLM.from_pretrained("GSAI-ML/LLaDA-8B-Base")
    fill_pipe = pipeline(
        "fill-mask",
        model=model,
        tokenizer=tokenizer,
        device_map="auto",
    )

    generated = fill_masks(masked_text, tokenizer.mask_token, fill_pipe)

    print("\n=== Masking configuration ===")
    print(f"Mask percent: {percent}%")
    print(f"Mask span: {span} tokens\n")

    print("=== Original ===")
    print(original)
    print("\n=== Masked ===")
    print(masked_text)
    print("\n=== Generated ===")
    print(generated)


if __name__ == "__main__":
    main()
