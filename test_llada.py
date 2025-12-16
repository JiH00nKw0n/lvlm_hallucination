import argparse
import json
import math
import os
import random
from typing import List, Tuple

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


def choose_mask_config(
        percent_override: int | None = None,
        span_override: int | None = None,
) -> Tuple[int, int]:
    """Pick a mask percentage and span length (optionally overridable)."""
    percent_choices = [10, 15, 20, 25, 30, 35, 40, 45, 50]
    span_choices = [3, 5, 7]
    percent = percent_override if percent_override is not None else random.choice(percent_choices)
    span = span_override if span_override is not None else random.choice(span_choices)
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


def fill_masks_with_model(
    model,
    tokenizer,
    masked_ids: List[int],
    mask_token_id: int,
    device: torch.device,
) -> str:
    """Greedily fill all mask positions using model logits to avoid pipeline overhead."""
    model.eval()
    seq = torch.tensor(masked_ids, device=device).unsqueeze(0)
    with torch.no_grad():
        while (seq == mask_token_id).any():
            outputs = model(seq)
            logits = outputs.logits  # [1, seq_len, vocab]
            mask_positions = seq == mask_token_id
            filled_tokens = torch.argmax(logits[mask_positions], dim=-1)
            seq[mask_positions] = filled_tokens
    return tokenizer.decode(seq[0].tolist(), skip_special_tokens=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate multiple mask-and-fill captions for LLaDA-8B.")
    parser.add_argument("--sample-count", type=int, default=1, help="Number of dataset examples to process.")
    parser.add_argument(
        "--generations-per-sample",
        type=int,
        default=1,
        help="How many masked generations to create for each example.",
    )
    parser.add_argument("--mask-percent", type=int, default=None, help="Override mask percent for all runs.")
    parser.add_argument("--mask-span", type=int, default=None, help="Override mask span length for all runs.")
    parser.add_argument(
        "--mask-token",
        type=str,
        default=None,
        help="Manually specify the mask token string (e.g., '[gMASK]').",
    )
    parser.add_argument(
        "--mask-id",
        type=int,
        default=126336,
        help="Mask token id to fall back to (default 126336 as noted in LLaDA eval code).",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="llada_mask_generations.json",
        help="Path to save generated captions JSON.",
    )
    parser.add_argument(
        "--print-examples",
        type=int,
        default=1,
        help="How many originals to log with masked/generated examples.",
    )
    parser.add_argument(
        "--print-generations",
        type=int,
        default=3,
        help="How many generations per printed original to show.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed for reproducibility.")
    return parser.parse_args()


def resolve_mask_token(tokenizer) -> Tuple[str, int]:
    """
    Resolve the mask token; fall back to gmask_token if mask_token is missing.
    """
    token = getattr(tokenizer, "mask_token", None)
    token_id = getattr(tokenizer, "mask_token_id", None)

    if token is None or token_id is None:
        gmask = getattr(tokenizer, "gmask_token", None)
        if gmask is not None:
            token = gmask
            token_id = tokenizer.convert_tokens_to_ids(gmask)

    if token is None or token_id is None:
        raise ValueError("Tokenizer does not define a usable mask token (mask_token or gmask_token).")

    return token, token_id


def main() -> None:
    args = parse_args()
    if args.seed is None:
        random.seed()
    else:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    dataset = load_dataset(
        "Lin-Chen/ShareGPT4V",
        "ShareGPT4V",
        split="train",
    )

    if args.sample_count > len(dataset):
        raise ValueError(f"Requested {args.sample_count} samples but dataset only has {len(dataset)} rows.")

    chosen_indices = random.sample(range(len(dataset)), k=args.sample_count)

    tokenizer = AutoTokenizer.from_pretrained(
        "GSAI-ML/LLaDA-8B-Base",
        trust_remote_code=True,
    )
    if args.mask_token:
        mask_token = args.mask_token
        mask_token_id = tokenizer.convert_tokens_to_ids(mask_token)
        if mask_token_id is None:
            raise ValueError(f"Provided mask token '{mask_token}' not found in tokenizer vocab.")
    else:
        try:
            mask_token, mask_token_id = resolve_mask_token(tokenizer)
        except ValueError:
            mask_token_id = args.mask_id
            mask_token = tokenizer.convert_ids_to_tokens(mask_token_id)
            if mask_token is None:
                raise ValueError(
                    "Could not resolve mask token automatically; please pass --mask-token or a valid --mask-id."
                )
    tokenizer.mask_token = mask_token
    tokenizer.mask_token_id = mask_token_id
    tokenizer.special_tokens_map["mask_token"] = mask_token
    if hasattr(tokenizer, "special_tokens_map_extended"):
        tokenizer.special_tokens_map_extended["mask_token"] = mask_token

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModel.from_pretrained(
        "GSAI-ML/LLaDA-8B-Base",
        trust_remote_code=True,
        torch_dtype=dtype,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    results: dict[str, dict[str, List[str]]] = {}
    for out_idx, ds_idx in enumerate(tqdm(chosen_indices, desc="Samples")):
        sample = dataset[int(ds_idx)]
        conversations = sample["conversations"]
        original = conversations[1]["value"]
        input_ids = tokenizer.encode(original, add_special_tokens=False)

        generated_captions: List[str] = []
        for gen_idx in tqdm(range(args.generations_per_sample), leave=False, desc="Generations"):
            percent, span = choose_mask_config(args.mask_percent, args.mask_span)
            masked_ids, _ = mask_tokens(
                input_ids,
                mask_token_id,
                percent,
                span,
            )
            masked_text = tokenizer.decode(masked_ids, skip_special_tokens=False)
            generated = fill_masks_with_model(
                model,
                tokenizer,
                masked_ids,
                mask_token_id,
                device,
            )
            generated_captions.append(generated)

            if out_idx < args.print_examples and gen_idx < args.print_generations:
                print("\n------------------------------")
                print(f"[Sample {out_idx} | Generation {gen_idx}]")
                print(f"Mask percent: {percent}%, span: {span}")
                print("Original :")
                print(original)
                print("\nMasked   :")
                print(masked_text)
                print("\nGenerated:")
                print(generated)
                print("------------------------------\n")


if __name__ == "__main__":
    main()
