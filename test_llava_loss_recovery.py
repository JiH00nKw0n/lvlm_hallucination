"""
Loss Recovery Score for SAE Reconstruction Quality.

Measures how well SAE reconstruction preserves downstream model behavior via CE loss:
    LR = (H0 - H*) / (H0 - H_orig)
where H_orig = original CE loss, H* = SAE-reconstructed CE loss, H0 = zero-ablated CE loss.

Two modes:
    1. Image Loss Recovery: replace image token hidden states with SAE reconstruction
    2. Text Loss Recovery: replace answer token hidden states with SAE reconstruction

Dataset: lmms-lab/llava-bench-coco (columns: image, question, answer)

Usage:
    python test_loss_recovery.py --num_samples 10 --k 256
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
from typing import Optional

import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch import Tensor
from tqdm import tqdm

from src.decoding.base import ModelHelper
from src.models.modeling_sae import (
    BatchTopKSAE,
    MatryoshkaSAE,
    TopKSAE,
    VLBatchTopKSAE,
    VLMatryoshkaSAE,
    VLTopKSAE,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Loss Recovery Score for SAE reconstruction")
    parser.add_argument("--model_name", type=str, default="llava-hf/llama3-llava-next-8b-hf")
    parser.add_argument("--sae_path", type=str, default="lmms-lab/llama3-llava-next-8b-hf-sae-131k")
    parser.add_argument("--layer_index", type=int, default=24)
    parser.add_argument("--num_samples", type=int, default=90)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--k", type=int, default=256)
    parser.add_argument("--output_dir", type=str, default="./results/loss_recovery")
    parser.add_argument("--dataset_name", type=str, default="lmms-lab/llava-bench-coco")
    return parser.parse_args()


def load_models(args: argparse.Namespace) -> tuple:
    """Load LLaVA-Next model, processor, and SAE. Returns (model, processor, sae, device)."""
    from transformers import AutoProcessor, LlavaNextForConditionalGeneration

    dtype = getattr(torch, args.dtype)
    device = torch.device(f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu")

    logger.info("Loading model: %s", args.model_name)
    model = LlavaNextForConditionalGeneration.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        attn_implementation="sdpa",
    )
    model.to(device)
    model.eval()
    model.requires_grad_(False)

    logger.info("Loading processor: %s", args.model_name)
    processor = AutoProcessor.from_pretrained(args.model_name, use_fast=True)

    logger.info("Loading SAE: %s", args.sae_path)
    if args.sae_path == "lmms-lab/llama3-llava-next-8b-hf-sae-131k":
        import sys
        _project_root = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, os.path.join(_project_root, "multimodal-sae", "train", "sae"))
        from sae.sae import Sae
        hookpoint = f"model.layers.{args.layer_index}"
        sae = Sae.load_from_hub(args.sae_path, hookpoint=hookpoint, device=device)
    else:
        from transformers import PretrainedConfig

        _SAE_ARCH_MAP = {
            "TopKSAE": TopKSAE,
            "VLTopKSAE": VLTopKSAE,
            "BatchTopKSAE": BatchTopKSAE,
            "VLBatchTopKSAE": VLBatchTopKSAE,
            "MatryoshkaSAE": MatryoshkaSAE,
            "VLMatryoshkaSAE": VLMatryoshkaSAE,
        }

        sae_config = PretrainedConfig.from_pretrained(args.sae_path)
        arch_name = getattr(sae_config, "architectures", ["TopKSAE"])[0]
        sae_cls = _SAE_ARCH_MAP.get(arch_name, TopKSAE)
        logger.info("Resolved SAE architecture: %s -> %s", arch_name, sae_cls.__name__)
        sae = sae_cls.from_pretrained(args.sae_path)
    logger.info("Setting SAE k = %d (user-specified)", args.k)
    sae.cfg.k = args.k
    sae.to(device)
    sae.eval()
    sae.requires_grad_(False)

    return model, processor, sae, device


def load_dataset_samples(args: argparse.Namespace):
    """Load llava-bench-coco dataset."""
    logger.info("Loading dataset: %s", args.dataset_name)
    return load_dataset(args.dataset_name, split="train")


def prepare_teacher_forcing_inputs(
    image,
    question: str,
    answer: str,
    processor,
    device: torch.device,
) -> Optional[dict]:
    """Build teacher-forcing inputs from image + question + answer.

    Returns dict with keys: input_ids, attention_mask, pixel_values, image_sizes,
    answer_start_idx, image_mask (bool tensor), answer_mask (bool tensor).
    Returns None on failure.
    """
    conversation = [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}
    ]
    question_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    full_prompt = question_prompt + answer

    try:
        full_inputs = processor(
            images=image, text=full_prompt, return_tensors="pt", return_mm_token_type_ids=True,
        )
        question_inputs = processor(images=image, text=question_prompt, return_tensors="pt")
    except Exception as e:
        logger.warning("Processor failed: %s", e)
        return None

    answer_start_idx = question_inputs["input_ids"].shape[1]

    mm_token_type_ids = full_inputs.get("mm_token_type_ids", None)
    if mm_token_type_ids is None:
        return None

    image_mask = mm_token_type_ids[0].bool()  # (seq_len,)
    seq_len = full_inputs["input_ids"].shape[1]
    answer_mask = torch.zeros(seq_len, dtype=torch.bool)
    answer_mask[answer_start_idx:] = True

    return {
        "input_ids": full_inputs["input_ids"].to(device),
        "attention_mask": full_inputs.get("attention_mask", torch.ones_like(full_inputs["input_ids"])).to(device),
        "pixel_values": full_inputs.get("pixel_values").to(device) if full_inputs.get("pixel_values") is not None else None,
        "image_sizes": full_inputs.get("image_sizes"),
        "answer_start_idx": answer_start_idx,
        "image_mask": image_mask,
        "answer_mask": answer_mask,
    }


def compute_ce_loss_on_answer(logits: Tensor, input_ids: Tensor, answer_start_idx: int) -> float:
    """Compute cross-entropy loss on answer tokens only (next-token prediction).

    shift_logits = logits[:, answer_start_idx - 1 : -1, :]  (predicting answer tokens)
    shift_labels = input_ids[:, answer_start_idx:]           (ground truth answer tokens)
    """
    vocab_size = logits.shape[-1]
    shift_logits = logits[:, answer_start_idx - 1 : -1, :].reshape(-1, vocab_size)
    shift_labels = input_ids[:, answer_start_idx:].reshape(-1)
    return F.cross_entropy(shift_logits.float(), shift_labels).item()


def make_sae_replacement_hook(sae, token_mask: Tensor):
    """Create a forward hook that replaces specified token hidden states with SAE reconstruction.

    Args:
        sae: SAE model with encode() and decode() methods.
        token_mask: Boolean mask of shape (seq_len,) indicating which tokens to replace.
    """
    def hook(module, input, output):
        # LlamaDecoderLayer returns a plain tensor, not a tuple
        hidden = output  # (B, seq_len, hidden_size)
        if not token_mask.any():
            return output
        mask = token_mask.to(hidden.device)
        tokens = hidden[:, mask, :]  # (B, n_selected, hidden_size)
        top_acts, top_indices = sae.encode(tokens)
        recon = sae.decode(top_acts, top_indices)
        modified = hidden.clone()
        modified[:, mask, :] = recon.to(hidden.dtype)
        return modified
    return hook


def make_zero_ablation_hook(token_mask: Tensor):
    """Create a forward hook that zeros out specified token hidden states.

    Args:
        token_mask: Boolean mask of shape (seq_len,) indicating which tokens to zero.
    """
    def hook(module, input, output):
        # LlamaDecoderLayer returns a plain tensor, not a tuple
        hidden = output  # (B, seq_len, hidden_size)
        if not token_mask.any():
            return output
        mask = token_mask.to(hidden.device)
        modified = hidden.clone()
        modified[:, mask, :] = 0.0
        return modified
    return hook


def run_forward_with_hook(
    model,
    inputs: dict,
    layer_index: int,
    hook_fn=None,
) -> float:
    """Run model forward pass with optional hook on target layer. Returns CE loss on answer tokens."""
    layers = ModelHelper.get_layers(model, "llava_next")
    handle = None
    if hook_fn is not None:
        handle = layers[layer_index].register_forward_hook(hook_fn)

    try:
        with torch.no_grad():
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                pixel_values=inputs["pixel_values"],
                image_sizes=inputs["image_sizes"],
                use_cache=False,
            )
        ce_loss = compute_ce_loss_on_answer(
            outputs.logits, inputs["input_ids"], inputs["answer_start_idx"],
        )
    finally:
        if handle is not None:
            handle.remove()

    return ce_loss


def process_sample(
    image,
    question: str,
    answer: str,
    model,
    processor,
    sae,
    layer_index: int,
    device: torch.device,
) -> Optional[dict]:
    """Process one sample: 5 forward passes (orig + image sae/zero + text sae/zero).

    Returns dict with image_lr, text_lr, and all CE losses, or None on failure.
    """
    try:
        image = image.convert("RGB")
    except Exception:
        return None

    inputs = prepare_teacher_forcing_inputs(image, question, answer, processor, device)
    if inputs is None:
        return None

    image_mask = inputs["image_mask"]
    answer_mask = inputs["answer_mask"]

    # 1. Original forward (shared for both modes)
    ce_orig = run_forward_with_hook(model, inputs, layer_index, hook_fn=None)

    # 2. Image Loss Recovery
    ce_sae_img = run_forward_with_hook(
        model, inputs, layer_index,
        hook_fn=make_sae_replacement_hook(sae, image_mask),
    )
    ce_zero_img = run_forward_with_hook(
        model, inputs, layer_index,
        hook_fn=make_zero_ablation_hook(image_mask),
    )
    denom_img = max(ce_zero_img - ce_orig, 1e-6)
    lr_img = (ce_zero_img - ce_sae_img) / denom_img

    # 3. Text Loss Recovery
    ce_sae_txt = run_forward_with_hook(
        model, inputs, layer_index,
        hook_fn=make_sae_replacement_hook(sae, answer_mask),
    )
    ce_zero_txt = run_forward_with_hook(
        model, inputs, layer_index,
        hook_fn=make_zero_ablation_hook(answer_mask),
    )
    denom_txt = max(ce_zero_txt - ce_orig, 1e-6)
    lr_txt = (ce_zero_txt - ce_sae_txt) / denom_txt

    return {
        "ce_orig": ce_orig,
        "image": {"ce_sae": ce_sae_img, "ce_zero": ce_zero_img, "score": lr_img},
        "text": {"ce_sae": ce_sae_txt, "ce_zero": ce_zero_txt, "score": lr_txt},
    }


def save_results(result: dict, output_dir: str, sae_path: str, k: int) -> str:
    """Save results to JSON. Returns the output file path."""
    os.makedirs(output_dir, exist_ok=True)
    sae_name = sae_path.rstrip("/").split("/")[-1]
    filename = f"LOSS_RECOVERY_{sae_name}_k{k}.json"
    filepath = os.path.join(output_dir, filename)
    with open(filepath, "w") as f:
        json.dump(result, f, indent=2)
    logger.info("Saved results to %s", filepath)
    return filepath


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    model, processor, sae, device = load_models(args)
    dataset = load_dataset_samples(args)

    num_samples = min(args.num_samples, len(dataset))
    indices = random.sample(range(len(dataset)), num_samples)

    image_per_sample: list[dict] = []
    text_per_sample: list[dict] = []
    skipped = 0

    for idx in tqdm(indices, desc="Processing samples"):
        sample = dataset[idx]
        result = process_sample(
            sample["image"],
            sample["question"],
            sample["answer"],
            model, processor, sae,
            args.layer_index, device,
        )
        if result is None:
            skipped += 1
            continue

        image_per_sample.append({
            "idx": idx,
            "score": result["image"]["score"],
            "ce_orig": result["ce_orig"],
            "ce_sae": result["image"]["ce_sae"],
            "ce_zero": result["image"]["ce_zero"],
        })
        text_per_sample.append({
            "idx": idx,
            "score": result["text"]["score"],
            "ce_orig": result["ce_orig"],
            "ce_sae": result["text"]["ce_sae"],
            "ce_zero": result["text"]["ce_zero"],
        })

    n = len(image_per_sample)
    logger.info("Done: %d processed, %d skipped", n, skipped)

    def summarize(per_sample: list[dict]) -> dict:
        if not per_sample:
            return {"mean_score": 0.0, "std_score": 0.0, "mean_ce_orig": 0.0, "mean_ce_sae": 0.0, "mean_ce_zero": 0.0, "per_sample": []}
        scores = torch.tensor([s["score"] for s in per_sample])
        ce_origs = torch.tensor([s["ce_orig"] for s in per_sample])
        ce_saes = torch.tensor([s["ce_sae"] for s in per_sample])
        ce_zeros = torch.tensor([s["ce_zero"] for s in per_sample])
        return {
            "mean_score": scores.mean().item(),
            "std_score": scores.std().item() if len(scores) > 1 else 0.0,
            "mean_ce_orig": ce_origs.mean().item(),
            "mean_ce_sae": ce_saes.mean().item(),
            "mean_ce_zero": ce_zeros.mean().item(),
            "per_sample": per_sample,
        }

    output = {
        "metadata": {
            "model": args.model_name,
            "sae_path": args.sae_path,
            "layer_index": args.layer_index,
            "k": args.k,
            "num_samples": num_samples,
            "skipped": skipped,
            "dataset": args.dataset_name,
        },
        "image_loss_recovery": summarize(image_per_sample),
        "text_loss_recovery": summarize(text_per_sample),
    }

    save_results(output, args.output_dir, args.sae_path, args.k)

    logger.info(
        "Image LR: %.4f +/- %.4f | Text LR: %.4f +/- %.4f",
        output["image_loss_recovery"]["mean_score"],
        output["image_loss_recovery"]["std_score"],
        output["text_loss_recovery"]["mean_score"],
        output["text_loss_recovery"]["std_score"],
    )


if __name__ == "__main__":
    main()
