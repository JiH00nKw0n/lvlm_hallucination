ㅋㅋ"""
VISTA prompt preparation utilities.

Reference: VISTA/model_loader.py:364-430

VISTA uses null prompt (without image) as negative:
    - pos_kwargs: Full prompt with image
    - neg_kwargs: Prompt without image tokens (text-only)
"""

from typing import Dict, Optional

import torch
import torch.nn as nn


def prepare_pos_prompt(prev_kwargs: Dict) -> Dict:
    """
    Prepare positive prompt (with image).

    Reference: VISTA/model_loader.py:364-365

    For VISTA, positive prompt is just the original kwargs.

    Args:
        prev_kwargs: Original model kwargs with image

    Returns:
        Same kwargs (no modification needed)
    """
    return prev_kwargs


def prepare_neg_prompt(
    model: nn.Module,
    tokenizer,
    questions: list,
    model_type: str = "llava",
) -> Dict:
    """
    Prepare negative prompt (without image).

    Reference: VISTA/model_loader.py:368-430

    Creates a null prompt by removing image tokens.

    Args:
        model: The VLM model
        tokenizer: Model tokenizer
        questions: List of formatted questions (with image placeholders)
        model_type: Model type (llava, llava_next, qwen2_vl, qwen2_5_vl)

    Returns:
        kwargs dict for model forward without image
    """
    return prepare_null_prompt(model, tokenizer, questions, model_type)


def prepare_null_prompt(
    model: nn.Module,
    tokenizer,
    questions: list,
    model_type: str = "llava",
) -> Dict:
    """
    Create null prompt by removing image tokens from questions.

    Reference: VISTA/model_loader.py:372-430

    Args:
        model: The VLM model
        tokenizer: Model tokenizer
        questions: List of formatted questions (with image placeholders)
        model_type: Model type

    Returns:
        kwargs dict for model forward without image
    """
    if model_type in ["llava", "llava_next"]:
        return _prepare_llava_null_prompt(tokenizer, questions)
    elif model_type in ["qwen2_vl", "qwen2_5_vl"]:
        return _prepare_qwen_null_prompt(model, tokenizer, questions)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")


def _prepare_llava_null_prompt(tokenizer, questions: list) -> Dict:
    """
    Create null prompt for LLaVA models.

    Reference: VISTA/model_loader.py:385-430

    Removes <ImageHere> placeholder and concatenates text parts.
    """
    # Split by image placeholder
    chunks = [q.split("<ImageHere>") for q in questions]
    chunk_before = [chunk[0] for chunk in chunks]
    chunk_after = [chunk[1] if len(chunk) > 1 else "" for chunk in chunks]

    token_before = tokenizer(
        chunk_before,
        return_tensors="pt",
        padding="longest",
        add_special_tokens=False,
    ).input_ids.to("cuda")

    token_after = tokenizer(
        chunk_after,
        return_tensors="pt",
        padding="longest",
        add_special_tokens=False,
    ).input_ids.to("cuda")

    batch_size = len(questions)
    bos = (
        torch.ones([batch_size, 1], dtype=token_before.dtype, device=token_before.device)
        * tokenizer.bos_token_id
    )

    # Concatenate without image token
    neg_prompt = torch.cat([bos, token_before, token_after], dim=1)

    return {"input_ids": neg_prompt, "images": None}


def _prepare_qwen_null_prompt(
    model: nn.Module,
    processor,
    questions: list,
) -> Dict:
    """
    Create null prompt for Qwen2-VL models.

    Removes vision tokens and creates text-only input.
    """
    device = next(model.parameters()).device

    # For Qwen, we need to create text-only input
    # Remove image-related content from questions
    text_only = []
    for q in questions:
        # Remove image tags if present
        text = q.replace("<|vision_start|>", "").replace("<|vision_end|>", "")
        text = text.replace("<|image_pad|>", "")
        text_only.append(text.strip())

    # Tokenize text-only
    inputs = processor.tokenizer(
        text_only,
        return_tensors="pt",
        padding="longest",
    ).to(device)

    return {
        "input_ids": inputs.input_ids,
        "attention_mask": inputs.attention_mask,
    }


def prepare_vsv_kwargs_pair(
    model: nn.Module,
    tokenizer,
    questions: list,
    original_kwargs: Dict,
    model_type: str = "llava",
) -> list:
    """
    Prepare (neg_kwargs, pos_kwargs) pair for VSV computation.

    Args:
        model: The VLM model
        tokenizer: Model tokenizer
        questions: Formatted questions
        original_kwargs: Original model kwargs (with image)
        model_type: Model type

    Returns:
        [(neg_kwargs, pos_kwargs)] - single pair for obtain_vsv
    """
    pos_kwargs = prepare_pos_prompt(original_kwargs)
    neg_kwargs = prepare_neg_prompt(model, tokenizer, questions, model_type)

    return [(neg_kwargs, pos_kwargs)]
