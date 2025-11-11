"""
Analyze the impact of color variations on LVLM object recognition.

Tests how different parameters affect the model's confidence (logprobs):
1. hue_range - Color gradient variation
2. blend_strength - Color blending intensity

Experiment:
1. Apply SAM segmentation + colorization with varying parameters
2. Prompt: "USER: <image>\n An image of ASSISTANT: "
3. Measure logprobs for the first predicted token (target object)
4. Average over 100 samples
5. Plot line charts for each parameter effect
6. Save random sample images and detailed JSON results
"""

import argparse
import json
import math
import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import SamModel, SamProcessor

from colorize_with_sam import get_sam_mask_auto
from test import CLASS_NAMES
from test.test_utils import colorize_subject


def load_vlm_model(model_name: str, device: str = "auto"):
    """Load vision-language model and processor with 8-bit quantization."""
    from transformers import LlavaForConditionalGeneration, AutoProcessor

    print(f"Loading VLM model: {model_name} (8-bit)")
    model = LlavaForConditionalGeneration.from_pretrained(
        model_name,
        load_in_8bit=True,
        device_map=device,
    )
    processor = AutoProcessor.from_pretrained(model_name)

    return model, processor


def load_sam_model(sam_model_name: str, device: str):
    """Load SAM model for segmentation."""
    print(f"Loading SAM model: {sam_model_name}")
    model = SamModel.from_pretrained(sam_model_name).to(device)
    processor = SamProcessor.from_pretrained(sam_model_name)
    return model, processor


def precompute_class_tokens(processor, all_class_names: dict) -> dict:
    """
    Precompute token IDs for all class names (without space prefix).

    Args:
        processor: VLM processor
        all_class_names: Dictionary of class names {class_id: class_name}

    Returns:
        Dictionary mapping class_id to list of token IDs
    """
    class_tokens_dict = {}
    for class_id, class_name in all_class_names.items():
        # No space prefix since "An image of " ends with a space
        tokens = processor.tokenizer.encode(class_name, add_special_tokens=False)
        class_tokens_dict[class_id] = tokens
    return class_tokens_dict


@torch.no_grad()
def get_logprobs_for_batch(
        model,
        processor,
        images: list,
        target_texts: list,
) -> list:
    """
    Get log probabilities for a batch of images with their target classes.

    Much more efficient than processing one image at a time with all classes.
    Processes N images x their respective target classes in one forward pass.

    Args:
        model: VLM model
        processor: VLM processor
        images: List of PIL Images
        target_texts: List of target class names (same length as images)

    Returns:
        List of log probabilities (one per image)
    """
    assert len(images) == len(target_texts), "Number of images and target texts must match"

    # Base prompt
    base_prompt = "USER: <image>\n Identify the name of the object. ASSISTANT: An image of"

    # Create batch of prompts (one per image with its target class)
    batch_prompts = []
    for target_text in target_texts:
        full_prompt = base_prompt + " " + target_text
        batch_prompts.append(full_prompt)

    # Set left padding for batch processing
    original_padding_side = processor.tokenizer.padding_side
    processor.tokenizer.padding_side = 'left'

    # Batch tokenization
    inputs = processor(
        text=batch_prompts,
        images=images,
        return_tensors="pt",
        padding=True
    )

    # Restore original padding side
    processor.tokenizer.padding_side = original_padding_side

    # Move to model device
    model_device = next(model.parameters()).device
    inputs = {k: v.to(model_device) for k, v in inputs.items()}

    # Batch forward pass
    outputs = model(use_cache=False, **inputs)
    logits = outputs.logits  # (batch_size, seq_len, vocab_size)

    # Get base prompt length (without padding) - use first image for reference
    base_inputs = processor(text=base_prompt, images=images[0], return_tensors="pt")
    base_length = base_inputs['input_ids'].shape[1]

    # Extract log probabilities for each image
    batch_log_probs = []
    pad_token_id = processor.tokenizer.pad_token_id

    for i, target_text in enumerate(target_texts):
        # Find actual sequence length (excluding padding)
        input_ids = inputs['input_ids'][i]
        seq_len = (input_ids != pad_token_id).sum().item()

        # Calculate where class tokens start and end
        padding_length = logits.shape[1] - seq_len
        class_start_pos = padding_length + base_length
        class_end_pos = padding_length + seq_len

        # Extract actual class tokens from the batch input
        actual_class_tokens = input_ids[class_start_pos:class_end_pos].tolist()

        if len(actual_class_tokens) == 0:
            batch_log_probs.append(-100.0)
            continue

        # Compute log probability for all tokens in class_name
        sum_log_prob = 0.0

        for j, token_id in enumerate(actual_class_tokens):
            # Position to get logits that predict this token
            logit_pos = class_start_pos + j - 1

            if logit_pos < 0 or logit_pos >= logits.shape[1]:
                sum_log_prob = -100.0
                break

            # Get log probability of this token
            log_probs = torch.log_softmax(logits[i, logit_pos, :], dim=-1)
            log_prob_token = log_probs[token_id].item()

            sum_log_prob += log_prob_token

        batch_log_probs.append(sum_log_prob)

    # Log summary
    print(f"[get_logprobs_for_batch] Processed {len(images)} images, "
          f"avg log P = {np.mean(batch_log_probs):.4f}")

    return batch_log_probs


@torch.no_grad()
def get_logprobs_for_token(
        model,
        processor,
        image,
        target_text: str,
        all_class_names: dict = None,
        class_tokens_dict: dict = None,
) -> float:
    """
    Get log probability of the target class for a single image.

    This is a wrapper around get_logprobs_for_batch for backward compatibility.

    Args:
        model: VLM model
        processor: VLM processor
        image: PIL Image
        target_text: Target class name (e.g., "apple")
        all_class_names: Unused (kept for compatibility)
        class_tokens_dict: Unused (kept for compatibility)

    Returns:
        Log probability of target class
    """
    batch_results = get_logprobs_for_batch(model, processor, [image], [target_text])
    return batch_results[0]


def evaluate_single_image_with_image(
        image_path: str,
        class_id: int,
        vlm_model,
        vlm_processor,
        sam_model,
        sam_processor,
        sam_device: str,
        class_tokens_dict: dict = None,
        hue_range: tuple = None,
        blend_strength: float = 0.0,
        grid_size: int = 5
):
    """
    Evaluate a single image and return both logprob and modified image.

    Returns:
        Tuple of (logprob, modified_image)
    """
    # Load original image
    img = Image.open(image_path).convert("RGB")

    # Get object name
    object_name = CLASS_NAMES.get(class_id, "object")

    # Generate mask with SAM (needed for colorization)
    if blend_strength > 0:
        mask = get_sam_mask_auto(img, sam_processor, sam_model, sam_device, grid_size=grid_size)
    else:
        mask = None

    # Apply modifications
    img_modified = img

    # Apply colorization if requested
    if blend_strength > 0 and hue_range is not None:
        img_modified = colorize_subject(
            img_modified, mask,
            saturation=230,
            hue_range=hue_range,
            blend_strength=blend_strength
        )

    # Get logprobs
    logprob = get_logprobs_for_token(
        vlm_model,
        vlm_processor,
        img_modified,
        object_name,
        class_tokens_dict=class_tokens_dict
    )

    return logprob, img_modified


def run_experiment(
        image_dir: str,
        vlm_model_name: str,
        sam_model_name: str,
        output_dir: str,
        num_samples: int = 100,
        device: str = "auto"
):
    """
    Run the main experiment.

    Args:
        image_dir: Directory containing input images (0.png, 1.png, ...)
        vlm_model_name: VLM model name
        sam_model_name: SAM model name
        output_dir: Output directory for results
        num_samples: Number of samples to evaluate
        device: Device for models
    """
    os.makedirs(output_dir, exist_ok=True)

    # Create subdirectories
    samples_dir = os.path.join(output_dir, "sample_images")
    os.makedirs(samples_dir, exist_ok=True)

    # Load models
    vlm_model, vlm_processor = load_vlm_model(vlm_model_name, device)

    # SAM device
    sam_device = "cuda" if torch.cuda.is_available() and device != "cpu" else "cpu"
    sam_model, sam_processor = load_sam_model(sam_model_name, sam_device)

    # Precompute class tokens once for efficiency
    print("Precomputing class tokens...")
    class_tokens_dict = precompute_class_tokens(vlm_processor, CLASS_NAMES)

    # Get image files
    image_path = Path(image_dir)
    image_files = sorted(image_path.glob("*.png"))[:num_samples]

    print(f"Found {len(image_files)} images")

    # Select random 5 samples for visualization
    random.seed(42)
    sample_indices = random.sample(range(len(image_files)), min(5, len(image_files)))
    sample_files = [image_files[i] for i in sample_indices]

    print(f"Selected {len(sample_files)} random samples for visualization: {[f.stem for f in sample_files]}")

    # Define parameter ranges
    hue_values = list(range(30, 181, 30))  # [30, 60, 90, 120, 150, 180]
    blend_values = [round(x * 0.2, 1) for x in range(6)]  # [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    print(f"Hue values: {hue_values}")
    print(f"Blend values: {blend_values}")

    # ========== Experiment 1: Vary hue_range (fix blend_strength=1.0) ==========
    print("\n" + "=" * 80)
    print("Experiment 1: Varying hue_range")
    print("=" * 80)

    hue_results = {hue: [] for hue in hue_values}
    hue_detailed = {hue: [] for hue in hue_values}  # Store (image_id, logprob)

    for hue_val in tqdm(hue_values, desc="Hue values"):
        hue_range = (hue_val, hue_val)

        # Load and process all images in batch
        batch_images = []
        batch_target_texts = []
        batch_class_ids = []
        batch_paths = []

        print(f"\nProcessing images for hue={hue_val}...")
        for img_path in tqdm(image_files, desc=f"Loading & modifying images (hue={hue_val})", leave=False):
            class_id = int(img_path.stem)

            try:
                # Load image
                img = Image.open(img_path).convert("RGB")

                # Generate mask and apply colorization
                mask = get_sam_mask_auto(img, sam_processor, sam_model, sam_device, grid_size=5)
                img_modified = colorize_subject(
                    img, mask,
                    saturation=230,
                    hue_range=hue_range,
                    blend_strength=1.0
                )

                # Save sample images
                if img_path in sample_files:
                    img_modified.save(os.path.join(samples_dir, f"exp1_hue{hue_val}_{img_path.stem}.png"))

                # Add to batch
                batch_images.append(img_modified)
                batch_target_texts.append(CLASS_NAMES[class_id])
                batch_class_ids.append(class_id)
                batch_paths.append(img_path)
            except Exception as e:
                print(f"Error processing {img_path.name}: {e}")
                continue

        # Batch inference
        print(f"Running batch inference for {len(batch_images)} images...")
        try:
            batch_logprobs = get_logprobs_for_batch(
                vlm_model,
                vlm_processor,
                batch_images,
                batch_target_texts
            )

            # Store results
            for class_id, logprob in zip(batch_class_ids, batch_logprobs):
                hue_results[hue_val].append(logprob)
                hue_detailed[hue_val].append({"image_id": class_id, "logprob": logprob})
        except Exception as e:
            print(f"Error in batch inference for hue={hue_val}: {e}")
            continue

    # Save detailed results
    with open(os.path.join(output_dir, "exp1_hue_detailed.json"), "w") as f:
        json.dump(hue_detailed, f, indent=2)

    # ========== Experiment 2: Vary blend_strength (fix hue_range=(180,180)) ==========
    print("\n" + "=" * 80)
    print("Experiment 2: Varying blend_strength")
    print("=" * 80)

    blend_results = {blend: [] for blend in blend_values}
    blend_detailed = {str(blend): [] for blend in blend_values}

    for blend_val in tqdm(blend_values, desc="Blend values"):
        # Load and process all images in batch
        batch_images = []
        batch_target_texts = []
        batch_class_ids = []
        batch_paths = []

        print(f"\nProcessing images for blend={blend_val}...")
        for img_path in tqdm(image_files, desc=f"Loading & modifying images (blend={blend_val})", leave=False):
            class_id = int(img_path.stem)

            try:
                # Load image
                img = Image.open(img_path).convert("RGB")

                # Apply colorization if blend_strength > 0
                if blend_val > 0:
                    mask = get_sam_mask_auto(img, sam_processor, sam_model, sam_device, grid_size=5)
                    img_modified = colorize_subject(
                        img, mask,
                        saturation=230,
                        hue_range=(180, 180),
                        blend_strength=blend_val
                    )
                else:
                    img_modified = img

                # Save sample images
                if img_path in sample_files:
                    img_modified.save(os.path.join(samples_dir, f"exp2_blend{blend_val}_{img_path.stem}.png"))

                # Add to batch
                batch_images.append(img_modified)
                batch_target_texts.append(CLASS_NAMES[class_id])
                batch_class_ids.append(class_id)
                batch_paths.append(img_path)
            except Exception as e:
                print(f"Error processing {img_path.name}: {e}")
                continue

        # Batch inference
        print(f"Running batch inference for {len(batch_images)} images...")
        try:
            batch_logprobs = get_logprobs_for_batch(
                vlm_model,
                vlm_processor,
                batch_images,
                batch_target_texts
            )

            # Store results
            for class_id, logprob in zip(batch_class_ids, batch_logprobs):
                blend_results[blend_val].append(logprob)
                blend_detailed[str(blend_val)].append({"image_id": class_id, "logprob": logprob})
        except Exception as e:
            print(f"Error in batch inference for blend={blend_val}: {e}")
            continue

    # Save detailed results
    with open(os.path.join(output_dir, "exp2_blend_detailed.json"), "w") as f:
        json.dump(blend_detailed, f, indent=2)

    # ========== Compute averages ==========
    print("\n" + "=" * 80)
    print("Computing averages...")
    print("=" * 80)

    # Hue results
    hue_avg = {hue: np.mean(logprobs) for hue, logprobs in hue_results.items() if len(logprobs) > 0}
    hue_std = {hue: np.std(logprobs) for hue, logprobs in hue_results.items() if len(logprobs) > 0}

    print("\nHue Range Results:")
    for hue in sorted(hue_avg.keys()):
        print(f"  Hue {hue:3d}: {hue_avg[hue]:.4f} ± {hue_std[hue]:.4f} (n={len(hue_results[hue])})")

    # Blend results
    blend_avg = {blend: np.mean(logprobs) for blend, logprobs in blend_results.items() if len(logprobs) > 0}
    blend_std = {blend: np.std(logprobs) for blend, logprobs in blend_results.items() if len(logprobs) > 0}

    print("\nBlend Strength Results:")
    for blend in sorted(blend_avg.keys()):
        print(f"  Blend {blend:.1f}: {blend_avg[blend]:.4f} ± {blend_std[blend]:.4f} (n={len(blend_results[blend])})")

    # ========== Plot results ==========
    print("\n" + "=" * 80)
    print("Plotting results...")
    print("=" * 80)

    # Plot 1: Hue range effect
    fig, ax = plt.subplots(figsize=(10, 6))
    hues = sorted(hue_avg.keys())
    avgs = [hue_avg[h] for h in hues]
    stds = [hue_std[h] for h in hues]
    ax.plot(hues, avgs, marker='o', linewidth=2, markersize=8, label='Mean logprob')
    ax.fill_between(hues, [a - s for a, s in zip(avgs, stds)], [a + s for a, s in zip(avgs, stds)], alpha=0.3)
    # Set y-axis limits based on avg values only (to nearest 0.25 multiple)
    y_min = math.floor(min(avgs) / 0.25) * 0.25
    y_max = math.ceil(max(avgs) / 0.25) * 0.25
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel('Hue Range Value', fontsize=12)
    ax.set_ylabel('Log Probability', fontsize=12)
    ax.set_title('Effect of Hue Range on Object Recognition\n(blend_strength=1.0)', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "hue_range_effect.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 2: Blend strength effect
    fig, ax = plt.subplots(figsize=(10, 6))
    blends = sorted(blend_avg.keys())
    avgs = [blend_avg[b] for b in blends]
    stds = [blend_std[b] for b in blends]
    ax.plot(blends, avgs, marker='o', linewidth=2, markersize=8, label='Mean logprob', color='coral')
    ax.fill_between(
        blends, [a - s for a, s in zip(avgs, stds)], [a + s for a, s in zip(avgs, stds)], alpha=0.3, color='coral'
        )
    # Set y-axis limits based on avg values only (to nearest 0.25 multiple)
    y_min = math.floor(min(avgs) / 0.25) * 0.25
    y_max = math.ceil(max(avgs) / 0.25) * 0.25
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel('Blend Strength', fontsize=12)
    ax.set_ylabel('Log Probability', fontsize=12)
    ax.set_title('Effect of Blend Strength on Object Recognition\n(hue_range=(180, 180))', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "blend_strength_effect.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # Save raw results
    results = {
        'hue_results': {k: v for k, v in hue_results.items()},
        'blend_results': {k: v for k, v in blend_results.items()},
        'hue_avg': hue_avg,
        'hue_std': hue_std,
        'blend_avg': blend_avg,
        'blend_std': blend_std,
    }

    torch.save(results, os.path.join(output_dir, "results.pt"))

    print(f"\nResults saved to {output_dir}/")
    print(f"  - hue_range_effect.png")
    print(f"  - blend_strength_effect.png")
    print(f"  - exp1_hue_detailed.json")
    print(f"  - exp2_blend_detailed.json")
    print(
        f"  - sample_images/ ({len(sample_files)} × 2 experiments = {len(sample_files) * 2 * len(hue_values) + len(sample_files) * len(blend_values)} images)"
        )
    print(f"  - results.pt")


def main():
    parser = argparse.ArgumentParser(description="Analyze color variations effect on LVLM")
    parser.add_argument(
        "--image_dir",
        type=str,
        default="images",
        help="Directory containing input images"
    )
    parser.add_argument(
        "--vlm_model",
        type=str,
        default="llava-hf/llava-1.5-7b-hf",
        help="Vision-language model name"
    )
    parser.add_argument(
        "--sam_model",
        type=str,
        default="facebook/sam-vit-base",
        help="SAM model name (base, large, huge)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="analysis_color_noise",
        help="Output directory"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=100,
        help="Number of samples to evaluate"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device for models (auto, cuda, cpu)"
    )

    args = parser.parse_args()

    run_experiment(
        image_dir=args.image_dir,
        vlm_model_name=args.vlm_model,
        sam_model_name=args.sam_model,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        device=args.device
    )


if __name__ == "__main__":
    main()
