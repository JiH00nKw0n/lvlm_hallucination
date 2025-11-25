"""
Analyze logit-lens behavior of LLaVA on local Colored MNIST-style images.

For each test image we compare colored vs. grayscale variants by:
1. Running a forward pass with a fixed "What digit is shown?" prompt
2. Extracting the hidden state for every image patch token at every layer
3. Projecting those hidden states through the lm_head (logit lens)
4. Recording the top-k tokens + probabilities per (patch, layer)

Results are saved as a JSON list with entries shaped like:
{
    "idx": <dataset index>,
    "colored": {
        "patch_0": {
            "layer_0": [{"rank": 1, "token": "...", "prob": 0.52}, ...],
            ...
        },
        ...
    },
    "black_white": {...}
}
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import torch
from PIL import Image, ImageOps
from tqdm import tqdm
from transformers import AutoProcessor, LlavaConfig

from src.models.llava.modeling_llava import CustomLlavaForConditionalGeneration


LABEL_PREFIX_TO_LABEL = {
    "blue": 0,
    "green": 9,
    "red": 4,
}
TARGET_LABELS = set(LABEL_PREFIX_TO_LABEL.values())
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Logit-lens analysis on Colored MNIST")
    parser.add_argument(
        "--model-name",
        type=str,
        default="llava-hf/llava-1.5-7b-hf",
        help="Hugging Face model identifier for LLaVA.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="test_2",
        help="Directory containing local Colored MNIST images (PNG/JPG).",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=50,
        help="Number of samples to analyze (starting from start-index). Use -1 for whole split.",
    )
    parser.add_argument(
        "--samples-per-label",
        type=int,
        default=5,
        help="How many samples to keep for each label. Set to 0 to disable per-label filtering.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Offset into the dataset before sampling.",
    )
    parser.add_argument(
        "--question",
        type=str,
        default="What digit is shown in the image?",
        help="Question text inserted after <image> in the chat template.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="analysis_colored_mnist/logit_lens_results.json",
        help="Where to store the JSON output.",
    )
    parser.add_argument(
        "--image-output-dir",
        type=str,
        default=None,
        help="Directory to save colored / grayscale visualizations. Defaults to <output-dir>/images.",
    )
    parser.add_argument(
        "--max-patches",
        type=int,
        default=4,
        help="Maximum number of image patch tokens to analyze per sample (starting from the first patch token).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top tokens (by probability) to save per layer.",
    )
    parser.add_argument(
        "--image-token-id",
        type=int,
        default=32000,
        help="Token id used to denote an image patch in the text sequence.",
    )
    return parser.parse_args()


def infer_label_from_filename(filename: str) -> Optional[int]:
    lower = filename.lower()
    for prefix, label in LABEL_PREFIX_TO_LABEL.items():
        if lower.startswith(prefix):
            return label
    return None


def load_local_entries(data_dir: str) -> List[Dict]:
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory '{data_dir}' does not exist.")

    entries: List[Dict] = []
    for idx, path in enumerate(sorted(data_path.glob("*"))):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        label = infer_label_from_filename(path.name)
        if label is None:
            continue
        entries.append({"idx": idx, "path": path, "label": label})

    if not entries:
        raise RuntimeError(f"No valid image files found in '{data_dir}'.")

    return entries


def select_sample_entries(
    entries: List[Dict],
    start_index: int,
    num_samples: int,
    samples_per_label: int,
) -> List[Dict]:
    if start_index >= len(entries):
        raise ValueError(f"start-index {start_index} is >= dataset size {len(entries)}")

    subset = entries[start_index:]

    if samples_per_label <= 0:
        count = len(subset) if num_samples == -1 else min(num_samples, len(subset))
        return subset[:count]

    counts = defaultdict(int)
    selected: List[Dict] = []

    def quota_met() -> bool:
        return all(counts.get(label, 0) >= samples_per_label for label in TARGET_LABELS)

    for entry in subset:
        label = entry["label"]
        if counts[label] >= samples_per_label:
            continue
        counts[label] += 1
        selected.append(entry)

        if num_samples > 0 and len(selected) >= num_samples:
            break
        if quota_met():
            break

    return selected


def load_llava(model_name: str):
    """
    Load LLaVA with EvalLlama backend so we can request hidden states.
    """
    print(f"Loading processor + model: {model_name}")
    processor = AutoProcessor.from_pretrained(model_name)

    config = LlavaConfig.from_pretrained(model_name)
    config.text_config.auto_map = {
        "AutoModel": "src.models.eval_llama.modeling_eval_llama.EvalLlamaModel",
        "AutoModelForCausalLM": "src.models.eval_llama.modeling_eval_llama.EvalLlamaForCausalLM",
    }

    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    device_map = "auto" if torch.cuda.is_available() else None

    model = CustomLlavaForConditionalGeneration.from_pretrained(
        model_name,
        config=config,
        torch_dtype=torch_dtype,
        device_map=device_map,
        attn_implementation="eager",
    )
    model.eval()
    return model, processor


def make_prompt(processor, question: str) -> str:
    """
    Wrap the provided question using LLaVA's chat template with an image placeholder.
    """
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": ""},
            ],
        }
    ]
    return processor.apply_chat_template(messages, add_generation_prompt=True)


def prepare_inputs(processor, image: Image.Image, prompt: str, device: torch.device):
    inputs = processor(images=image, text=prompt, return_tensors="pt")
    return inputs.to(device)


def convert_to_grayscale(image: Image.Image) -> Image.Image:
    """
    Convert image to grayscale but keep 3 channels for CLIP encoder compatibility.
    """
    return ImageOps.grayscale(image).convert("RGB")


def find_patch_positions(input_ids: torch.Tensor, image_token_id: int) -> List[int]:
    """
    Locate token positions that correspond to image patches.
    """
    return [idx for idx, tid in enumerate(input_ids.tolist()) if tid == image_token_id]


def get_layerwise_topk(
    model,
    hidden_states: List[torch.Tensor],
    tokenizer,
    token_position: int,
    top_k: int,
) -> Dict[str, List[Dict[str, float]]]:
    """
    Apply the lm_head to the hidden state at a specific sequence position for each layer.
    """
    lm_head = model.lm_head
    layer_results: Dict[str, List[Dict[str, float]]] = {}

    for layer_idx, layer_hidden in enumerate(hidden_states):
        hidden_vec = layer_hidden[0, token_position, :]  # (hidden_dim,)
        logits = lm_head(hidden_vec)
        probs = torch.softmax(logits, dim=-1)
        topk = torch.topk(probs, k=top_k)

        entries = []
        for rank, (token_id, prob) in enumerate(zip(topk.indices.tolist(), topk.values.tolist()), start=1):
            decoded = tokenizer.decode([token_id]).strip()
            entries.append(
                {
                    "rank": rank,
                    "token": decoded,
                    "prob": float(prob),
                }
            )

        layer_results[f"layer_{layer_idx}"] = entries

    return layer_results


def analyze_variant(
    model,
    processor,
    image: Image.Image,
    prompt: str,
    image_token_id: int,
    max_patches: int,
    top_k: int,
) -> Dict[str, Dict[str, List[Dict[str, float]]]]:
    """
    Run a forward pass and collect layer-wise top-k tokens for the first N image patches.
    """
    model_device = next(model.parameters()).device
    prepared = prepare_inputs(processor, image, prompt, model_device)

    with torch.no_grad():
        outputs = model(
            **prepared,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )

    hidden_states = list(outputs.hidden_states)
    patch_positions = find_patch_positions(prepared["input_ids"][0].to("cpu"), image_token_id)

    if not patch_positions:
        raise RuntimeError("No image patch tokens found in the input sequence.")

    patch_dict: Dict[str, Dict[str, List[Dict[str, float]]]] = {}
    tokenizer = processor.tokenizer
    num_patches = min(len(patch_positions), max_patches if max_patches > 0 else len(patch_positions))

    for patch_idx in range(num_patches):
        seq_pos = patch_positions[patch_idx]
        patch_dict[f"patch_{patch_idx}"] = get_layerwise_topk(
            model=model,
            hidden_states=hidden_states,
            tokenizer=tokenizer,
            token_position=seq_pos,
            top_k=top_k,
        )

    return patch_dict


def main():
    args = parse_args()
    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    image_output_dir = args.image_output_dir
    if image_output_dir is None:
        image_output_dir = os.path.join(output_dir or ".", "images")
    os.makedirs(image_output_dir, exist_ok=True)

    model, processor = load_llava(args.model_name)
    prompt = make_prompt(processor, args.question)

    print(f"Loading local dataset from {args.data_dir}...")
    all_entries = load_local_entries(args.data_dir)

    selected_entries = select_sample_entries(
        entries=all_entries,
        start_index=args.start_index,
        num_samples=args.num_samples,
        samples_per_label=args.samples_per_label,
    )

    if not selected_entries:
        raise RuntimeError("No samples selected. Check sampling parameters.")

    print(
        f"Processing {len(selected_entries)} samples "
        f"(start_index={args.start_index}, samples_per_label={args.samples_per_label})"
    )

    results = []

    for entry in tqdm(selected_entries, desc="Colored MNIST samples"):
        ds_idx = entry["idx"]
        image = Image.open(entry["path"]).convert("RGB")
        grayscale_image = convert_to_grayscale(image)

        colored_path = os.path.join(image_output_dir, f"idx_{ds_idx}_colored.png")
        bw_path = os.path.join(image_output_dir, f"idx_{ds_idx}_black_white.png")
        image.save(colored_path)
        grayscale_image.save(bw_path)

        entry = {
            "idx": ds_idx,
            "colored": analyze_variant(
                model=model,
                processor=processor,
                image=image,
                prompt=prompt,
                image_token_id=args.image_token_id,
                max_patches=args.max_patches,
                top_k=args.top_k,
            ),
            "black_white": analyze_variant(
                model=model,
                processor=processor,
                image=grayscale_image,
                prompt=prompt,
                image_token_id=args.image_token_id,
                max_patches=args.max_patches,
                top_k=args.top_k,
            ),
        }

        results.append(entry)

    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(results)} entries to {args.output_path}")
    print(f"Images saved to {image_output_dir}")


if __name__ == "__main__":
    main()
