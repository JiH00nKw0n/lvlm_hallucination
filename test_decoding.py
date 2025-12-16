"""
Evaluate decoding strategies on MME and run PCA diagnostics on Pico-Banana.

Strategies live in src/decoding/* and are toggled by CLI flags.
"""

import argparse
import json
import os
import random
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, LlavaConfig, LlavaForConditionalGeneration

from src.datasets.mme import MMEDatasetBuilder
from src.decoding.greedy import GreedyDecoder
from src.decoding.noise_contrastive import NoiseContrastiveDecoder
from src.decoding.rotation import InstructionRotationDecoder


def load_model_and_processor(model_name: str):
    processor = AutoProcessor.from_pretrained(model_name, load_in_8bit=True)

    config = LlavaConfig.from_pretrained(model_name)
    config.text_config.auto_map = {
        "AutoModel": "src.models.eval_llama.modeling_eval_llama.EvalLlamaModel",
        "AutoModelForCausalLM": "src.models.eval_llama.modeling_eval_llama.EvalLlamaForCausalLM",
    }

    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    device_map = "auto" if torch.cuda.is_available() else None

    model = LlavaForConditionalGeneration.from_pretrained(
        model_name,
        config=config,
        torch_dtype=torch_dtype,
        device_map=device_map,
        attn_implementation="eager",
    )
    model.eval()
    return model, processor


def build_prompt(processor, question: str, question_suffix: str) -> str:
    user_text = f"{question.strip()} {question_suffix}".strip()
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": user_text},
            ],
        }
    ]
    return processor.apply_chat_template(messages, add_generation_prompt=True)


def make_inputs(processor, prompt: str, image: Image.Image, device: torch.device):
    inputs = processor(images=image, text=prompt, return_tensors="pt")
    return {k: v.to(device) for k, v in inputs.items()}


class VibrantHueNoise:
    """
    Apply a random hue rotation plus saturation boost to create vivid noisy images.
    """

    def __init__(self, min_saturation: int = 160, max_saturation: int = 255):
        self.min_saturation = min_saturation
        self.max_saturation = max_saturation

    def __call__(self, image: Image.Image) -> Image.Image:
        hsv = image.convert("HSV")
        h, s, v = [np.array(ch, dtype=np.float32) for ch in hsv.split()]

        hue_shift = random.randint(0, 255)
        h = (h + hue_shift) % 255

        target_sat = random.uniform(self.min_saturation, self.max_saturation)
        s = np.clip(0.4 * s + target_sat, 0, 255)

        noisy = Image.merge(
            "HSV",
            (
                Image.fromarray(h.astype(np.uint8)),
                Image.fromarray(s.astype(np.uint8)),
                Image.fromarray(v.astype(np.uint8)),
            ),
        )
        return noisy.convert("RGB")


def normalize_yes_no(text: str) -> str:
    lowered = text.lower()
    if "yes" in lowered:
        return "yes"
    if "no" in lowered:
        return "no"
    return ""


def evaluate_mme(
    model,
    processor,
    strategies,
    *,
    split: str,
    max_samples: int,
    question_suffix: str,
    max_new_tokens: int,
    noise_scale: float,
    output_json: Optional[str] = None,
) -> None:
    tokenizer = processor.tokenizer
    device = next(model.parameters()).device
    builder = MMEDatasetBuilder(split=split)
    dataset = builder.build_dataset()

    total_samples = len(dataset)
    max_samples = total_samples if max_samples == -1 else min(max_samples, total_samples)

    noise_generator = VibrantHueNoise()

    correct_counts: Dict[str, Dict[str, int]] = {}
    totals_per_cat: Dict[str, int] = {}
    total_seen = 0

    for idx in tqdm(range(max_samples), desc="Evaluating MME"):
        sample = dataset[idx]
        image: Image.Image = sample["image"].convert("RGB")
        answer: str = sample["answer"].strip().lower()
        category: str = sample["category"]

        prompt = build_prompt(processor, sample["question"], question_suffix)
        clean_inputs = make_inputs(processor, prompt, image, device)

        noisy_inputs = None
        if any(strat.name == "noise_contrastive" for strat in strategies):
            noisy_image = noise_generator(image)
            noisy_inputs = make_inputs(processor, prompt, noisy_image, device)

        totals_per_cat[category] = totals_per_cat.get(category, 0) + 1
        total_seen += 1

        for strat in strategies:
            if strat.name == "noise_contrastive" and noisy_inputs is None:
                continue

            result = strat.decode(
                model,
                tokenizer,
                clean_inputs=clean_inputs,
                noisy_inputs=noisy_inputs,
                max_new_tokens=max_new_tokens,
                noise_scale=noise_scale,
            )
            pred = normalize_yes_no(result.text)
            correct = int(pred == answer)
            cat_counts = correct_counts.setdefault(strat.name, {})
            cat_counts[category] = cat_counts.get(category, 0) + correct

    summary = {
        "config": {
            "split": split,
            "max_samples": max_samples,
            "max_new_tokens": max_new_tokens,
            "noise_scale": noise_scale,
            "question_suffix": question_suffix,
        },
        "totals_per_category": totals_per_cat,
        "strategies": {},
        "total_seen": total_seen,
    }

    print("\n=== MME results ===")
    for strat in strategies:
        print(f"\nStrategy: {strat.name}")
        cat_counts = correct_counts.get(strat.name, {})
        total_correct = 0
        total_seen = 0
        per_cat = {}
        for cat, total_cat in totals_per_cat.items():
            c = cat_counts.get(cat, 0)
            acc = c / total_cat if total_cat else 0.0
            total_correct += c
            total_seen += total_cat
            print(f"  {cat:20s} acc={acc:.3f} ({c}/{total_cat})")
            per_cat[cat] = {"correct": c, "total": total_cat, "accuracy": acc}
        overall = total_correct / total_seen if total_seen else 0.0
        print(f"  Overall acc={overall:.3f} ({total_correct}/{total_seen})")
        summary["strategies"][strat.name] = {
            "overall": {"correct": total_correct, "total": total_seen, "accuracy": overall},
            "per_category": per_cat,
        }

    if output_json:
        out_dir = os.path.dirname(output_json)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"\nSaved summary to {output_json}")


def load_pico_entries(path: str) -> List[Dict]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} does not exist")
    entries: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))
    return entries


def compute_text_hidden_means(
    model,
    tokenizer,
    texts: List[str],
    batch_size: int = 2,
) -> List[List[torch.Tensor]]:
    device = next(model.parameters()).device
    all_means: List[List[torch.Tensor]] = []
    for start in tqdm(range(0, len(texts), batch_size), desc="Encoding text", leave=False):
        batch = texts[start : start + batch_size]
        tokenized = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        tokenized = {k: v.to(device) for k, v in tokenized.items()}
        attention = tokenized["attention_mask"].unsqueeze(-1)
        with torch.no_grad():
            outputs = model.model.language_model(
                **tokenized,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
        hidden_states = outputs.hidden_states[1:]  # drop embeddings
        for b_idx in range(len(batch)):
            layer_means: List[torch.Tensor] = []
            attn_mask = attention[b_idx]
            denom = attn_mask.sum(dim=0, keepdim=True).clamp(min=1)
            for layer_hidden in hidden_states:
                masked = layer_hidden[b_idx] * attn_mask
                mean_vec = masked.sum(dim=0) / denom
                layer_means.append(mean_vec.detach().cpu())
            all_means.append(layer_means)
    return all_means


def run_pca(matrix: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    matrix = matrix.float()
    mean = matrix.mean(dim=0, keepdim=True)
    centered = matrix - mean
    q = min(centered.shape[0], centered.shape[1])
    _, _, v = torch.pca_lowrank(centered, q=q)
    components = v.T
    return components, mean.squeeze(0)


def pca_on_pico_text(model, tokenizer, jsonl_path: str, output_dir: str, batch_size: int = 2) -> None:
    try:
        entries = load_pico_entries(jsonl_path)
    except FileNotFoundError as e:
        print(f"[PCA text] {e}")
        return
    instructions: List[str] = []
    for entry in entries:
        prompts = entry.get("metadata_edit_turn_prompts") or []
        instructions.extend(prompts)

    if not instructions:
        print("No instructions found for PCA text analysis.")
        return

    means = compute_text_hidden_means(model, tokenizer, instructions, batch_size=batch_size)
    num_layers = len(means[0])
    os.makedirs(output_dir, exist_ok=True)
    all_components = {}
    all_means = {}
    for layer_idx in range(num_layers):
        layer_stack = torch.stack([m[layer_idx] for m in means], dim=0)
        comps, mean_vec = run_pca(layer_stack)
        all_components[layer_idx] = comps
        all_means[layer_idx] = mean_vec
    torch.save({"components": all_components, "means": all_means}, os.path.join(output_dir, "text_pca.pt"))
    print(f"Saved text PCA components to {os.path.join(output_dir, 'text_pca.pt')}")


def pca_on_pico_images(model, processor, jsonl_path: str, output_dir: str) -> None:
    try:
        entries = load_pico_entries(jsonl_path)
    except FileNotFoundError as e:
        print(f"[PCA image] {e}")
        return
    device = next(model.parameters()).device
    diffs: List[torch.Tensor] = []

    for entry in tqdm(entries, desc="Pico image PCA"):
        local_input = entry.get("local_input_image")
        files = entry.get("files", [])
        final_path = None
        for f in files:
            if f.get("id") == "final_image":
                final_path = f.get("url")
                break
        if not local_input or not final_path:
            continue
        if not (os.path.exists(local_input) and os.path.exists(final_path)):
            continue

        try:
            orig_img = Image.open(local_input).convert("RGB")
            final_img = Image.open(final_path).convert("RGB")
        except Exception:
            continue

        with torch.no_grad():
            orig_inputs = processor(images=orig_img, text="", return_tensors="pt")
            final_inputs = processor(images=final_img, text="", return_tensors="pt")
            orig_feats = model.model.get_image_features(
                pixel_values=orig_inputs["pixel_values"].to(device),
                image_sizes=orig_inputs.get("image_sizes"),
            )
            final_feats = model.model.get_image_features(
                pixel_values=final_inputs["pixel_values"].to(device),
                image_sizes=final_inputs.get("image_sizes"),
            )
        orig_vec = torch.cat(orig_feats, dim=0).mean(dim=0).cpu()
        final_vec = torch.cat(final_feats, dim=0).mean(dim=0).cpu()
        diffs.append(final_vec - orig_vec)

    if not diffs:
        print("No image pairs with local files found; skipping image PCA.")
        return

    diff_tensor = torch.stack(diffs, dim=0)
    components, mean_vec = run_pca(diff_tensor)
    os.makedirs(output_dir, exist_ok=True)
    torch.save(
        {"components": components, "mean": mean_vec},
        os.path.join(output_dir, "image_pca.pt"),
    )
    print(f"Saved image PCA components to {os.path.join(output_dir, 'image_pca.pt')}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified decoding and PCA experiments")
    parser.add_argument("--model-name", type=str, default="llava-hf/llava-1.5-7b-hf")
    parser.add_argument("--split", type=str, default="test", help="MME split")
    parser.add_argument("--max-samples", type=int, default=100, help="Samples for MME (-1 for all)")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--noise-scale", type=float, default=0.5, help="Noise weight for contrastive decoding")
    parser.add_argument("--rotation-degrees", type=str, default="5,10,15", help="Comma-separated degrees")
    parser.add_argument("--question-suffix", type=str, default="Please answer with Yes or No.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-json", type=str, default="results/test_decoding_summary.json")
    parser.add_argument("--run-greedy", action="store_true")
    parser.add_argument("--run-noise-contrastive", action="store_true")
    parser.add_argument("--run-simple-rotation", action="store_true")
    parser.add_argument("--run-pca-text", action="store_true")
    parser.add_argument("--run-pca-image", action="store_true")
    parser.add_argument(
        "--pico-jsonl",
        type=str,
        default="pico_banana/multi_turn_with_local_source_image_path.jsonl",
    )
    parser.add_argument("--pca-output-dir", type=str, default="results/pico_pca")
    parser.add_argument("--text-batch-size", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    model, processor = load_model_and_processor(args.model_name)
    tokenizer = processor.tokenizer
    device = next(model.parameters()).device
    print(f"Loaded model on {device}")

    strategies = []
    if args.run_greedy:
        strategies.append(GreedyDecoder())
    if args.run_noise_contrastive:
        strategies.append(NoiseContrastiveDecoder(noise_scale=args.noise_scale))
    if args.run_simple_rotation:
        degrees = [float(x) for x in args.rotation_degrees.split(",") if x.strip()]
        for deg in degrees:
            strategies.append(InstructionRotationDecoder(degrees=deg, contrast_scale=1.0))

    if strategies:
        evaluate_mme(
            model=model,
            processor=processor,
            strategies=strategies,
            split=args.split,
            max_samples=args.max_samples,
            question_suffix=args.question_suffix,
            max_new_tokens=args.max_new_tokens,
            noise_scale=args.noise_scale,
            output_json=args.output_json,
        )
    else:
        print("No decoding strategies enabled; skipping MME evaluation.")

    if args.run_pca_text:
        pca_on_pico_text(
            model=model,
            tokenizer=tokenizer,
            jsonl_path=args.pico_jsonl,
            output_dir=args.pca_output_dir,
            batch_size=args.text_batch_size,
        )

    if args.run_pca_image:
        pca_on_pico_images(
            model=model,
            processor=processor,
            jsonl_path=args.pico_jsonl,
            output_dir=args.pca_output_dir,
        )


if __name__ == "__main__":
    main()
