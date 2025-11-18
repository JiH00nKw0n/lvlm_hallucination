"""
Compare three decoding strategies on the MME color subset:
1. Standard greedy decoding on the clean image
2. Simple contrastive greedy decoding between clean and noisy images
3. VCD-style decoding (adaptive plausibility constraints) using noisy images

Noise is a vibrant hue shift so every noisy sample stays highly saturated.
"""

import argparse
import json
import os
import random
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, LlavaConfig

from src.datasets.mme import MMEDatasetBuilder
from src.models.llava.modeling_llava import CustomLlavaForConditionalGeneration


@dataclass
class DecodeResult:
    text: str
    normalized: str
    correct: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Contrastive decoding test on MME color subset")
    parser.add_argument(
        "--model-name",
        type=str,
        default="llava-hf/llava-1.5-7b-hf",
        help="Hugging Face identifier of the LLaVA checkpoint.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to use for MME.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=100,
        help="Maximum number of samples from the color subset. Use -1 for all.",
    )
    parser.add_argument(
        "--noise-scale",
        type=float,
        default=0.5,
        help="Coefficient applied to noisy logits when computing contrastive scores.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=32,
        help="Maximum number of tokens to decode.",
    )
    parser.add_argument(
        "--noise-mode",
        type=str,
        default="vibrant_hue",
        choices=["vibrant_hue"],
        help="Type of noise to apply to images (currently only vibrant hue shift).",
    )
    parser.add_argument(
        "--question-suffix",
        type=str,
        default="Please answer with Yes or No.",
        help="Suffix appended to each question in the prompt.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional path to save per-sample details as JSON.",
    )
    parser.add_argument(
        "--image-output-dir",
        type=str,
        default="results/mme_color_contrast_images",
        help="Directory to save clean / noisy image variants.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for noise selection.",
    )
    parser.add_argument(
        "--cd-alpha",
        type=float,
        default=0.5,
        help="Alpha parameter used in VCD to mix clean/noisy logits.",
    )
    parser.add_argument(
        "--cd-beta",
        type=float,
        default=0.1,
        help="Beta parameter controlling adaptive plausibility constraints.",
    )
    return parser.parse_args()


def load_model_and_processor(model_name: str):
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


class DiffusionNoiseGenerator:
    """
    Generate VCD-style diffusion noise samples by simulating q(x_t | x_0, t).
    """

    def __init__(self, num_steps: int = 1000, beta_start: float = 1e-5, beta_end: float = 5e-3):
        steps = torch.linspace(-6, 6, num_steps)
        betas = torch.sigmoid(steps) * (beta_end - beta_start) + beta_start
        alphas = 1 - betas
        self.alphas_bar = torch.cumprod(alphas, dim=0)
        self.sqrt_alphas = torch.sqrt(self.alphas_bar)
        self.sqrt_one_minus = torch.sqrt(1 - self.alphas_bar)
        self.num_steps = num_steps

    def add_noise(self, image_tensor: torch.Tensor, step: int) -> torch.Tensor:
        noise = torch.randn_like(image_tensor)
        alpha = self.sqrt_alphas[step].to(image_tensor.device)
        one_minus = self.sqrt_one_minus[step].to(image_tensor.device)
        return alpha * image_tensor + one_minus * noise

    def __call__(self, image: Image.Image, step: Optional[int] = None) -> Tuple[Image.Image, int]:
        if step is None:
            step = random.randint(0, self.num_steps - 1)
        arr = np.array(image).astype(np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        noisy = self.add_noise(tensor, step)
        noisy = torch.clamp(noisy, 0.0, 1.0)
        noisy_arr = (noisy.squeeze(0).permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
        return Image.fromarray(noisy_arr), step


def greedy_decode(model, inputs: Dict[str, torch.Tensor], max_new_tokens: int, tokenizer) -> str:
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
    )
    new_tokens = generated_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def contrastive_greedy_decode(
    model,
    clean_inputs: Dict[str, torch.Tensor],
    noisy_inputs: Dict[str, torch.Tensor],
    noise_scale: float,
    max_new_tokens: int,
    tokenizer,
) -> str:
    clean_ids = clean_inputs["input_ids"]
    noisy_ids = noisy_inputs["input_ids"]
    clean_pixels = clean_inputs.get("pixel_values")
    noisy_pixels = noisy_inputs.get("pixel_values")
    clean_mask = clean_inputs.get("attention_mask")
    noisy_mask = noisy_inputs.get("attention_mask")

    eos_token_id = tokenizer.eos_token_id or tokenizer.pad_token_id
    generated_tokens = []

    for _ in range(max_new_tokens):
        clean_kwargs = {
            "input_ids": clean_ids,
            "pixel_values": clean_pixels,
            "attention_mask": clean_mask,
            "use_cache": False,
            "return_dict": True,
        }
        noisy_kwargs = {
            "input_ids": noisy_ids,
            "pixel_values": noisy_pixels,
            "attention_mask": noisy_mask,
            "use_cache": False,
            "return_dict": True,
        }

        clean_logits = model(**clean_kwargs).logits[:, -1, :]
        noisy_logits = model(**noisy_kwargs).logits[:, -1, :]

        contrastive_logits = clean_logits - noise_scale * noisy_logits
        next_token = torch.argmax(contrastive_logits, dim=-1, keepdim=True)

        generated_tokens.append(next_token)
        clean_ids = torch.cat([clean_ids, next_token], dim=-1)
        noisy_ids = torch.cat([noisy_ids, next_token], dim=-1)
        if clean_mask is not None:
            clean_mask = torch.cat(
                [clean_mask, torch.ones_like(next_token, device=clean_mask.device)],
                dim=-1,
            )
        if noisy_mask is not None:
            noisy_mask = torch.cat(
                [noisy_mask, torch.ones_like(next_token, device=noisy_mask.device)],
                dim=-1,
            )

        if eos_token_id is not None and next_token.item() == eos_token_id:
            break

    if not generated_tokens:
        return ""

    gen_ids = torch.cat(generated_tokens, dim=-1)
    return tokenizer.decode(gen_ids[0], skip_special_tokens=True).strip()


def vcd_decode(
    model,
    clean_inputs: Dict[str, torch.Tensor],
    noisy_inputs: Dict[str, torch.Tensor],
    max_new_tokens: int,
    tokenizer,
    cd_alpha: float,
    cd_beta: float,
) -> str:
    clean_ids = clean_inputs["input_ids"]
    noisy_ids = noisy_inputs["input_ids"]
    clean_pixels = clean_inputs.get("pixel_values")
    noisy_pixels = noisy_inputs.get("pixel_values")
    clean_mask = clean_inputs.get("attention_mask")
    noisy_mask = noisy_inputs.get("attention_mask")

    eos_token_id = tokenizer.eos_token_id or tokenizer.pad_token_id
    generated_tokens = []

    log_beta = torch.log(torch.tensor(cd_beta, device=clean_ids.device))

    for _ in range(max_new_tokens):
        clean_kwargs = {
            "input_ids": clean_ids,
            "pixel_values": clean_pixels,
            "attention_mask": clean_mask,
            "use_cache": False,
            "return_dict": True,
        }
        noisy_kwargs = {
            "input_ids": noisy_ids,
            "pixel_values": noisy_pixels,
            "attention_mask": noisy_mask,
            "use_cache": False,
            "return_dict": True,
        }

        clean_logits = model(**clean_kwargs).logits[:, -1, :]
        noisy_logits = model(**noisy_kwargs).logits[:, -1, :]

        cutoff = log_beta + clean_logits.max(dim=-1, keepdim=True).values
        diffs = (1 + cd_alpha) * clean_logits - cd_alpha * noisy_logits
        diffs = diffs.masked_fill(clean_logits < cutoff, -float("inf"))

        next_token = torch.argmax(diffs, dim=-1, keepdim=True)
        generated_tokens.append(next_token)

        clean_ids = torch.cat([clean_ids, next_token], dim=-1)
        noisy_ids = torch.cat([noisy_ids, next_token], dim=-1)
        if clean_mask is not None:
            clean_mask = torch.cat([clean_mask, torch.ones_like(next_token)], dim=-1)
        if noisy_mask is not None:
            noisy_mask = torch.cat([noisy_mask, torch.ones_like(next_token)], dim=-1)

        if eos_token_id is not None and next_token.item() == eos_token_id:
            break

    if not generated_tokens:
        return ""

    gen_ids = torch.cat(generated_tokens, dim=-1)
    return tokenizer.decode(gen_ids[0], skip_special_tokens=True).strip()


def normalize_yes_no(text: str) -> str:
    lowered = text.lower()
    if "yes" in lowered:
        return "yes"
    if "no" in lowered:
        return "no"
    return ""


def evaluate_prediction(text: str, answer: str) -> DecodeResult:
    normalized = normalize_yes_no(text)
    correct = normalized == answer
    return DecodeResult(text=text, normalized=normalized, correct=correct)


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    model, processor = load_model_and_processor(args.model_name)
    tokenizer = processor.tokenizer
    device = next(model.parameters()).device

    builder = MMEDatasetBuilder(split=args.split)
    dataset = builder.build_dataset()
    color_dataset = dataset.filter(lambda ex: ex["category"] == "color")

    total_samples = len(color_dataset)
    max_samples = total_samples if args.max_samples == -1 else min(args.max_samples, total_samples)

    noise_generator = VibrantHueNoise()
    diffusion_noise = DiffusionNoiseGenerator()
    os.makedirs(args.image_output_dir, exist_ok=True)

    results: List[Dict] = []
    baseline_correct = 0
    contrastive_correct = 0
    vcd_correct = 0

    for idx in tqdm(range(max_samples), desc="Evaluating color samples"):
        sample = color_dataset[idx]
        image: Image.Image = sample["image"].convert("RGB")
        answer: str = sample["answer"].strip().lower()

        hue_noisy_image = noise_generator(image)
        vcd_noisy_image, vcd_step = diffusion_noise(image)

        question_id = sample.get("question_id") or f"sample_{idx}"
        safe_qid = re.sub(r"[^a-zA-Z0-9_-]", "_", str(question_id))
        clean_path = os.path.join(args.image_output_dir, f"{safe_qid}_clean.png")
        hue_path = os.path.join(args.image_output_dir, f"{safe_qid}_hue.png")
        vcd_path = os.path.join(args.image_output_dir, f"{safe_qid}_vcd.png")
        image.save(clean_path)
        hue_noisy_image.save(hue_path)
        vcd_noisy_image.save(vcd_path)

        prompt = build_prompt(processor, sample["question"], args.question_suffix)
        clean_inputs = make_inputs(processor, prompt, image, device)
        hue_inputs = make_inputs(processor, prompt, hue_noisy_image, device)
        vcd_inputs = make_inputs(processor, prompt, vcd_noisy_image, device)

        baseline_text = greedy_decode(model, clean_inputs, args.max_new_tokens, tokenizer)
        contrastive_text = contrastive_greedy_decode(
            model=model,
            clean_inputs=clean_inputs,
            noisy_inputs=hue_inputs,
            noise_scale=args.noise_scale,
            max_new_tokens=args.max_new_tokens,
            tokenizer=tokenizer,
        )
        vcd_text = vcd_decode(
            model=model,
            clean_inputs=clean_inputs,
            noisy_inputs=vcd_inputs,
            max_new_tokens=args.max_new_tokens,
            tokenizer=tokenizer,
            cd_alpha=args.cd_alpha,
            cd_beta=args.cd_beta,
        )

        baseline_result = evaluate_prediction(baseline_text, answer)
        contrastive_result = evaluate_prediction(contrastive_text, answer)
        vcd_result = evaluate_prediction(vcd_text, answer)

        baseline_correct += int(baseline_result.correct)
        contrastive_correct += int(contrastive_result.correct)
        vcd_correct += int(vcd_result.correct)

        results.append(
            {
                "idx": idx,
                "question_id": question_id,
                "question": sample["question"],
                "answer": answer,
                "noise_mode": args.noise_mode,
                "vcd_noise_step": vcd_step,
                "image_paths": {
                    "clean": clean_path,
                    "hue_noise": hue_path,
                    "vcd_noise": vcd_path,
                },
                "baseline_text": baseline_result.text,
                "baseline_pred": baseline_result.normalized,
                "baseline_correct": baseline_result.correct,
                "contrastive_text": contrastive_result.text,
                "contrastive_pred": contrastive_result.normalized,
                "contrastive_correct": contrastive_result.correct,
                "vcd_text": vcd_result.text,
                "vcd_pred": vcd_result.normalized,
                "vcd_correct": vcd_result.correct,
            }
        )

    total = len(results)
    baseline_acc = baseline_correct / total if total else 0.0
    contrastive_acc = contrastive_correct / total if total else 0.0
    vcd_acc = vcd_correct / total if total else 0.0

    print("=== MME Color Contrastive Decoding Results ===")
    print(f"Samples evaluated: {total}")
    print(f"Baseline greedy accuracy:     {baseline_acc:.3f}")
    print(f"Contrastive greedy accuracy:  {contrastive_acc:.3f}")
    print(f"VCD accuracy:                 {vcd_acc:.3f}")
    print(f"Image variants saved to:      {args.image_output_dir}")

    if args.output_json:
        output_dir = os.path.dirname(args.output_json)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "config": {
                        "model_name": args.model_name,
                        "max_samples": args.max_samples,
                        "noise_scale": args.noise_scale,
                        "max_new_tokens": args.max_new_tokens,
                        "noise_mode": args.noise_mode,
                        "cd_alpha": args.cd_alpha,
                        "cd_beta": args.cd_beta,
                        "question_suffix": args.question_suffix,
                        "seed": args.seed,
                        "image_output_dir": args.image_output_dir,
                    },
                    "summary": {
                        "baseline_accuracy": baseline_acc,
                        "contrastive_accuracy": contrastive_acc,
                        "vcd_accuracy": vcd_acc,
                    },
                    "samples": [
                        {
                            "idx": entry["idx"],
                            "question_id": entry["question_id"],
                            "question": entry["question"],
                            "answer": entry["answer"],
                            "noise_mode": entry["noise_mode"],
                            "vcd_noise_step": entry["vcd_noise_step"],
                            "images": entry["image_paths"],
                            "greedy": {
                                "text": entry["baseline_text"],
                                "pred": entry["baseline_pred"],
                                "correct": entry["baseline_correct"],
                            },
                            "contrastive": {
                                "text": entry["contrastive_text"],
                                "pred": entry["contrastive_pred"],
                                "correct": entry["contrastive_correct"],
                            },
                            "vcd": {
                                "text": entry["vcd_text"],
                                "pred": entry["vcd_pred"],
                                "correct": entry["vcd_correct"],
                            },
                        }
                        for entry in results
                    ],
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"Saved detailed results to {args.output_json}")


if __name__ == "__main__":
    main()
