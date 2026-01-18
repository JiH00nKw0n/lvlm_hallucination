"""
Evaluate decoding strategies on MME and run PCA diagnostics on Pico-Banana.

Strategies live in src/decoding/* and are toggled by CLI flags.
"""

import argparse
import json
import os
import random
import math
from typing import Dict, List, Tuple, Optional
import requests
import asyncio
import aiohttp
from tqdm.asyncio import tqdm_asyncio

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, LlavaConfig, LlavaForConditionalGeneration

from src.datasets.mme import MMEDatasetBuilder
from src.decoding.greedy import GreedyDecoder
from src.decoding.noise_contrastive import NoiseContrastiveDecoder
from src.decoding.rotation import InstructionRotationDecoder
from src.decoding.pca_steering import PcaSteeringDecoder


def load_model_and_processor(model_name: str):
    processor = AutoProcessor.from_pretrained(model_name, use_fast=True)

    config = LlavaConfig.from_pretrained(model_name)
    config.text_config.auto_map = {
        "AutoModel": "src.models.eval_llama.modeling_eval_llama.EvalLlamaModel",
        "AutoModelForCausalLM": "src.models.eval_llama.modeling_eval_llama.EvalLlamaForCausalLM",
    }

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    device_map = "auto" if torch.cuda.is_available() else None

    model = LlavaForConditionalGeneration.from_pretrained(
        model_name,
        config=config,
        dtype=dtype,
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
    use_cache: bool = False,
    category_fraction: float = 1.0,
) -> None:
    torch.set_grad_enabled(False)
    tokenizer = processor.tokenizer
    device = next(model.parameters()).device
    builder = MMEDatasetBuilder(split=split)
    dataset = builder.build_dataset()

    total_samples = len(dataset)

    # build category-wise indices (preserve order)
    cat_to_indices: Dict[str, List[int]] = {}
    for idx in range(total_samples):
        cat = dataset[idx]["category"]
        cat_to_indices.setdefault(cat, []).append(idx)

    selected_indices: List[int] = []
    frac = max(0.0, min(1.0, category_fraction))
    for cat, idxs in cat_to_indices.items():
        if frac >= 1.0:
            take = len(idxs)
        else:
            take = max(1, math.ceil(len(idxs) * frac)) if len(idxs) > 0 else 0
        selected_indices.extend(idxs[:take])

    if max_samples != -1:
        selected_indices = selected_indices[:max_samples]
    samples = [dataset[i] for i in selected_indices]
    totals_per_cat: Dict[str, int] = {}
    for sample in samples:
        cat = sample["category"]
        totals_per_cat[cat] = totals_per_cat.get(cat, 0) + 1
    total_seen = len(samples)

    noise_generator = VibrantHueNoise()
    correct_counts: Dict[str, Dict[str, int]] = {}
    seen_counts: Dict[str, Dict[str, int]] = {}

    strategy_names = [s.name for s in strategies]
    print(
        f"MME evaluation | strategies: {strategy_names} | max_samples: {len(samples)} | "
        f"category_fraction={category_fraction}"
    )

    summary_config = {
        "split": split,
        "max_samples": max_samples,
        "max_new_tokens": max_new_tokens,
        "noise_scale": noise_scale,
        "question_suffix": question_suffix,
    }

    def build_summary() -> Dict:
        summary = {
            "config": summary_config,
            "totals_per_category": totals_per_cat,
            "strategies": {},
            "total_seen": total_seen,
        }
        for strat in strategies:
            cat_counts = correct_counts.get(strat.name, {})
            cat_seen = seen_counts.get(strat.name, {})
            per_cat = {}
            total_correct = 0
            total_seen_so_far = 0
            for cat, total_cat in totals_per_cat.items():
                seen = cat_seen.get(cat, 0)
                correct = cat_counts.get(cat, 0)
                acc = correct / seen if seen else 0.0
                per_cat[cat] = {
                    "correct": correct,
                    "seen": seen,
                    "total": total_cat,
                    "accuracy": acc,
                }
                total_correct += correct
                total_seen_so_far += seen
            overall_acc = total_correct / total_seen_so_far if total_seen_so_far else 0.0
            summary["strategies"][strat.name] = {
                "overall": {
                    "correct": total_correct,
                    "seen": total_seen_so_far,
                    "total": total_seen,
                    "accuracy": overall_acc,
                },
                "per_category": per_cat,
            }
        return summary

    def write_summary() -> None:
        if not output_json:
            return
        out_dir = os.path.dirname(output_json)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        summary = build_summary()
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    for strat in strategies:
        if isinstance(strat, PcaSteeringDecoder) and not os.path.exists(strat.pca_path):
            print(f"[MME] Skipping {strat.name}: PCA file not found -> {strat.pca_path}")
            continue
        strat_counts: Dict[str, int] = {}
        strat_seen: Dict[str, int] = {}
        pbar = tqdm(range(total_seen), desc=f"MME [{strat.name}]")
        for idx in pbar:
            sample = samples[idx]
            image: Image.Image = sample["image"].convert("RGB")
            answer: str = sample["answer"].strip().lower()
            category: str = sample["category"]

            prompt = build_prompt(processor, sample["question"], question_suffix)
            clean_inputs = make_inputs(processor, prompt, image, device)

            if isinstance(strat, NoiseContrastiveDecoder):
                noisy_image = noise_generator(image)
                noisy_inputs = make_inputs(processor, prompt, noisy_image, device)
                result = strat.decode(
                    model,
                    tokenizer,
                    clean_inputs=clean_inputs,
                    noisy_inputs=noisy_inputs,
                    max_new_tokens=max_new_tokens,
                    noise_scale=noise_scale,
                    use_cache=use_cache,
                )
            else:
                result = strat.decode(
                    model,
                    tokenizer,
                    clean_inputs=clean_inputs,
                    max_new_tokens=max_new_tokens,
                    use_cache=use_cache,
                )

            pred = normalize_yes_no(result.text)
            correct = int(pred == answer)
            strat_counts[category] = strat_counts.get(category, 0) + correct
            strat_seen[category] = strat_seen.get(category, 0) + 1

            # Persist partial progress so long runs keep intermediate results.
            correct_counts[strat.name] = strat_counts
            seen_counts[strat.name] = strat_seen
            write_summary()

        correct_counts[strat.name] = strat_counts
        seen_counts[strat.name] = strat_seen

    summary = build_summary()

    print("\n=== MME results ===")
    for strat in strategies:
        print(f"\nStrategy: {strat.name}")
        strat_summary = summary["strategies"].get(strat.name, {})
        per_cat = strat_summary.get("per_category", {})
        overall_stats = strat_summary.get("overall", {})
        for cat, total_cat in totals_per_cat.items():
            cat_stats = per_cat.get(cat, {})
            c = cat_stats.get("correct", 0)
            seen = cat_stats.get("seen", 0)
            acc = cat_stats.get("accuracy", 0.0)
            suffix = f" of {total_cat} total" if seen != total_cat else ""
            print(f"  {cat:20s} acc={acc:.3f} ({c}/{seen}{suffix})")
        overall_acc = overall_stats.get("accuracy", 0.0)
        overall_seen = overall_stats.get("seen", 0)
        overall_correct = overall_stats.get("correct", 0)
        suffix = f" of {total_seen} planned" if overall_seen != total_seen else ""
        print(f"  Overall acc={overall_acc:.3f} ({overall_correct}/{overall_seen}{suffix})")

    if output_json:
        write_summary()
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
    batch_size: int = 1,
) -> Tuple[List[List[torch.Tensor]], List[str], int]:
    device = next(model.parameters()).device
    all_means: List[List[torch.Tensor]] = []
    valid_texts: List[str] = []
    skipped_nonfinite = 0
    for start in tqdm(range(0, len(texts), batch_size), desc="Encoding text", leave=False):
        batch = texts[start : start + batch_size]
        tokenized = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
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
            bad_sample = False
            for layer_hidden in hidden_states:
                masked = layer_hidden[b_idx].float() * attn_mask
                mean_vec = masked.sum(dim=0) / denom.squeeze()
                mean_vec = mean_vec.squeeze()
                if not torch.isfinite(mean_vec).all():
                    bad_sample = True
                    break
                layer_means.append(mean_vec.detach().cpu())
            if bad_sample:
                skipped_nonfinite += 1
                continue
            all_means.append(layer_means)
            valid_texts.append(batch[b_idx])
    return all_means, valid_texts, skipped_nonfinite


def run_pca(matrix: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    matrix = matrix.float()
    # Drop non-finite rows defensively
    mask = torch.isfinite(matrix).all(dim=1)
    if mask.sum() == 0:
        return torch.empty(0, matrix.shape[1]), torch.zeros(matrix.shape[1], dtype=matrix.dtype)
    matrix = matrix[mask]
    mean = matrix.mean(dim=0, keepdim=True)
    centered = matrix - mean
    q = min(centered.shape[0], centered.shape[1])
    if q == 0:
        return torch.empty(0, matrix.shape[1]), mean.squeeze(0)
    _, _, v = torch.pca_lowrank(centered, q=q)
    components = v.T
    return components, mean.squeeze(0)


def pca_on_pico_text(
    model, tokenizer, jsonl_path: str, output_dir: str, batch_size: int = 2, sample_fraction: float = 1.0
) -> None:
    try:
        entries = load_pico_entries(jsonl_path)
    except FileNotFoundError as e:
        print(f"[PCA text] {e}")
        return
    print(f"[PCA text] Loaded {len(entries)} entries from {jsonl_path}")
    # subsample entries if requested (preserve order)
    if sample_fraction < 1.0:
        keep = max(1, int(len(entries) * sample_fraction))
        entries = entries[:keep]
        print(f"[PCA text] Subsampled to {len(entries)} entries (fraction={sample_fraction})")
    print("[PCA text] Encoding instructions...")

    instructions: List[str] = []
    for entry in entries:
        prompts = entry.get("metadata_edit_turn_prompts") or []
        instructions.extend(prompts)

    if not instructions:
        print("No instructions found for PCA text analysis.")
        return
    instructions = list(set(instructions))
    print(f"[PCA text] Total instructions collected: {len(instructions)}")
    means, valid_instructions, skipped_nf = compute_text_hidden_means(
        model, tokenizer, instructions, batch_size=batch_size
    )
    print(f"[PCA text] Encoded instructions (finite): {len(means)} | skipped non-finite: {skipped_nf}")
    if not means:
        print("[PCA text] No valid instructions to encode; aborting text PCA.")
        return
    num_layers = len(means[0])
    os.makedirs(output_dir, exist_ok=True)
    all_components = {}
    all_means = {}
    nearest: Dict[int, List[Dict]] = {}
    empty_layers: Dict[int, int] = {}
    for layer_idx in tqdm(range(num_layers), desc="PCA text layers", leave=False):
        layer_stack = torch.stack([m[layer_idx] for m in means], dim=0)
        # Drop rows with any non-finite values
        mask = torch.isfinite(layer_stack).all(dim=1)
        if mask.sum() != layer_stack.shape[0]:
            print(
                f"[PCA text] Layer {layer_idx}: removed {layer_stack.shape[0]-mask.sum().item()} non-finite samples "
                f"(kept {mask.sum().item()})"
            )
        layer_stack = layer_stack[mask]
        comps, mean_vec = run_pca(layer_stack)
        all_components[layer_idx] = comps
        all_means[layer_idx] = mean_vec
        # top-5 PCs, nearest 5 by absolute projection
        if comps.numel() == 0 or layer_stack.shape[0] == 0:
            empty_layers[layer_idx] = layer_stack.shape[0]
            continue
        centered = layer_stack - mean_vec
        coords = torch.matmul(centered, comps.T)
        top_pc = min(5, comps.shape[0])
        layer_near: List[Dict] = []
        for pc_idx in range(top_pc):
            scores = coords[:, pc_idx]
            n = scores.numel()
            if n <= 0:
                print(f"[PCA text] Layer {layer_idx} PC {pc_idx} has 0 samples; skipping.")
                continue
            topk = min(5, n)
            try:
                vals, idxs = torch.topk(scores.abs(), k=topk)
            except RuntimeError as e:
                print(f"[PCA text] topk failed at layer {layer_idx}, pc {pc_idx}, n={n}: {e}")
                continue
            for rank, (v, ix) in enumerate(zip(vals.tolist(), idxs.tolist())):
                layer_near.append(
                    {
                        "layer": layer_idx,
                        "pc_index": pc_idx,
                        "rank": rank,
                        "score": v,
                        "instruction_index": ix,
                        "instruction": valid_instructions[ix],
                    }
                )
        nearest[layer_idx] = layer_near
    torch.save({"components": all_components, "means": all_means}, os.path.join(output_dir, "text_pca.pt"))
    with open(os.path.join(output_dir, "text_pca_nearest.json"), "w", encoding="utf-8") as f:
        json.dump(nearest, f, ensure_ascii=False, indent=2)
    if empty_layers:
        print(f"[PCA text] Layers with 0 valid instructions: {sorted(empty_layers.keys())}")
    print(f"Saved text PCA components to {os.path.join(output_dir, 'text_pca.pt')}")


def project_single_layer(hs: torch.Tensor, projector, slice_idx: int, hidden_size: int) -> torch.Tensor:
    """
    Project a single vision layer hidden state through the multimodal projector slice.
    """
    w1 = projector.linear_1.weight[:, slice_idx * hidden_size : (slice_idx + 1) * hidden_size]
    b1 = projector.linear_1.bias
    x = torch.nn.functional.linear(hs, w1, b1)
    x = projector.act(x)
    x = projector.linear_2(x)
    return x


def download_if_needed(path: str, url_candidates: List[str]) -> Optional[str]:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    for url in url_candidates:
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                with open(path, "wb") as f:
                    f.write(resp.content)
                return path
        except Exception:
            continue
    return None


async def _download_file(session, sem, path: str, urls: List[str]) -> bool:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    async with sem:
        for url in urls:
            try:
                async with session.get(url, timeout=15) as resp:
                    if resp.status != 200:
                        continue
                    data = await resp.read()
                    with open(path, "wb") as f:
                        f.write(data)
                    return True
            except Exception:
                continue
    return False


async def download_many(missing: Dict[str, List[str]], max_concurrency: int = 64) -> List[str]:
    if not missing:
        return []
    sem = asyncio.Semaphore(max_concurrency)
    downloaded = []
    async with aiohttp.ClientSession() as session:
        tasks = []
        keys = []
        for path, urls in missing.items():
            if os.path.exists(path):
                continue
            tasks.append(_download_file(session, sem, path, urls))
            keys.append(path)
        results = await tqdm_asyncio.gather(*tasks, desc="Downloading missing images")
        for path, ok in zip(keys, results):
            if ok:
                downloaded.append(path)
    return downloaded


def pca_on_pico_images(
    model,
    processor,
    jsonl_path: str,
    output_dir: str,
    save_images: bool = False,
    sample_fraction: float = 1.0,
    image_base_url: str = "",
) -> None:
    try:
        entries = load_pico_entries(jsonl_path)
    except FileNotFoundError as e:
        print(f"[PCA image] {e}")
        return
    print(f"[PCA image] Loaded {len(entries)} entries from {jsonl_path}")
    if sample_fraction < 1.0:
        keep = max(1, int(len(entries) * sample_fraction))
        entries = entries[:keep]
        print(f"[PCA image] Subsampled to {len(entries)} entries (fraction={sample_fraction})")
    print("[PCA image] Gathering image feature differences...")
    device = next(model.parameters()).device
    vision_cfg = model.config.vision_config
    projector = model.model.multi_modal_projector
    vision_layers_cfg = model.config.vision_feature_layer
    if isinstance(vision_layers_cfg, int):
        layers_to_use = [vision_layers_cfg]
    else:
        layers_to_use = list(vision_layers_cfg)

    diffs_per_layer: Dict[int, List[torch.Tensor]] = {layer: [] for layer in layers_to_use}
    metas_per_layer: Dict[int, List[Dict]] = {layer: [] for layer in layers_to_use}
    saved_dir = os.path.join(output_dir, "nearest_images")
    if save_images:
        os.makedirs(saved_dir, exist_ok=True)

    # prepare download list
    missing: Dict[str, List[str]] = {}

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
        # prepare download candidates if missing
        if not os.path.exists(local_input):
            orig_url = None
            for f in files:
                if f.get("id") == "original_input_image":
                    orig_url = f.get("url")
                    break
            candidates = []
            if orig_url:
                candidates.append(orig_url)
            if image_base_url and local_input:
                candidates.append(image_base_url.rstrip("/") + "/" + local_input)
            if candidates:
                missing[local_input] = candidates
        if not os.path.exists(final_path):
            candidates = []
            if image_base_url and final_path.startswith("images/"):
                candidates.append(image_base_url.rstrip("/") + "/" + final_path)
            for f in files:
                if f.get("id") == "final_image":
                    url = f.get("url")
                    if url:
                        candidates.append(url)
            if candidates:
                missing[final_path] = candidates

    # download missing files concurrently
    downloaded = asyncio.run(download_many(missing))
    if downloaded:
        print(f"[PCA image] Downloaded {len(downloaded)} missing files.")

    # restart loop now that downloads attempted
    for entry in tqdm(entries, desc="Pico image PCA (with downloads)"):
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
            orig_inputs = {k: v.to(device) for k, v in orig_inputs.items()}
            final_inputs = {k: v.to(device) for k, v in final_inputs.items()}
            orig_vis = model.model.vision_tower(orig_inputs["pixel_values"], output_hidden_states=True)
            final_vis = model.model.vision_tower(final_inputs["pixel_values"], output_hidden_states=True)

        hidden_size = vision_cfg.hidden_size
        for slice_idx, layer_idx in enumerate(layers_to_use):
            try:
                orig_hs = orig_vis.hidden_states[layer_idx]
                final_hs = final_vis.hidden_states[layer_idx]
            except IndexError:
                continue

            # Drop CLS if default strategy (matches get_image_features behavior)
            orig_tokens = orig_hs[:, 1:] if model.config.vision_feature_select_strategy == "default" else orig_hs
            final_tokens = final_hs[:, 1:] if model.config.vision_feature_select_strategy == "default" else final_hs

            orig_proj = project_single_layer(orig_tokens, projector, slice_idx, hidden_size)
            final_proj = project_single_layer(final_tokens, projector, slice_idx, hidden_size)

            orig_vec = orig_proj.mean(dim=1).cpu()
            final_vec = final_proj.mean(dim=1).cpu()
            diff_vec = (final_vec - orig_vec).squeeze(0)
            diffs_per_layer[layer_idx].append(diff_vec)
            meta_entry = {
                "local_input_image": local_input,
                "final_image": final_path,
                "prompts": entry.get("metadata_edit_turn_prompts") or [],
            }
            if save_images:
                base = f"layer{layer_idx}_idx{len(metas_per_layer[layer_idx])}"
                local_out = os.path.join(saved_dir, f"{base}_orig.png")
                final_out = os.path.join(saved_dir, f"{base}_final.png")
                try:
                    orig_img.save(local_out)
                    final_img.save(final_out)
                    meta_entry["saved_local_image"] = local_out
                    meta_entry["saved_final_image"] = final_out
                except Exception:
                    pass
            metas_per_layer[layer_idx].append(meta_entry)

    any_diff = any(len(v) > 0 for v in diffs_per_layer.values())
    if not any_diff:
        print("No image pairs with local files found; skipping image PCA.")
        return

    os.makedirs(output_dir, exist_ok=True)
    comps_dict: Dict[int, torch.Tensor] = {}
    means_dict: Dict[int, torch.Tensor] = {}
    empty_layers: Dict[int, int] = {}
    nearest: Dict[int, List[Dict]] = {}
    for layer_idx, vecs in diffs_per_layer.items():
        if not vecs:
            print(f"[PCA image] Layer {layer_idx} has 0 valid pairs; skipping.")
            empty_layers[layer_idx] = 0
            continue
        diff_tensor = torch.stack(vecs, dim=0)
        components, mean_vec = run_pca(diff_tensor)
        comps_dict[layer_idx] = components
        means_dict[layer_idx] = mean_vec
        if components.numel() == 0:
            continue
        centered = diff_tensor - mean_vec
        coords = torch.matmul(centered, components.T)
        top_pc = min(5, components.shape[0])
        layer_near: List[Dict] = []
        for pc_idx in range(top_pc):
            scores = coords[:, pc_idx]
            topk = min(5, scores.shape[0])
            if topk == 0:
                continue
            vals, idxs = torch.topk(scores.abs(), k=topk)
            for rank, (v, ix) in enumerate(zip(vals.tolist(), idxs.tolist())):
                meta = metas_per_layer[layer_idx][ix]
                entry = {
                    "layer": layer_idx,
                    "pc_index": pc_idx,
                    "rank": rank,
                    "score": v,
                    "local_input_image": meta.get("local_input_image"),
                    "final_image": meta.get("final_image"),
                    "prompts": meta.get("prompts", []),
                }
                if save_images:
                    entry["saved_local_image"] = meta.get("saved_local_image")
                    entry["saved_final_image"] = meta.get("saved_final_image")
                layer_near.append(entry)
        nearest[layer_idx] = layer_near

    torch.save(
        {"components": comps_dict, "means": means_dict},
        os.path.join(output_dir, "image_pca.pt"),
    )
    with open(os.path.join(output_dir, "image_pca_nearest.json"), "w", encoding="utf-8") as f:
        json.dump(nearest, f, ensure_ascii=False, indent=2)
    if empty_layers:
        print(f"[PCA image] Layers with 0 valid pairs skipped: {sorted(empty_layers.keys())}")
    used_counts = {k: len(v) for k, v in diffs_per_layer.items() if v}
    print(f"[PCA image] Layer-wise pairs used: {used_counts} | saved to {os.path.join(output_dir, 'image_pca.pt')}")


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
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Enable generation cache during decoding (faster, higher memory).",
    )
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
    parser.add_argument("--pca-components-path", type=str, default="results/pico_pca/text_pca.pt")
    parser.add_argument(
        "--pca-layer",
        type=int,
        default=None,
        help="Layer index for PCA steering (None → last available in checkpoint).",
    )
    parser.add_argument("--pca-top-k", type=int, default=4, help="Top-k PCA components to remove when steering.")
    parser.add_argument(
        "--pca-contrast-scale",
        type=float,
        default=0.5,
        help="Contrastive scale when comparing clean vs. steered logits.",
    )
    parser.add_argument("--run-pca-steering", action="store_true", help="Enable PCA steering with components path.")
    parser.add_argument(
        "--pca-steering-mode",
        type=str,
        default="text",
        help="Which PCA file to use for steering: text, image, or both (comma-separated or 'both').",
    )
    parser.add_argument(
        "--category-fraction",
        type=float,
        default=1.0,
        help="Fraction of each category to evaluate (0-1].",
    )
    parser.add_argument(
        "--pca-save-images",
        action="store_true",
        help="When running image PCA, save nearest-pair images into pca-output-dir/nearest_images.",
    )
    parser.add_argument(
        "--pico-image-base-url",
        type=str,
        default="",
        help="Base URL prefix to download images if local files are missing (e.g., https://ml-site.cdn-apple.com/datasets/pico-banana-300k/nb).",
    )
    parser.add_argument(
        "--pca-sample-fraction",
        type=float,
        default=1.0,
        help="Fraction of entries to use when computing PCA (0-1].",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    model, processor = load_model_and_processor(args.model_name)
    tokenizer = processor.tokenizer
    device = next(model.parameters()).device
    print(f"Loaded model on {device}")

    # Determine steering modes list up front
    steering_modes: List[str] = []
    modes_raw = [m.strip() for m in args.pca_steering_mode.split(",") if m.strip()]
    for m in modes_raw:
        if m.lower() == "both":
            steering_modes.extend(["text", "image"])
        else:
            steering_modes.append(m.lower())
    if not steering_modes:
        steering_modes = ["text"]

    # Auto-run PCA if missing and steering requested
    if args.run_pca_steering:
        if "text" in steering_modes and not os.path.exists(args.pca_components_path):
            print(f"[PCA text] Not found at {args.pca_components_path}, computing...")
            pca_on_pico_text(
                model=model,
                tokenizer=tokenizer,
                jsonl_path=args.pico_jsonl,
                output_dir=args.pca_output_dir,
                batch_size=args.text_batch_size,
                sample_fraction=args.pca_sample_fraction,
            )
        if "image" in steering_modes:
            image_pca_path = os.path.join(args.pca_output_dir, "image_pca.pt")
            if not os.path.exists(image_pca_path):
                print(f"[PCA image] Not found at {image_pca_path}, computing...")
                pca_on_pico_images(
                    model=model,
                    processor=processor,
                    jsonl_path=args.pico_jsonl,
                    output_dir=args.pca_output_dir,
                    save_images=args.pca_save_images,
                    sample_fraction=args.pca_sample_fraction,
                    image_base_url=args.pico_image_base_url,
                )

    strategies = []
    if args.run_greedy:
        strategies.append(GreedyDecoder())
    if args.run_noise_contrastive:
        strategies.append(NoiseContrastiveDecoder(noise_scale=args.noise_scale))
    if args.run_simple_rotation:
        degrees = [float(x) for x in args.rotation_degrees.split(",") if x.strip()]
        for deg in degrees:
            strategies.append(InstructionRotationDecoder(degrees=deg, contrast_scale=1.0))
    if args.run_pca_steering:
        for mode in steering_modes:
            if mode == "text":
                pca_path = args.pca_components_path
            elif mode == "image":
                pca_path = os.path.join(args.pca_output_dir, "image_pca.pt")
            else:
                continue
            strategies.append(
                PcaSteeringDecoder(
                    pca_path=pca_path,
                    pca_layer=args.pca_layer,
                    top_k=args.pca_top_k,
                    contrast_scale=args.pca_contrast_scale,
                    use_cache=args.use_cache,
                    source=mode,
                )
            )

    if strategies:
        print(f"Enabled strategies: {[s.name for s in strategies]}")
    else:
        print("No decoding strategies enabled; skipping MME evaluation.")

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
            use_cache=args.use_cache,
            category_fraction=args.category_fraction,
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
            save_images=args.pca_save_images,
            sample_fraction=args.pca_sample_fraction,
            image_base_url=args.pico_image_base_url,
        )


if __name__ == "__main__":
    main()
