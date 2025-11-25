"""
Compute per-layer hidden-state means for originals and generated captions,
run per-layer PCA on difference vectors, visualize 2D projections, and pick
nearest samples to top principal components.
"""

import argparse
import json
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from src.models.llava.modeling_llava import CustomLlavaForConditionalGeneration


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PCA analysis on LLaVA hidden states.")
    parser.add_argument(
        "--json-path",
        type=str,
        default="llada_mask_generations.json",
        help="Path to JSON produced by test_llada.py.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="llava-hf/llava-1.5-7b-hf",
        help="LLaVA checkpoint to load.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="llava_pca_outputs",
        help="Directory to store PCA plots and matrices.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size for encoding text.",
    )
    parser.add_argument(
        "--top-k-pc",
        type=int,
        default=5,
        help="Number of leading principal components to search nearest samples for.",
    )
    parser.add_argument(
        "--nearest-per-pc",
        type=int,
        default=5,
        help="Number of nearest samples to keep per principal component.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="Model dtype.",
    )
    return parser.parse_args()


def load_mask_json(path: str) -> Tuple[List[str], List[List[str]]]:
    with open(path, "r", encoding="utf-8") as f:
        data: Dict[str, Dict[str, List[str]]] = json.load(f)
    originals: List[str] = []
    generated: List[List[str]] = []
    for idx in sorted(data.keys(), key=lambda x: int(x)):
        entry = data[idx]
        originals.append(entry["original"])
        generated.append(entry["generated"])
    return originals, generated


def prepare_model(model_name: str, dtype: torch.dtype, device: torch.device):
    model = CustomLlavaForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=None,
    ).to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer


def compute_mean_hidden_states(
    texts: List[str],
    tokenizer,
    language_model,
    device: torch.device,
    batch_size: int,
) -> List[List[torch.Tensor]]:
    all_means: List[List[torch.Tensor]] = []
    for start in tqdm(range(0, len(texts), batch_size), desc="Encoding", leave=False):
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
            outputs = language_model(
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


def run_pca(data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Run PCA via torch.pca_lowrank on centered data.
    Returns (components, mean), where components are (num_components, hidden_dim).
    """
    data = data.float()
    data_mean = data.mean(dim=0, keepdim=True)
    centered = data - data_mean
    q = min(centered.shape[0], centered.shape[1])
    _, _, v = torch.pca_lowrank(centered, q=q)
    components = v.T  # (num_components, hidden_dim)
    return components, data_mean.squeeze(0)


def save_pca_plot(coords: torch.Tensor, layer_idx: int, output_dir: str) -> None:
    plt.figure(figsize=(6, 6))
    plt.scatter(coords[:, 0].cpu().numpy(), coords[:, 1].cpu().numpy(), s=4, alpha=0.6)
    plt.title(f"Layer {layer_idx} PCA (PC1 vs PC2)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"pca_layer_{layer_idx}.png"), dpi=200)
    plt.close()


def find_nearest_pairs(
    diffs: torch.Tensor,
    components: torch.Tensor,
    num_pc: int,
    top_k: int,
    num_generated: int,
    originals: List[str],
    generated: List[List[str]],
    layer_idx: int,
) -> List[Dict]:
    diffs = diffs.float()
    diffs_norm = torch.nn.functional.normalize(diffs, dim=1)
    selected_pairs: List[Dict] = []
    for pc_idx in range(min(num_pc, components.shape[0])):
        pc_vec = torch.nn.functional.normalize(components[pc_idx], dim=0)
        scores = torch.matmul(diffs_norm, pc_vec)
        top_scores, top_indices = torch.topk(scores, k=min(top_k, scores.shape[0]))
        for rank, (score, flat_idx) in enumerate(zip(top_scores.tolist(), top_indices.tolist())):
            orig_idx = flat_idx // num_generated
            gen_idx = flat_idx % num_generated
            selected_pairs.append(
                {
                    "layer": layer_idx,
                    "pc_index": pc_idx,
                    "rank": rank,
                    "score": score,
                    "original_index": orig_idx,
                    "generated_index": gen_idx,
                    "original": originals[orig_idx],
                    "generated": generated[orig_idx][gen_idx],
                }
            )
    return selected_pairs


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    dtype = dtype_map[args.dtype]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    originals, generated = load_mask_json(args.json_path)
    num_originals = len(originals)
    num_generated = len(generated[0]) if generated else 0
    print(f"Loaded {num_originals} originals with {num_generated} generations each.")

    model, tokenizer = prepare_model(args.model_name, dtype, device)
    language_model = model.language_model

    print("Encoding originals...")
    original_means = compute_mean_hidden_states(
        originals, tokenizer, language_model, device, args.batch_size
    )
    print("Encoding generated captions...")
    flat_generated = [g for group in generated for g in group]
    generated_means = compute_mean_hidden_states(
        flat_generated, tokenizer, language_model, device, args.batch_size
    )

    num_layers = len(original_means[0])
    print(f"Detected {num_layers} layers (excluding embeddings).")

    # Reshape generated means back into [num_originals, num_generated_per, num_layers]
    gen_means_structured: List[List[List[torch.Tensor]]] = []
    for i in tqdm(range(num_originals), desc="Reshaping generations", leave=False):
        start = i * num_generated
        end = start + num_generated
        gen_means_structured.append(generated_means[start:end])

    components_per_layer: Dict[int, torch.Tensor] = {}
    pca_means_per_layer: Dict[int, torch.Tensor] = {}
    all_selected_pairs: List[Dict] = []

    for layer_idx in tqdm(range(num_layers), desc="Layers"):
        print(f"Processing layer {layer_idx}...")
        orig_layer = torch.stack([o[layer_idx] for o in original_means], dim=0)
        gen_layer = torch.stack(
            [
                torch.stack([g[layer_idx] for g in gen_group], dim=0)
                for gen_group in gen_means_structured
            ],
            dim=0,
        )  # (num_originals, num_generated, hidden_dim)

        # Broadcast subtraction to build difference vectors
        diffs = (gen_layer - orig_layer.unsqueeze(1)).reshape(-1, orig_layer.shape[-1])

        components, mean_vec = run_pca(diffs)
        components_per_layer[layer_idx] = components
        pca_means_per_layer[layer_idx] = mean_vec

        diffs_for_proj = diffs.float() - mean_vec
        coords_2d = torch.matmul(diffs_for_proj, components[:2].T)
        save_pca_plot(coords_2d, layer_idx, args.output_dir)

        selected_pairs = find_nearest_pairs(
            diffs,
            components,
            args.top_k_pc,
            args.nearest_per_pc,
            num_generated,
            originals,
            generated,
            layer_idx,
        )
        all_selected_pairs.extend(selected_pairs)

    torch.save(
        {
            "components": components_per_layer,
            "means": pca_means_per_layer,
        },
        os.path.join(args.output_dir, "pca_matrices.pt"),
    )
    with open(os.path.join(args.output_dir, "top_pairs.json"), "w", encoding="utf-8") as f:
        json.dump(all_selected_pairs, f, ensure_ascii=False, indent=2)

    print(f"Saved PCA matrices and pairs to {args.output_dir}")


if __name__ == "__main__":
    main()
