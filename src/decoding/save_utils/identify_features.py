import json
from pathlib import Path

import torch
from safetensors.torch import load_file


def generate_best_separation_features(
    save_dir: str,
    layer: int,
    topk: int = 10,
) -> str:
    module_path = f"model.layers.{layer}"
    correct_file = Path(save_dir) / "correct" / module_path / "Rank0_final.safetensors"
    incorrect_file = Path(save_dir) / "incorrect" / module_path / "Rank0_final.safetensors"

    if not correct_file.exists() or not incorrect_file.exists():
        raise FileNotFoundError(
            "Missing safetensors for feature identification. "
            "Run SAVE collect_sae_activations.py first."
        )

    correct_data = load_file(str(correct_file))
    incorrect_data = load_file(str(incorrect_file))

    correct_acts = correct_data["activations"]
    incorrect_acts = incorrect_data["activations"]
    correct_acts = correct_acts[:incorrect_acts.shape[0]]

    if correct_acts.dim() == 3:
        num_features = correct_acts.shape[2]
        correct_feats = (correct_acts > 0).nonzero(as_tuple=False)
        incorrect_feats = (incorrect_acts > 0).nonzero(as_tuple=False)
        correct_feat_indices = correct_feats[:, 2]
        incorrect_feat_indices = incorrect_feats[:, 2]
    elif correct_acts.dim() == 2:
        num_features = correct_acts.shape[1]
        correct_feats = (correct_acts > 0).nonzero(as_tuple=False)
        incorrect_feats = (incorrect_acts > 0).nonzero(as_tuple=False)
        correct_feat_indices = correct_feats[:, 1]
        incorrect_feat_indices = incorrect_feats[:, 1]
    else:
        raise ValueError(f"Unexpected activations shape: {correct_acts.shape}")

    correct_counts = torch.zeros(num_features, dtype=torch.float32)
    incorrect_counts = torch.zeros(num_features, dtype=torch.float32)

    for feat in correct_feat_indices:
        correct_counts[feat] += 1
    for feat in incorrect_feat_indices:
        incorrect_counts[feat] += 1

    correct_total = correct_acts.shape[0]
    incorrect_total = incorrect_acts.shape[0]

    p_correct = correct_counts / (correct_total + 1e-8)
    p_incorrect = incorrect_counts / (incorrect_total + 1e-8)

    separation_pos = p_correct - p_incorrect
    separation_neg = p_incorrect - p_correct

    _, topk_pos_indices = torch.topk(separation_pos, topk)
    _, topk_neg_indices = torch.topk(separation_neg, topk)

    result = {
        "correct_indices": topk_pos_indices.tolist(),
        "hallucinated_indices": topk_neg_indices.tolist(),
    }

    output_path = Path(save_dir) / "best_separation_feature.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(output_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--save-dir", type=str, required=True)
    parser.add_argument("--layer", type=int, default=24)
    parser.add_argument("--topk", type=int, default=10)
    args = parser.parse_args()

    path = generate_best_separation_features(args.save_dir, args.layer, args.topk)
    print(f"Saved: {path}")
