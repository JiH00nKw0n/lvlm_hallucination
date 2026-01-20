import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from huggingface_hub import hf_hub_download
from .identify_features import generate_best_separation_features


def remove_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    cleaned: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            cleaned[key[7:]] = value
        else:
            cleaned[key] = value
    return cleaned


def load_feature_indices(path: str) -> Tuple[List[int], List[int]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data.get("correct_indices", []), data.get("hallucinated_indices", [])


def resolve_or_generate_feature_path(
    feature_path: Optional[str],
    save_dir: str,
    layer: int,
) -> str:
    base_dir = Path(save_dir)
    auto_path = base_dir / "best_separation_feature.json"

    if feature_path:
        path = Path(feature_path)
        if path.is_dir():
            candidate = path / "best_separation_feature.json"
            if candidate.exists():
                return str(candidate)
        if path.exists():
            return str(path)

    if auto_path.exists():
        return str(auto_path)

    return generate_best_separation_features(save_dir, layer)


def resolve_sae_checkpoint(
    repo_id: str,
    filename: str,
    local_dir: Optional[str] = None,
) -> str:
    return hf_hub_download(repo_id=repo_id, filename=filename, local_dir=local_dir)
