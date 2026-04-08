"""
Extract CLIP embeddings and GT class feature directions from Tiny ImageNet.

Two GT methods:
  1. Linear Probing  — logistic regression weights as class directions
  2. Mean Embedding  — (class_mean - global_mean), normalised

Outputs are cached to disk; re-running with the same --output-dir skips
any stage whose files already exist.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# Helpers                                                            #
# ------------------------------------------------------------------ #

def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_arg)


def _normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norms, eps, None)


# ------------------------------------------------------------------ #
# Class-name resolution                                              #
# ------------------------------------------------------------------ #

def _resolve_class_names(ds) -> list[str]:
    """Return human-readable class names for the 200 Tiny ImageNet classes."""
    label_feature = ds.features["label"]
    raw_names: list[str] = label_feature.names  # WordNet IDs or names

    # If names look like WordNet IDs (e.g. n02124075), resolve via NLTK
    if raw_names and re.match(r"^n\d{8}$", raw_names[0]):
        try:
            from nltk.corpus import wordnet as wn
            resolved: list[str] = []
            for wnid in raw_names:
                offset = int(wnid[1:])
                synset = wn.synset_from_pos_and_offset("n", offset)
                lemmas = synset.lemmas()  # type: ignore[union-attr]
                resolved.append(lemmas[0].name().replace("_", " "))
            return resolved
        except Exception:
            logger.warning("NLTK WordNet resolution failed; using raw IDs")
            return raw_names

    return raw_names


# ------------------------------------------------------------------ #
# Template loading                                                   #
# ------------------------------------------------------------------ #

def _load_templates(template_file: str) -> list[str]:
    """Parse template.txt and return list of format-string templates."""
    text = Path(template_file).read_text(encoding="utf-8")
    templates: list[str] = []
    for line in text.splitlines():
        line = line.strip().strip(",")
        # Match quoted strings like 'a photo of a {}.'
        m = re.match(r"""^['\"](.+?)['\"]\.?$""", line)
        if m:
            templates.append(m.group(1))
    if not templates:
        raise ValueError(f"No templates parsed from {template_file}")
    logger.info("Loaded %d templates from %s", len(templates), template_file)
    return templates


def _make_paired_text(
    class_name: str,
    templates: list[str],
) -> str:
    """Create a text prompt for one sample: random template + random label part."""
    parts = [p.strip() for p in class_name.split(",")]
    chosen_part = random.choice(parts)
    template = random.choice(templates)
    return template.format(chosen_part)


# ------------------------------------------------------------------ #
# CLIP embedding extraction                                          #
# ------------------------------------------------------------------ #

class _ImageDataset(Dataset):
    """Wraps a HuggingFace dataset split for batched CLIP image extraction."""

    def __init__(self, hf_ds, processor):
        self.hf_ds = hf_ds
        self.processor = processor

    def __len__(self) -> int:
        return len(self.hf_ds)

    def __getitem__(self, idx: int):
        item = self.hf_ds[idx]
        image = item["image"].convert("RGB")
        label = item["label"]
        pixel_values = self.processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)
        return pixel_values, label


def _extract_image_embeddings(
    hf_ds,
    clip_model,
    processor,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract CLIP image embeddings. Returns (embeds, labels)."""
    dataset = _ImageDataset(hf_ds, processor)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=device.type == "cuda",
    )

    all_embeds: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    for pixel_values, labels in tqdm(loader, desc="Image embeddings"):
        pixel_values = pixel_values.to(device)
        with torch.no_grad():
            embeds = clip_model.get_image_features(pixel_values=pixel_values)
        all_embeds.append(embeds.cpu())
        all_labels.append(labels)

    return torch.cat(all_embeds, dim=0), torch.cat(all_labels, dim=0)


def _extract_text_embeddings(
    hf_ds,
    clip_model,
    processor,
    class_names: list[str],
    templates: list[str],
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract CLIP text embeddings with random template pairing.

    Returns (embeds, labels) — one text embedding per image sample.
    """
    # Pre-generate all texts
    texts: list[str] = [
        _make_paired_text(class_names[item["label"]], templates)
        for item in hf_ds
    ]
    labels = torch.tensor([item["label"] for item in hf_ds], dtype=torch.long)

    all_embeds: list[torch.Tensor] = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Text embeddings"):
        batch_texts = texts[i : i + batch_size]
        inputs = processor(text=batch_texts, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            embeds = clip_model.get_text_features(**inputs)
        all_embeds.append(embeds.cpu())

    return torch.cat(all_embeds, dim=0), labels


# ------------------------------------------------------------------ #
# GT feature direction extraction                                    #
# ------------------------------------------------------------------ #

def _extract_gt_linear_probe(
    train_embeds: np.ndarray,
    train_labels: np.ndarray,
    val_embeds: np.ndarray,
    val_labels: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    """Logistic regression → normalised weight vectors as class directions.

    Returns (directions (n_classes, d), probe_loss, probe_accuracy).
    """
    clf = LogisticRegression(
        max_iter=1000,
        C=1.0,
        multi_class="multinomial",
        solver="lbfgs",
    )
    clf.fit(train_embeds, train_labels)
    probe_accuracy = float(clf.score(val_embeds, val_labels))
    probe_loss = float(log_loss(val_labels, clf.predict_proba(val_embeds)))
    directions = _normalize_rows(clf.coef_.astype(np.float64))
    return directions, probe_loss, probe_accuracy


def _extract_gt_mean_embedding(
    embeds: np.ndarray,
    labels: np.ndarray,
    n_classes: int,
) -> np.ndarray:
    """Mean-embedding directions: (class_mean - global_mean), normalised.

    Returns directions (n_classes, d).
    """
    mu_all = embeds.mean(axis=0)
    mu_c = np.array([embeds[labels == c].mean(axis=0) for c in range(n_classes)])
    w = mu_c - mu_all
    return _normalize_rows(w.astype(np.float64))


# ------------------------------------------------------------------ #
# Cross-modal GT similarity                                         #
# ------------------------------------------------------------------ #

def _compute_cross_modal_similarity(
    img_directions: np.ndarray,
    txt_directions: np.ndarray,
) -> dict[str, float]:
    """Per-class cosine similarity between image and text GT directions."""
    cos_sims = (img_directions * txt_directions).sum(axis=1)
    return {
        "mean": float(cos_sims.mean()),
        "std": float(cos_sims.std()),
        "min": float(cos_sims.min()),
        "max": float(cos_sims.max()),
        "q25": float(np.percentile(cos_sims, 25)),
        "median": float(np.percentile(cos_sims, 50)),
        "q75": float(np.percentile(cos_sims, 75)),
    }


# ------------------------------------------------------------------ #
# Main pipeline                                                      #
# ------------------------------------------------------------------ #

def _embeddings_exist(output_dir: Path) -> bool:
    """Check if all embedding files already exist."""
    required = [
        "train_image_embeds.pt", "train_text_embeds.pt", "train_labels.pt",
        "val_image_embeds.pt", "val_text_embeds.pt", "val_labels.pt",
        "class_names.json",
    ]
    return all((output_dir / f).exists() for f in required)


def _gt_features_exist(output_dir: Path) -> bool:
    """Check if GT feature files already exist."""
    return (output_dir / "gt_features.pt").exists() and (output_dir / "gt_features_meta.json").exists()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(args.device)

    # ---- Stage 1: Embedding extraction ---- #
    if _embeddings_exist(output_dir):
        logger.info("Embeddings already exist in %s — skipping extraction", output_dir)
    else:
        logger.info("Extracting CLIP embeddings ...")
        from datasets import load_dataset
        from transformers import CLIPModel, CLIPProcessor

        ds_train = load_dataset("zh-plus/tiny-imagenet", split="train")
        ds_val = load_dataset("zh-plus/tiny-imagenet", split="valid")

        class_names = _resolve_class_names(ds_train)
        with open(output_dir / "class_names.json", "w", encoding="utf-8") as f:
            json.dump(class_names, f, indent=2, ensure_ascii=False)

        clip_model = CLIPModel.from_pretrained(args.clip_model)
        clip_model = clip_model.to(device).eval()  # type: ignore[assignment]
        processor = CLIPProcessor.from_pretrained(args.clip_model)
        templates = _load_templates(args.template_file)

        # Image embeddings
        train_img_embeds, train_labels = _extract_image_embeddings(
            ds_train, clip_model, processor, args.clip_batch_size, device,
        )
        val_img_embeds, val_labels = _extract_image_embeddings(
            ds_val, clip_model, processor, args.clip_batch_size, device,
        )

        # Text embeddings (paired with each image)
        random.seed(42)
        train_txt_embeds, _ = _extract_text_embeddings(
            ds_train, clip_model, processor, class_names, templates, args.clip_batch_size, device,
        )
        random.seed(43)
        val_txt_embeds, _ = _extract_text_embeddings(
            ds_val, clip_model, processor, class_names, templates, args.clip_batch_size, device,
        )

        # Save
        torch.save(train_img_embeds, output_dir / "train_image_embeds.pt")
        torch.save(train_txt_embeds, output_dir / "train_text_embeds.pt")
        torch.save(train_labels, output_dir / "train_labels.pt")
        torch.save(val_img_embeds, output_dir / "val_image_embeds.pt")
        torch.save(val_txt_embeds, output_dir / "val_text_embeds.pt")
        torch.save(val_labels, output_dir / "val_labels.pt")

        logger.info(
            "Saved embeddings: train=(%d, %d) val=(%d, %d)",
            train_img_embeds.shape[0], train_img_embeds.shape[1],
            val_img_embeds.shape[0], val_img_embeds.shape[1],
        )

        # Free GPU memory
        del clip_model, processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---- Stage 2: GT feature extraction ---- #
    if _gt_features_exist(output_dir):
        logger.info("GT features already exist in %s — skipping extraction", output_dir)
    else:
        logger.info("Extracting GT class feature directions ...")

        # Load cached embeddings
        train_img = torch.load(output_dir / "train_image_embeds.pt", weights_only=True).numpy()
        train_txt = torch.load(output_dir / "train_text_embeds.pt", weights_only=True).numpy()
        train_lbl = torch.load(output_dir / "train_labels.pt", weights_only=True).numpy()
        val_img = torch.load(output_dir / "val_image_embeds.pt", weights_only=True).numpy()
        val_txt = torch.load(output_dir / "val_text_embeds.pt", weights_only=True).numpy()
        val_lbl = torch.load(output_dir / "val_labels.pt", weights_only=True).numpy()

        n_classes = int(train_lbl.max()) + 1

        # Linear probing — image
        logger.info("Linear probing (image) ...")
        lp_img, img_probe_loss, img_probe_acc = _extract_gt_linear_probe(
            train_img, train_lbl, val_img, val_lbl,
        )
        logger.info("  accuracy=%.4f  loss=%.4f", img_probe_acc, img_probe_loss)

        # Linear probing — text
        logger.info("Linear probing (text) ...")
        lp_txt, txt_probe_loss, txt_probe_acc = _extract_gt_linear_probe(
            train_txt, train_lbl, val_txt, val_lbl,
        )
        logger.info("  accuracy=%.4f  loss=%.4f", txt_probe_acc, txt_probe_loss)

        # Mean embedding — image & text
        me_img = _extract_gt_mean_embedding(train_img, train_lbl, n_classes)
        me_txt = _extract_gt_mean_embedding(train_txt, train_lbl, n_classes)

        # Cross-modal similarity
        cross_modal_lp = _compute_cross_modal_similarity(lp_img, lp_txt)
        cross_modal_me = _compute_cross_modal_similarity(me_img, me_txt)

        # Save GT features
        gt_features = {
            "lp_image": lp_img,
            "lp_text": lp_txt,
            "me_image": me_img,
            "me_text": me_txt,
        }
        torch.save(gt_features, output_dir / "gt_features.pt")

        # Save metadata
        gt_meta = {
            "image_probe_loss": img_probe_loss,
            "image_probe_accuracy": img_probe_acc,
            "text_probe_loss": txt_probe_loss,
            "text_probe_accuracy": txt_probe_acc,
            "cross_modal_linear_probe": cross_modal_lp,
            "cross_modal_mean_embed": cross_modal_me,
        }
        with open(output_dir / "gt_features_meta.json", "w", encoding="utf-8") as f:
            json.dump(gt_meta, f, indent=2, ensure_ascii=False)

        logger.info("GT feature directions saved to %s", output_dir)
        logger.info(
            "Cross-modal cosine (LP): mean=%.4f std=%.4f | (ME): mean=%.4f std=%.4f",
            cross_modal_lp["mean"], cross_modal_lp["std"],
            cross_modal_me["mean"], cross_modal_me["std"],
        )

    logger.info("Done.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract CLIP embeddings and GT class feature directions from Tiny ImageNet",
    )
    parser.add_argument("--clip-model", type=str, default="openai/clip-vit-base-patch16")
    parser.add_argument("--clip-batch-size", type=int, default=256)
    parser.add_argument("--output-dir", type=str, default="outputs/tinyimagenet_features")
    parser.add_argument("--device", type=str, choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument("--template-file", type=str, default="template.txt")
    return parser.parse_args()


if __name__ == "__main__":
    main()
