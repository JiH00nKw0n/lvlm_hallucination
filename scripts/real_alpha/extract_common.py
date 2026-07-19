"""Shared model-forward and GPU-decode helpers for embedding extractors.

Centralizes the per-backend forward closures that were previously copied
between extract_clip_{coco,cc3m}_cache.py / extract_imagenet_cache.py, and
adds:
  * an `aimv2` branch (Aimv2Model.get_image_features / get_text_features,
    `projection_dim` read from the checkpoint config — never hardcoded);
  * a GPU JPEG-decode + preprocess path (`decode_to_device` +
    `GpuImagePreprocessor`) so extraction on CPU-starved boxes (2 cores)
    moves the decode bottleneck onto the GPU (nvJPEG via torchvision).

The legacy extractors are left untouched; new extractors import from here.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class ModelForwards:
    fwd_img: Callable          # list[PIL] -> Tensor(B, d) cpu fp32
    fwd_txt: Callable          # list[str] -> Tensor(B, d) cpu fp32
    fwd_pixels: Callable | None  # Tensor(B,3,H,W) on device -> Tensor(B, d) cpu fp32
    emb_dim: int
    kind: str                  # "clip" | "siglip" | "aimv2" | "openclip"
    image_processor: object | None  # HF image processor (None for openclip)


def load_model_forwards(model_id: str, device: torch.device,
                        backend: str = "transformers",
                        pretrained: str = "") -> ModelForwards:
    if backend == "openclip":
        import open_clip
        logger.info("Loading OpenCLIP %s (pretrained=%s)", model_id, pretrained or None)
        model, _, oc_preprocess = open_clip.create_model_and_transforms(
            model_id, pretrained=pretrained or None, device=device)
        model.eval()
        oc_tokenizer = open_clip.get_tokenizer(model_id)
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224, device=device)
            emb_dim = int(model.encode_image(dummy).shape[-1])

        @torch.no_grad()
        def oc_fwd_img(pils):
            batch = torch.stack([oc_preprocess(p) for p in pils]).to(device)  # type: ignore[operator]
            return model.encode_image(batch).float().cpu()

        @torch.no_grad()
        def oc_fwd_txt(texts):
            tokens = oc_tokenizer(texts).to(device)
            return model.encode_text(tokens).float().cpu()

        return ModelForwards(oc_fwd_img, oc_fwd_txt, None, emb_dim, "openclip", None)

    from transformers import AutoModel, AutoProcessor
    logger.info("Loading transformers %s", model_id)
    tmodel = AutoModel.from_pretrained(model_id).to(device).eval()
    tmodel.requires_grad_(False)
    processor = AutoProcessor.from_pretrained(model_id)
    mtype = getattr(tmodel.config, "model_type", "")
    kind = {"siglip": "siglip", "siglip2": "siglip", "aimv2": "aimv2"}.get(mtype, "clip")

    if kind == "siglip":
        emb_dim = tmodel.config.vision_config.hidden_size
    else:
        emb_dim = int(tmodel.config.projection_dim)

    tokenizer = getattr(processor, "tokenizer", processor)
    image_processor = getattr(processor, "image_processor", None)

    def _tokenize(texts):
        if kind == "siglip":
            return processor(text=texts, return_tensors="pt",
                             padding="max_length", truncation=True, max_length=64)
        # clip / aimv2: pad to longest, truncate at the tokenizer's own
        # model_max_length (77 for CLIP-style vocabs — not hardcoded here).
        return tokenizer(texts, return_tensors="pt", padding=True, truncation=True)

    @torch.no_grad()
    def fwd_pixels(pixel_values: torch.Tensor) -> torch.Tensor:
        if kind == "aimv2":
            return tmodel.get_image_features(pixel_values=pixel_values).float().cpu()
        v_out = tmodel.vision_model(pixel_values=pixel_values)
        if kind == "siglip":
            return v_out.pooler_output.float().cpu()
        return tmodel.visual_projection(v_out.pooler_output).float().cpu()

    @torch.no_grad()
    def fwd_img(pils):
        inputs = processor(images=pils, return_tensors="pt")
        return fwd_pixels(inputs["pixel_values"].to(device))

    @torch.no_grad()
    def fwd_txt(texts):
        tok = _tokenize(texts)
        input_ids = tok["input_ids"].to(device)
        attn = tok.get("attention_mask")
        if attn is not None:
            attn = attn.to(device)
        if kind == "aimv2":
            return tmodel.get_text_features(
                input_ids=input_ids, attention_mask=attn).float().cpu()
        t_out = tmodel.text_model(input_ids=input_ids, attention_mask=attn)
        if kind == "siglip":
            return t_out.pooler_output.float().cpu()
        return tmodel.text_projection(t_out.pooler_output).float().cpu()

    return ModelForwards(fwd_img, fwd_txt, fwd_pixels, emb_dim, kind, image_processor)


# ---------------------------------------------------------------------------
# GPU decode + preprocess
# ---------------------------------------------------------------------------

_PIL_RESAMPLE_TO_TV = {0: "nearest", 1: "lanczos", 2: "bilinear", 3: "bicubic"}


def decode_to_device(data: bytes, device: torch.device) -> torch.Tensor | None:
    """Decode raw image bytes to a uint8 (3, H, W) tensor on `device`.

    JPEG goes through nvJPEG on CUDA (zero CPU decode cost); PNG/other
    formats decode on CPU via torchvision, PIL as the last resort.
    Returns None if the sample is undecodable.
    """
    from torchvision.io import ImageReadMode, decode_image, decode_jpeg
    buf = torch.frombuffer(bytearray(data), dtype=torch.uint8)
    if data[:2] == b"\xff\xd8" and device.type == "cuda":
        try:
            out = decode_jpeg(buf, mode=ImageReadMode.RGB, device=device)
            return out if isinstance(out, torch.Tensor) else out[0]
        except Exception:
            pass
    try:
        return decode_image(buf, mode=ImageReadMode.RGB).to(device)
    except Exception:
        pass
    try:
        from PIL import Image
        pil = Image.open(io.BytesIO(data)).convert("RGB")
        arr = np.asarray(pil)
        return torch.from_numpy(arr).permute(2, 0, 1).contiguous().to(device)
    except Exception:
        return None


class GpuImagePreprocessor:
    """Replicates an HF image processor's resize/crop/normalize on-device.

    Reads the processor's own config (size, crop_size, resample, mean, std,
    rescale factor) so it stays model-agnostic; small interpolation
    differences vs PIL are acceptable because every embedding in a given
    cache is produced by the same path (verified against the CPU processor
    by the extractor's smoke check).
    """

    def __init__(self, image_processor, device: torch.device):
        from torchvision.transforms.v2 import functional as F
        self._F = F
        size = getattr(image_processor, "size", None) or {}
        self.resize_size: list[int]
        if "shortest_edge" in size:
            self.resize_size = [int(size["shortest_edge"])]
        elif "height" in size:
            self.resize_size = [int(size["height"]), int(size["width"])]
        else:
            self.resize_size = [224]
        crop = getattr(image_processor, "crop_size", None)
        self.crop_size = None
        if getattr(image_processor, "do_center_crop", False) and crop:
            self.crop_size = [int(crop["height"]), int(crop["width"])]
        from torchvision.transforms import InterpolationMode
        resample = _PIL_RESAMPLE_TO_TV.get(int(getattr(image_processor, "resample", 3)), "bicubic")
        self.interp = {
            "nearest": InterpolationMode.NEAREST,
            "bilinear": InterpolationMode.BILINEAR,
            "bicubic": InterpolationMode.BICUBIC,
            "lanczos": InterpolationMode.BICUBIC,  # tv has no lanczos; closest
        }[resample]
        rescale = float(getattr(image_processor, "rescale_factor", 1 / 255))
        mean = torch.tensor(image_processor.image_mean, device=device).view(3, 1, 1)
        std = torch.tensor(image_processor.image_std, device=device).view(3, 1, 1)
        self.rescale = rescale
        self.mean, self.std = mean, std

    def __call__(self, img_u8: torch.Tensor) -> torch.Tensor:
        """uint8 (C,H,W) on device → float normalized (3,h,w)."""
        x = img_u8
        if x.shape[0] == 1:
            x = x.expand(3, -1, -1)
        elif x.shape[0] == 4:
            x = x[:3]
        x = self._F.resize(x, self.resize_size, interpolation=self.interp, antialias=True)
        if self.crop_size is not None:
            x = self._F.center_crop(x, self.crop_size)
        x = x.float().mul_(self.rescale)
        return (x - self.mean) / self.std


__all__ = ["ModelForwards", "load_model_forwards", "decode_to_device",
           "GpuImagePreprocessor"]
