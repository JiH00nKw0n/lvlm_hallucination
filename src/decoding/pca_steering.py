from typing import Dict, Optional

import torch

from src.decoding.base import DecodeResult, DecodingStrategy


class PcaSteeringDecoder(DecodingStrategy):
    """
    Contrast clean vs. PCA-steered image embeddings.

    Steps:
      1) Load PCA components/means (per vision layer).
      2) Compute instruction vector (mean of text token embeddings).
      3) Project instruction onto PCA axes, pick top-k (by abs weight).
      4) Remove those components from image embeddings, then contrast logits:
         clean_logits - contrast_scale * steered_logits.
    """

    name = "pca_steering"

    def __init__(
            self,
            pca_path: str,
            pca_layer: Optional[int] = None,
            top_k: int = 4,
            contrast_scale: float = 1.0,
            use_cache: bool = False,
            source: str = "text",
    ):
        self.pca_path = pca_path
        self.pca_layer = pca_layer
        self.top_k = top_k
        self.contrast_scale = contrast_scale
        self.use_cache = use_cache
        self.source = source
        self.name = f"pca_steering_{source}"

        self.components: Dict[int, torch.Tensor] = {}
        self.means: Dict[int, torch.Tensor] = {}
        self._loaded = False

    def _load_pca(self, device: torch.device, dtype: torch.dtype):
        if self._loaded:
            return
        data = torch.load(self.pca_path, map_location="cpu")
        comps = data.get("components") or {}
        means = data.get("means") or {}
        # components can be dict[int, Tensor] or Tensor; normalize to dict
        if isinstance(comps, dict):
            self.components = {int(k): v.to(device=device, dtype=dtype) for k, v in comps.items()}
        else:
            self.components = {0: comps.to(device=device, dtype=dtype)}
        if isinstance(means, dict):
            self.means = {int(k): v.to(device=device, dtype=dtype) for k, v in means.items()}
        else:
            self.means = {0: means.to(device=device, dtype=dtype)}
        self._loaded = True

    def _pick_layer(self) -> int:
        if self.pca_layer is not None:
            return self.pca_layer
        if not self.components:
            return 0
        return max(self.components.keys())

    def _steer_image_tokens(
            self,
            image_tokens: torch.Tensor,
            instr_vec: torch.Tensor,
            layer_idx: int,
    ) -> torch.Tensor:
        comps = self.components[layer_idx]  # (num_pc, hidden)
        mean = self.means[layer_idx]  # (hidden,)
        # center instruction and image tokens
        instr_center = instr_vec - mean
        weights = torch.matmul(instr_center, comps.T).squeeze(0)
        topk = min(self.top_k, comps.shape[0])
        if topk == 0:
            return image_tokens
        _, idx = torch.topk(weights.abs(), k=topk)
        pcs = comps[idx]  # (k, hidden), orthonormal from PCA

        img_center = image_tokens - mean
        proj = torch.matmul(img_center, pcs.T)  # (num_img, k)
        recon = torch.matmul(proj, pcs)  # (num_img, hidden)
        steered_center = img_center - recon

        # preserve per-token norm
        orig_norm = img_center.norm(dim=-1, keepdim=True)
        steered_norm = steered_center.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        steered_center = steered_center * (orig_norm / steered_norm)

        steered = steered_center + mean
        return steered

    @torch.no_grad()
    def decode(
            self,
            model,
            tokenizer,
            *,
            clean_inputs: Dict[str, torch.Tensor],
            max_new_tokens: int,
            use_cache: Optional[bool] = None,
            **kwargs,
    ) -> DecodeResult:
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        self._load_pca(device=device, dtype=dtype)
        layer_idx = self._pick_layer()
        cache_flag = self.use_cache if use_cache is None else use_cache

        input_ids = clean_inputs["input_ids"].to(device)
        attention_mask = clean_inputs.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=device)
        else:
            attention_mask = attention_mask.to(device)
        pixel_values = clean_inputs.get("pixel_values")
        if pixel_values is None:
            raise ValueError("pixel_values must be provided for PCA steering.")
        pixel_values = pixel_values.to(device)

        embed_layer = model.get_input_embeddings()
        base_embeds = embed_layer(input_ids)

        # image features (projector output)
        image_features_list = model.model.get_image_features(pixel_values=pixel_values)
        image_features = torch.cat(image_features_list, dim=0).to(device, base_embeds.dtype)

        image_token_id = model.config.image_token_id
        image_mask = (input_ids == image_token_id)
        if image_mask.sum() != image_features.shape[0]:
            raise ValueError(
                f"Image tokens ({image_mask.sum().item()}) do not match image features ({image_features.shape[0]})."
            )

        text_mask = (attention_mask.bool() & ~image_mask)
        text_tokens = base_embeds[text_mask]
        if text_tokens.numel() == 0:
            raise ValueError("No text tokens found to compute instruction embedding.")
        instr_vec = text_tokens.mean(dim=0, keepdim=True)

        steered_image = self._steer_image_tokens(image_features, instr_vec, layer_idx)

        # Build two embed streams
        mask_expanded = image_mask.unsqueeze(-1).expand_as(base_embeds)
        embeds_clean = base_embeds.clone().masked_scatter(mask_expanded, image_features)
        embeds_steer = base_embeds.clone().masked_scatter(mask_expanded, steered_image)

        ids_clean = input_ids.clone()
        ids_steer = input_ids.clone()
        attn_clean = attention_mask.clone()
        attn_steer = attention_mask.clone()

        eos_token_id = tokenizer.eos_token_id or tokenizer.pad_token_id
        generated_tokens = []
        past_clean = None
        past_steer = None

        for _ in range(max_new_tokens):
            common = {"use_cache": cache_flag, "return_dict": True}
            out_clean = model(
                inputs_embeds=embeds_clean,
                attention_mask=attn_clean,
                past_key_values=past_clean,
                **common,
            )
            out_steer = model(
                inputs_embeds=embeds_steer,
                attention_mask=attn_steer,
                past_key_values=past_steer,
                **common,
            )
            logits_clean = out_clean.logits[:, -1, :]
            logits_steer = out_steer.logits[:, -1, :]
            past_clean = out_clean.past_key_values if cache_flag else None
            past_steer = out_steer.past_key_values if cache_flag else None

            contrastive_logits = logits_clean - self.contrast_scale * logits_steer
            next_token = torch.argmax(contrastive_logits, dim=-1, keepdim=True)
            generated_tokens.append(next_token)

            next_embed = embed_layer(next_token)
            ids_clean = torch.cat([ids_clean, next_token], dim=-1)
            ids_steer = torch.cat([ids_steer, next_token], dim=-1)
            embeds_clean = torch.cat([embeds_clean, next_embed], dim=1)
            embeds_steer = torch.cat([embeds_steer, next_embed], dim=1)
            attn_clean = torch.cat([attn_clean, torch.ones_like(next_token)], dim=-1)
            attn_steer = torch.cat([attn_steer, torch.ones_like(next_token)], dim=-1)

            if eos_token_id is not None and next_token.item() == eos_token_id:
                break

        if not generated_tokens:
            return DecodeResult(text="", token_ids=None)

        gen_ids = torch.cat(generated_tokens, dim=-1)
        text = tokenizer.decode(gen_ids[0], skip_special_tokens=True).strip()
        return DecodeResult(text=text, token_ids=gen_ids)
