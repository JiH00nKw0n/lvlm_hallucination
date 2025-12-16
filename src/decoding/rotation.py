import math
from typing import Dict

import torch

from src.decoding.base import DecodeResult, DecodingStrategy


def _rotate_towards(vec: torch.Tensor, target: torch.Tensor, angle_rad: float) -> torch.Tensor:
    """
    Rotate vec toward target by angle_rad in the plane they span.
    """
    if angle_rad == 0:
        return vec

    vec_norm = torch.nn.functional.normalize(vec, dim=-1)
    target_norm = torch.nn.functional.normalize(target, dim=-1)

    dot = (vec_norm * target_norm).sum(dim=-1, keepdim=True).clamp(-1 + 1e-6, 1 - 1e-6)
    orth = target_norm - dot * vec_norm
    orth_norm = torch.nn.functional.normalize(orth, dim=-1)

    cos = math.cos(angle_rad)
    sin = math.sin(angle_rad)
    rotated_dir = cos * vec_norm + sin * orth_norm
    rotated_dir = torch.nn.functional.normalize(rotated_dir, dim=-1)
    return rotated_dir * vec.norm(dim=-1, keepdim=True)


class InstructionRotationDecoder(DecodingStrategy):
    """
    Contrastive decoding using rotated image features relative to instruction embedding.
    """

    def __init__(self, degrees: float = 5.0, contrast_scale: float = 1.0):
        self.degrees = degrees
        self.contrast_scale = contrast_scale
        self.name = f"instruction_rotation_{degrees:g}deg"

    @torch.no_grad()
    def decode(
        self,
        model,
        tokenizer,
        *,
        clean_inputs: Dict[str, torch.Tensor],
        max_new_tokens: int,
        **kwargs,
    ) -> DecodeResult:
        device = next(model.parameters()).device
        angle_rad = math.radians(self.degrees)

        input_ids = clean_inputs["input_ids"].to(device)
        attention_mask = clean_inputs.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=device)
        else:
            attention_mask = attention_mask.to(device)

        pixel_values = clean_inputs.get("pixel_values")
        if pixel_values is None:
            raise ValueError("pixel_values must be provided for rotation decoding.")
        pixel_values = pixel_values.to(device)

        # Build base embeddings
        embed_layer = model.get_input_embeddings()
        base_embeds = embed_layer(input_ids)

        # Extract image features and insert rotated versions
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
        instruction_vec = text_tokens.mean(dim=0, keepdim=True)

        toward_features = _rotate_towards(image_features, instruction_vec, angle_rad)
        away_features = _rotate_towards(image_features, instruction_vec, -angle_rad)

        embeds_toward = base_embeds.clone()
        embeds_away = base_embeds.clone()
        mask_expanded = image_mask.unsqueeze(-1).expand_as(base_embeds)
        embeds_toward = embeds_toward.masked_scatter(mask_expanded, toward_features)
        embeds_away = embeds_away.masked_scatter(mask_expanded, away_features)

        ids_toward = input_ids.clone()
        ids_away = input_ids.clone()
        attn_toward = attention_mask.clone()
        attn_away = attention_mask.clone()

        eos_token_id = tokenizer.eos_token_id or tokenizer.pad_token_id
        generated_tokens = []

        for _ in range(max_new_tokens):
            kwargs_common = {"use_cache": False, "return_dict": True}
            out_toward = model(
                inputs_embeds=embeds_toward,
                attention_mask=attn_toward,
                **kwargs_common,
            )
            out_away = model(
                inputs_embeds=embeds_away,
                attention_mask=attn_away,
                **kwargs_common,
            )

            logits_toward = out_toward.logits[:, -1, :]
            logits_away = out_away.logits[:, -1, :]
            mixed = logits_toward - self.contrast_scale * logits_away
            next_token = torch.argmax(mixed, dim=-1, keepdim=True)

            generated_tokens.append(next_token)

            next_embed = embed_layer(next_token)
            ids_toward = torch.cat([ids_toward, next_token], dim=-1)
            ids_away = torch.cat([ids_away, next_token], dim=-1)
            embeds_toward = torch.cat([embeds_toward, next_embed], dim=1)
            embeds_away = torch.cat([embeds_away, next_embed], dim=1)
            attn_toward = torch.cat([attn_toward, torch.ones_like(next_token)], dim=-1)
            attn_away = torch.cat([attn_away, torch.ones_like(next_token)], dim=-1)

            if eos_token_id is not None and next_token.item() == eos_token_id:
                break

        if not generated_tokens:
            return DecodeResult(text="", token_ids=None)

        gen_ids = torch.cat(generated_tokens, dim=-1)
        text = tokenizer.decode(gen_ids[0], skip_special_tokens=True).strip()
        return DecodeResult(text=text, token_ids=gen_ids)
