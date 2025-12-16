from typing import Dict

import torch

from src.decoding.base import DecodeResult, DecodingStrategy


class NoiseContrastiveDecoder(DecodingStrategy):
    """
    Contrastive greedy decoding between clean and noisy contexts.
    """

    name = "noise_contrastive"

    def __init__(self, noise_scale: float = 0.5):
        self.noise_scale = noise_scale

    @torch.no_grad()
    def decode(
        self,
        model,
        tokenizer,
        *,
        clean_inputs: Dict[str, torch.Tensor],
        noisy_inputs: Dict[str, torch.Tensor],
        max_new_tokens: int,
        noise_scale: float = None,
        use_cache: bool = False,
        **kwargs,
    ) -> DecodeResult:
        scale = self.noise_scale if noise_scale is None else noise_scale
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
                "pixel_values": clean_pixels if not use_cache or clean_past is None else None,
                "attention_mask": clean_mask,
                "use_cache": use_cache,
                "past_key_values": clean_past,
                "return_dict": True,
            }
            noisy_kwargs = {
                "input_ids": noisy_ids,
                "pixel_values": noisy_pixels if not use_cache or noisy_past is None else None,
                "attention_mask": noisy_mask,
                "use_cache": use_cache,
                "past_key_values": noisy_past,
                "return_dict": True,
            }

            clean_out = model(**clean_kwargs)
            noisy_out = model(**noisy_kwargs)
            clean_logits = clean_out.logits[:, -1, :]
            noisy_logits = noisy_out.logits[:, -1, :]
            clean_past = clean_out.past_key_values if use_cache else None
            noisy_past = noisy_out.past_key_values if use_cache else None

            contrastive_logits = clean_logits - scale * noisy_logits
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
            return DecodeResult(text="", token_ids=None)

        gen_ids = torch.cat(generated_tokens, dim=-1)
        text = tokenizer.decode(gen_ids[0], skip_special_tokens=True).strip()
        return DecodeResult(text=text, token_ids=gen_ids)
