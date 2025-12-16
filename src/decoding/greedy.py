from typing import Dict

import torch

from src.decoding.base import DecodeResult, DecodingStrategy


class GreedyDecoder(DecodingStrategy):
    """
    Standard greedy decoding wrapper.
    """

    name = "greedy"

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
        generated_ids = model.generate(
            **clean_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
        prompt_len = clean_inputs["input_ids"].shape[1]
        new_tokens = generated_ids[0][prompt_len:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return DecodeResult(text=text, token_ids=new_tokens)
