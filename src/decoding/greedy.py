"""
Greedy decoding wrapper.

Uses model.generate with deterministic decoding (do_sample=False).
"""

from typing import Optional

import torch
import torch.nn as nn

from .base import BaseMitigator


class GreedyMitigator(BaseMitigator):
    """Greedy decoding (no sampling)."""

    name: str = "GreedyMitigator"

    def setup(self) -> None:
        return

    def cleanup(self) -> None:
        return

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        gen_kwargs = {
            "max_new_tokens": self.config.max_new_tokens,
            "do_sample": False,
            "temperature": self.config.temperature,
            "top_k": self.config.top_k,
            "top_p": self.config.top_p,
        }
        gen_kwargs.update(kwargs)

        if pixel_values is not None:
            gen_kwargs["pixel_values"] = pixel_values

        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **gen_kwargs,
        )
