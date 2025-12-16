import abc
from dataclasses import dataclass
from typing import Dict, Optional

import torch


@dataclass
class DecodeResult:
    """
    Lightweight container for a decoded string and optional token ids.
    """

    text: str
    token_ids: Optional[torch.Tensor] = None


class DecodingStrategy(abc.ABC):
    """
    Base interface for decoding strategies.
    """

    name: str = "base"

    @abc.abstractmethod
    def decode(
        self,
        model,
        tokenizer,
        *,
        clean_inputs: Dict[str, torch.Tensor],
        **kwargs,
    ) -> DecodeResult:
        """
        Run decoding given already-prepared model inputs (including image tokens).
        """
        raise NotImplementedError
