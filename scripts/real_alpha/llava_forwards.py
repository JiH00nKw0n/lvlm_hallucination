"""LLaVA-1.5 HACL-style paired EOS-embedding forwards.

Loads ``llava-hf/llava-1.5-7b-hf`` and exposes two forwards that return the LLM
final-layer hidden state at the EOS position -- the HACL *global representation*
(Jiang et al., "Hallucination Augmented Contrastive Learning for Multimodal
Large Language Model", CVPR 2024):

    fwd_image_eos(pils)  -> (B, H) fp32 cpu   # [<image>(576 vis toks), EOS] -> Vicuna -> EOS hidden
    fwd_text_eos(texts)  -> (B, H) fp32 cpu   # [text toks, EOS]             -> SAME Vicuna -> EOS hidden

Both vectors live in the shared LM hidden space (H = 4096 for Vicuna-7B), so
their cosine is exactly the HACL image<->text similarity. We LEFT-pad so the
EOS (the final real token) is always at position ``-1`` -- robust to variable
text length and to the internal ``<image>`` -> 576-token expansion, which
happens mid-sequence and never after the appended EOS.

This is deliberately dependency-light (only ``transformers`` + ``torch``) so it
can be imported by the extractor without pulling the CLIP-specific
``extract_common`` machinery, which assumes a dual CLIP encoder.
"""
from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "llava-hf/llava-1.5-7b-hf"


class LlavaForwards:
    """Paired image/text EOS-hidden-state extractor for LLaVA-1.5."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        image_prompt: str = "<image>",
        max_text_len: int = 256,
    ) -> None:
        from transformers import AutoProcessor, LlavaForConditionalGeneration

        self.model_id = model_id
        self.device = torch.device(
            device if (torch.cuda.is_available() or device != "cuda") else "cpu"
        )
        self.dtype = dtype if self.device.type == "cuda" else torch.float32
        self.image_prompt = image_prompt
        self.max_text_len = max_text_len

        logger.info("loading %s (dtype=%s, device=%s) ...", model_id, self.dtype, self.device)
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.tok = self.processor.tokenizer
        # Left-pad so the appended EOS is always the final column (index -1) for
        # every sample regardless of length.
        self.tok.padding_side = "left"
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token

        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=self.dtype, low_cpu_mem_usage=True
        ).to(self.device).eval()

        self.emb_dim = int(self.model.config.text_config.hidden_size)
        self.eos_id = int(self.tok.eos_token_id)
        logger.info(
            "loaded LLaVA: emb_dim=%d eos_id=%d image_token_index=%s",
            self.emb_dim, self.eos_id, self.model.config.image_token_index,
        )

    # ------------------------------------------------------------------ utils
    def _append_eos(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """Append one EOS column on the right; with left padding this makes EOS
        the last real token (position -1) for every sample."""
        b = input_ids.shape[0]
        eos_col = torch.full((b, 1), self.eos_id, dtype=input_ids.dtype)
        one_col = torch.ones((b, 1), dtype=attention_mask.dtype)
        return (
            torch.cat([input_ids, eos_col], dim=1),
            torch.cat([attention_mask, one_col], dim=1),
        )

    # --------------------------------------------------------------- forwards
    @torch.no_grad()
    def fwd_image_eos(self, pils) -> torch.Tensor:
        """[<image>, EOS] through the full LLaVA model; return EOS final hidden."""
        pils = list(pils)
        prompts = [self.image_prompt] * len(pils)
        inputs = self.processor(
            images=pils, text=prompts, return_tensors="pt", padding=True
        )
        ids, mask = self._append_eos(inputs["input_ids"], inputs["attention_mask"])
        out = self.model(
            input_ids=ids.to(self.device),
            attention_mask=mask.to(self.device),
            pixel_values=inputs["pixel_values"].to(self.device, self.dtype),
            output_hidden_states=True,
            use_cache=False,
        )
        # hidden_states[-1]: (B, expanded_seq, H); EOS is the final token.
        return out.hidden_states[-1][:, -1, :].float().cpu()

    @torch.no_grad()
    def fwd_text_eos(self, texts) -> torch.Tensor:
        """[text, EOS] through the SAME language model; return EOS final hidden."""
        texts = [str(t) for t in texts]
        enc = self.tok(
            texts, return_tensors="pt", padding=True, truncation=True,
            max_length=self.max_text_len,
        )
        ids, mask = self._append_eos(enc["input_ids"], enc["attention_mask"])
        out = self.model(  # no pixel_values -> pure text path through the LM
            input_ids=ids.to(self.device),
            attention_mask=mask.to(self.device),
            output_hidden_states=True,
            use_cache=False,
        )
        return out.hidden_states[-1][:, -1, :].float().cpu()
