import os
from typing import Iterable, Optional

import torch
from torch import Tensor
from transformers import AutoTokenizer, PreTrainedModel

from src.integrations.saebench_autointerp import (
    get_bos_pad_eos_mask,
    load_and_tokenize_dataset,
    _sae_dense_acts,
)


def load_tokenized_text_dataset(
    dataset: Iterable[dict],
    tokenizer: AutoTokenizer,
    ctx_len: int,
    total_tokens: int,
    text_column: str,
    cache_path: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    if cache_path and os.path.exists(cache_path):
        tokens = torch.load(cache_path)
    else:
        tokens = load_and_tokenize_dataset(
            dataset,
            tokenizer,
            ctx_len=ctx_len,
            num_tokens=total_tokens,
            column_name=text_column,
        )
        if cache_path:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            torch.save(tokens, cache_path)
    if device is not None:
        tokens = tokens.to(device)
    return tokens


def compute_l0(
    tokens: Tensor,
    model: PreTrainedModel,
    sae: object,
    tokenizer: AutoTokenizer,
    batch_size: int,
    hidden_state_index: int,
) -> float:
    l0_values = []
    for i in range(0, tokens.shape[0], batch_size):
        tokens_bl = tokens[i : i + batch_size]
        attention_mask = (
            (tokens_bl != tokenizer.pad_token_id).long()
            if tokenizer.pad_token_id is not None
            else None
        )
        outputs = model(
            input_ids=tokens_bl.to(model.device),
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        hidden_states = outputs.hidden_states[hidden_state_index]
        acts = _sae_dense_acts(sae, hidden_states)
        active = (acts > 0).to(dtype=torch.float32)
        mask = get_bos_pad_eos_mask(tokens_bl, tokenizer).to(active.device)
        active = active * mask[:, :, None]
        l0 = active.sum(dim=-1)
        l0_values.append(l0.flatten())
    l0_all = torch.cat(l0_values, dim=0)
    return l0_all.mean().item()


def _get_module_by_path(model: PreTrainedModel, module_path: str):
    return model.get_submodule(module_path)


def _replace_output_hook(replacement: Tensor):
    def hook_fn(module, inputs, outputs):
        if isinstance(outputs, tuple):
            return (replacement,) + outputs[1:]
        return replacement

    return hook_fn


def _sae_reconstruct(sae: object, hidden_states: Tensor) -> Tensor:
    encoded = sae.encode(hidden_states)
    if isinstance(encoded, tuple) and len(encoded) == 2:
        top_acts, top_indices = encoded
        return sae.decode(top_acts, top_indices)
    return sae.decode(encoded)


@torch.no_grad()
def compute_loss_recovered(
    tokens: Tensor,
    model: PreTrainedModel,
    sae: object,
    tokenizer: AutoTokenizer,
    batch_size: int,
    hidden_state_index: int,
    hook_module_path: str,
    exclude_special_tokens: bool = True,
) -> dict[str, float]:
    ce_losses_base = []
    ce_losses_abl = []
    ce_losses_sae = []

    for i in range(0, tokens.shape[0], batch_size):
        tokens_bl = tokens[i : i + batch_size]
        attention_mask = (
            (tokens_bl != tokenizer.pad_token_id).long()
            if tokenizer.pad_token_id is not None
            else None
        )
        labels = tokens_bl.clone()
        if exclude_special_tokens:
            if tokenizer.pad_token_id is not None:
                labels[labels == tokenizer.pad_token_id] = -100
            if tokenizer.eos_token_id is not None:
                labels[labels == tokenizer.eos_token_id] = -100
            if tokenizer.bos_token_id is not None:
                labels[labels == tokenizer.bos_token_id] = -100

        base_outputs = model(
            input_ids=tokens_bl.to(model.device),
            attention_mask=attention_mask,
            labels=labels.to(model.device),
            output_hidden_states=True,
            use_cache=False,
        )
        hidden_states = base_outputs.hidden_states[hidden_state_index]

        ce_losses_base.append(base_outputs.loss.detach())

        module = _get_module_by_path(model, hook_module_path)

        zeros = torch.zeros_like(hidden_states)
        handle = module.register_forward_hook(_replace_output_hook(zeros))
        try:
            abl_outputs = model(
                input_ids=tokens_bl.to(model.device),
                attention_mask=attention_mask,
                labels=labels.to(model.device),
                use_cache=False,
            )
        finally:
            handle.remove()
        ce_losses_abl.append(abl_outputs.loss.detach())

        recon = _sae_reconstruct(sae, hidden_states)

        handle = module.register_forward_hook(_replace_output_hook(recon))
        try:
            sae_outputs = model(
                input_ids=tokens_bl.to(model.device),
                attention_mask=attention_mask,
                labels=labels.to(model.device),
                use_cache=False,
            )
        finally:
            handle.remove()
        ce_losses_sae.append(sae_outputs.loss.detach())

    ce_base = torch.stack(ce_losses_base).mean().item()
    ce_abl = torch.stack(ce_losses_abl).mean().item()
    ce_sae = torch.stack(ce_losses_sae).mean().item()

    denom = max(ce_abl - ce_base, 1e-8)
    loss_recovered = (ce_abl - ce_sae) / denom
    return {
        "ce_loss_without_sae": ce_base,
        "ce_loss_with_ablation": ce_abl,
        "ce_loss_with_sae": ce_sae,
        "loss_recovered": loss_recovered,
    }
