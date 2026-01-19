"""
VISTA Visual Steering Vector computation.

Reference: VISTA/steering_vector.py

VISTA computes VSV on-the-fly per image:
    1. pos_kwargs: prompt with image
    2. neg_kwargs: prompt without image (null prompt)
    3. Direction: pos - neg (각 layer의 last token)
    4. PCA로 최종 방향 계산
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from .pca import PCA


def _get_layers(model: nn.Module) -> nn.ModuleList:
    """
    Get transformer layers from model.

    Reference: VISTA/llm_layers.py:get_layers

    Supports: LLaVA, Qwen2-VL
    """
    # LLaVA / LLaVA-NeXT
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return model.model.layers
    # Qwen2-VL
    if hasattr(model, 'layers'):
        return model.layers
    raise ValueError(f"Cannot find layers in model: {type(model)}")


@dataclass
class ResidualStream:
    """
    Container for hidden states from each layer.

    Reference: VISTA/steering_vector.py:10-11
    """
    hidden: List[List[torch.Tensor]]

    def __init__(self):
        self.hidden = []


class ForwardTrace:
    """
    Trace object to store activations during forward pass.

    Reference: VISTA/steering_vector.py:14-19
    """

    def __init__(self):
        self.residual_stream: ResidualStream = ResidualStream()
        self.attentions: Optional[torch.Tensor] = None


class ForwardTracer:
    """
    Context manager to register hooks and collect hidden states.

    Reference: VISTA/steering_vector.py:22-88

    Usage:
        forward_trace = ForwardTrace()
        with ForwardTracer(model, forward_trace):
            _ = model(**kwargs)
        hidden_states = forward_trace.residual_stream.hidden
    """

    def __init__(self, model: nn.Module, forward_trace: ForwardTrace):
        """
        Initialize ForwardTracer.

        Args:
            model: The LLM model (not VLM wrapper)
            forward_trace: ForwardTrace object to store results
        """
        self._model = model
        self._forward_trace = forward_trace
        self._layers = _get_layers(model)
        self._hooks = []

    def __enter__(self):
        """Register forward hooks on each layer."""
        self._register_forward_hooks()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Remove hooks and process collected activations."""
        for hook in self._hooks:
            hook.remove()

        if exc_type is None:
            residual_stream = self._forward_trace.residual_stream

            # Remove empty first element if exists
            if residual_stream.hidden and residual_stream.hidden[0] == []:
                residual_stream.hidden.pop(0)

            # Stack activations for each layer
            for i, layer_acts in enumerate(residual_stream.hidden):
                if layer_acts:
                    residual_stream.hidden[i] = torch.cat(layer_acts, dim=0)
                else:
                    residual_stream.hidden[i] = torch.zeros(1)

            # Stack all layers: [num_layers, ...]
            if residual_stream.hidden:
                residual_stream.hidden = torch.stack(residual_stream.hidden).transpose(0, 1)

    def _register_forward_hooks(self):
        """
        Register hooks on each transformer layer.

        Reference: VISTA/steering_vector.py:59-88
        """
        residual_stream = self._forward_trace.residual_stream

        def store_activations(layer_num: int):
            def hook(module, inp, out):
                if isinstance(out, tuple):
                    out = out[0]
                out = out.float().to("cpu", non_blocking=True)

                while len(residual_stream.hidden) < layer_num + 1:
                    residual_stream.hidden.append([])
                try:
                    residual_stream.hidden[layer_num].append(out)
                except IndexError:
                    print(f"IndexError: len={len(residual_stream.hidden)}, layer_num={layer_num}")
            return hook

        for i, layer in enumerate(self._layers):
            hook = layer.register_forward_hook(store_activations(i + 1))
            self._hooks.append(hook)


def get_hiddenstates(model: nn.Module, kwargs_list: List[List[dict]]) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Extract hidden states from model for each (neg, pos) pair.

    Reference: VISTA/steering_vector.py:91-110

    Args:
        model: The LLM model
        kwargs_list: List of [(neg_kwargs, pos_kwargs), ...] pairs

    Returns:
        List of (neg_hidden, pos_hidden) tuples
        Each hidden: [num_layers, hidden_dim] (last token per layer)
    """
    h_all = []

    for example_id in range(len(kwargs_list)):
        embeddings_for_all_styles = []

        for style_id in range(len(kwargs_list[example_id])):
            forward_trace = ForwardTrace()

            with ForwardTracer(model, forward_trace):
                _ = model(
                    use_cache=True,
                    **kwargs_list[example_id][style_id],
                )
                h = forward_trace.residual_stream.hidden

            # Extract last token from each layer
            embedding_token = []
            for layer in range(len(h)):
                embedding_token.append(h[layer][:, -1])

            embedding_token = torch.cat(embedding_token, dim=0).cpu().clone()
            embeddings_for_all_styles.append(embedding_token)

        h_all.append(tuple(embeddings_for_all_styles))

    return h_all


def obtain_vsv(
    model: nn.Module,
    kwargs_list: List[List[dict]],
    rank: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Visual Steering Vector from (neg, pos) prompt pairs.

    Reference: VISTA/steering_vector.py:113-129

    VISTA Direction: pos - neg = (with_image) - (without_image)

    Args:
        model: The LLM model
        kwargs_list: List of [(neg_kwargs, pos_kwargs)] pairs
        rank: PCA rank (default: 1)

    Returns:
        (direction, neg_embedding)
        - direction: [num_layers, hidden_dim] - VSV direction
        - neg_embedding: [num_layers, hidden_dim] - mean negative embedding
    """
    hidden_states = get_hiddenstates(model, kwargs_list)
    num_demonstration = len(hidden_states)

    neg_all = []
    pos_all = []
    hidden_states_all = []

    for demonstration_id in range(num_demonstration):
        # Direction: pos - neg = (with_image) - (without_image)
        h = hidden_states[demonstration_id][1].view(-1) - hidden_states[demonstration_id][0].view(-1)
        hidden_states_all.append(h)
        neg_all.append(hidden_states[demonstration_id][0].view(-1))
        pos_all.append(hidden_states[demonstration_id][1].view(-1))

    fit_data = torch.stack(hidden_states_all)
    neg_emb = torch.stack(neg_all).mean(0)
    pos_emb = torch.stack(pos_all).mean(0)

    # PCA for direction
    pca = PCA(n_components=rank).to(fit_data.device).fit(fit_data.float())

    direction = (pca.components_.sum(dim=0, keepdim=True) + pca.mean_).mean(0).view(
        hidden_states[demonstration_id][0].size(0),
        hidden_states[demonstration_id][0].size(1)
    )

    return direction, neg_emb.view(
        hidden_states[demonstration_id][0].size(0),
        hidden_states[demonstration_id][0].size(1)
    )
