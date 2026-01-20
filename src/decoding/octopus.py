"""
Octopus: Dynamic Strategy Selection

Uses the reference Octopus generation logic (avisc_sample.py) with a MyModel-style
policy wrapper that outputs action logits and model outputs each step.

Reference:
    - Octopus/eval_bench/train_token_amber.py:132-256 (MyModel classifier)
    - Octopus/avisc_utils/avisc_sample.py (custom sample/generate loop)
    - Octopus/eval_bench/train_token_amber.py:611-631 (DPO loss)

Key Implementation Notes:
    1. Policy wrapper mirrors MyModel forward (masking inputs_embeds, LLaMA forward)
    2. Classifier mirrors MyModel (TransformerEncoder + MLP, float32)
    3. Generation delegates to Octopus/avisc_utils/avisc_sample.py

Supports: LLaVA, LLaVA-NeXT (reference Octopus code targets LLaVA)
"""

import math
import types
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.generation.utils import GenerateOutput
from transformers.modeling_outputs import CausalLMOutputWithPast

from .base import TrainableMitigator, add_diffusion_noise, get_image_token_indices, ModelHelper


class OctopusClassifier(nn.Module):
    """
    Octopus action classifier.

    Matches reference MyModel classifier implementation:
        - cls_token (float32)
        - TransformerEncoder (nhead=2, num_layers=2, batch_first=False)
        - MLP: d_model -> d_model//4 -> num_classes (LeakyReLU)
    """

    def __init__(
            self,
            d_model: int = 4096,
            num_classes: int = 4,
            nhead: int = 2,
            num_layers: int = 2,
            n_query: int = 4,
            bt: int = 1,
    ):
        super().__init__()
        self.n_query = n_query
        self.bt = bt

        self.cls_token = nn.Parameter(torch.randn(1, d_model).to(dtype=torch.float32))
        self.queries = nn.Parameter(torch.randn(n_query, d_model).to(dtype=torch.float32))

        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead).to(dtype=torch.float32)
        encoder_layer.apply(self.init_weights)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers).to(dtype=torch.float32)

        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model // 4).to(dtype=torch.float32),
            nn.LeakyReLU(),
            nn.Linear(d_model // 4, num_classes).to(dtype=torch.float32),
        )

        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
                nn.init.constant_(layer.bias, 0)

    @staticmethod
    def init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
            if m.bias is not None:
                fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(m.weight)
                bound = 1 / math.sqrt(fan_in)
                torch.nn.init.uniform_(m.bias, -bound, bound)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        inputs_transformer = torch.cat(
            (self.cls_token.unsqueeze(0).expand(self.bt, -1, -1), hidden_states.to(dtype=torch.float32)),
            dim=1,
        )
        out_transformer = self.transformer(inputs_transformer)
        logits_mlp = self.mlp(out_transformer)
        return logits_mlp[:, 0, :]


class OctopusPolicy(nn.Module):
    """
    MyModel-style policy wrapper from the reference implementation.

    Returns (action_logits, model_outputs) in forward.
    """

    def __init__(
            self,
            model: nn.Module,
            classifier: OctopusClassifier,
            model_type: str = "llava",
    ):
        super().__init__()
        self.model = model
        self.Llama = model.model
        self.classifier = classifier
        self.model_type = ModelHelper.normalize_model_type(model_type)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            past_key_values=None,
            inputs_embeds=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            images=None,
            pixel_values=None,
            images_cd=None,
            cd_beta=None,
            cd_alpha=None,
            img_idx=None,
            mask_idx=None,
            return_dict=None,
            kernel_size=None,
            use_avisc=None,
            layer_gamma=None,
            masking_scheme=None,
            lamb=None,
            question_id=None,
            use_m3id=None,
            is_eval=None,
            temp=None,
            image_grid_thw=None,
            cache_position=None,
            position_ids=None,
            rope_deltas=None,
    ) -> Tuple[torch.Tensor, Union[CausalLMOutputWithPast, Tuple[torch.Tensor, ...]]]:
        if pixel_values is None and images is not None:
            pixel_values = images

        if attention_mask is None and input_ids is not None:
            attention_mask = torch.ones(input_ids.shape[:2], dtype=torch.long, device=input_ids.device)

        is_qwen = self.model_type in ("qwen2_vl", "qwen2_5_vl")
        if not is_qwen:
            if input_ids is not None:
                input_ids, attention_mask, past_key_values, inputs_embeds, labels = (
                    self.model.prepare_inputs_labels_for_multimodal(
                        input_ids, attention_mask, past_key_values, labels, images
                    )
                )

            if mask_idx is not None and past_key_values is None and inputs_embeds is not None:
                img_start, _ = get_image_token_indices(
                    input_ids,
                    model_type=self.model_type,
                    config=getattr(self.model, "config", None),
                )
                for input_embed, idx in zip(inputs_embeds, mask_idx):
                    if masking_scheme is None:
                        masking_scheme = "zeros"
                    pos = idx + img_start
                    if masking_scheme.lower() == "ones":
                        input_embed[pos] = 1.0
                    elif masking_scheme.lower() == "zeros":
                        input_embed[pos] = 0.0
                    elif masking_scheme.lower() == "noise":
                        input_embed[pos] = torch.randn(
                            input_embed[pos].size(),
                            dtype=input_embed.dtype,
                            device=input_embed.device,
                        )
                    else:
                        input_embed[pos] = 0.0

            with torch.no_grad():
                outputs = self.Llama(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    use_cache=True,
                    output_attentions=True,
                    output_hidden_states=False,
                    return_dict=True,
                )

            hidden_states = outputs[0]
            logits = self.model.lm_head(hidden_states)
        else:
            if input_ids is None:
                raise ValueError("OctopusPolicy requires input_ids for Qwen2-VL inputs.")

            inputs_embeds = self.model.model.get_input_embeddings()(input_ids)
            if pixel_values is not None:
                image_embeds = self.model.model.get_image_features(pixel_values, image_grid_thw)
                image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
                image_mask, _ = self.model.model.get_placeholder_mask(
                    input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
                )
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            if mask_idx is not None and past_key_values is None:
                img_start, _ = get_image_token_indices(
                    input_ids,
                    model_type=self.model_type,
                    config=getattr(self.model, "config", None),
                )
                for input_embed, idx in zip(inputs_embeds, mask_idx):
                    if masking_scheme is None:
                        masking_scheme = "zeros"
                    pos = idx + img_start
                    if masking_scheme.lower() == "ones":
                        input_embed[pos] = 1.0
                    elif masking_scheme.lower() == "zeros":
                        input_embed[pos] = 0.0
                    elif masking_scheme.lower() == "noise":
                        input_embed[pos] = torch.randn(
                            input_embed[pos].size(),
                            dtype=input_embed.dtype,
                            device=input_embed.device,
                        )
                    else:
                        input_embed[pos] = 0.0

            if position_ids is None:
                rope_state = getattr(self.model.model, "rope_deltas", None)
                if rope_state is None or cache_position is None or cache_position[0] == 0:
                    position_ids, rope_deltas = self.model.model.get_rope_index(
                        input_ids, image_grid_thw, None, attention_mask
                    )
                    self.model.model.rope_deltas = rope_deltas
                else:
                    batch_size, seq_length, _ = inputs_embeds.shape
                    position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                    position_ids = position_ids.view(1, 1, -1).expand(3, batch_size, -1)
                    delta = (cache_position[0] + self.model.model.rope_deltas).to(inputs_embeds.device)
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                    position_ids = position_ids + delta.to(position_ids.device)

            with torch.no_grad():
                outputs = self.model.model.language_model(
                    input_ids=None,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    use_cache=True,
                    output_attentions=True,
                    output_hidden_states=False,
                    return_dict=True,
                    cache_position=cache_position,
                )

            hidden_states = outputs[0]
            logits = self.model.lm_head(hidden_states)
        loss = None

        if not return_dict:
            output = (logits,) + outputs[1:]
        else:
            output = CausalLMOutputWithPast(
                loss=loss,
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )

        action_logits = self.classifier(hidden_states)
        return action_logits, output

    def generate(
            self,
            inputs: Optional[torch.Tensor] = None,
            **kwargs: object,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        from src.decoding.octopus_utils import avisc_sample

        orig_sample = getattr(self.model, "sample", None)
        orig_prepare_method = getattr(self.model, "prepare_inputs_for_generation_method", None)
        orig_prepare_cd = getattr(self.model, "prepare_inputs_for_generation_cd", None)
        orig_prepare_m3id = getattr(self.model, "prepare_inputs_for_generation_m3id", None)
        image_kwarg = ModelHelper.get_image_kwarg_name(self.model_type)

        def _prepare_with_image(model_self, input_ids, past_key_values=None,
                                attention_mask=None, inputs_embeds=None, image_key=None, **model_kwargs):
            if past_key_values:
                input_ids = input_ids[:, -1:]
            if inputs_embeds is not None and past_key_values is None:
                model_inputs = {"inputs_embeds": inputs_embeds}
            else:
                model_inputs = {"input_ids": input_ids}

            if image_key is not None and model_kwargs.get(image_key) is not None:
                model_inputs[image_kwarg] = model_kwargs.get(image_key)

            for key in ("cache_position", "position_ids", "rope_deltas", "image_grid_thw"):
                if key in model_kwargs and model_kwargs[key] is not None:
                    model_inputs[key] = model_kwargs[key]

            model_inputs.update(
                {
                    "past_key_values": past_key_values,
                    "use_cache": model_kwargs.get("use_cache"),
                    "attention_mask": attention_mask,
                }
            )
            return model_self.prepare_inputs_for_generation(**model_inputs)

        def _prepare_inputs_for_generation_method(model_self, input_ids, past_key_values=None,
                                                  attention_mask=None, inputs_embeds=None, **model_kwargs):
            return _prepare_with_image(
                model_self,
                input_ids,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                image_key="images",
                **model_kwargs,
            )

        def _prepare_inputs_for_generation_cd(model_self, input_ids, past_key_values=None,
                                              attention_mask=None, inputs_embeds=None, **model_kwargs):
            return _prepare_with_image(
                model_self,
                input_ids,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                image_key="images_cd",
                **model_kwargs,
            )

        def _prepare_inputs_for_generation_m3id(model_self, input_ids, past_key_values=None,
                                                attention_mask=None, inputs_embeds=None, **model_kwargs):
            input_ids_m3id = input_ids
            attention_mask_m3id = attention_mask
            if past_key_values:
                input_ids_m3id = input_ids[:, -1:]
            elif self.model_type in ("qwen2_vl", "qwen2_5_vl"):
                config = getattr(model_self, "config", None)
                vision_start_id = getattr(config, "vision_start_token_id", 151652)
                vision_end_id = getattr(config, "vision_end_token_id", 151653)
                image_token_id = getattr(config, "image_token_id", None)
                mask = (input_ids != vision_start_id) & (input_ids != vision_end_id)
                if image_token_id is not None:
                    mask = mask & (input_ids != image_token_id)
                input_ids_m3id = input_ids[mask].view(input_ids.shape[0], -1)
                if attention_mask is not None:
                    attention_mask_m3id = attention_mask[mask].view(attention_mask.shape[0], -1)
            else:
                if input_ids is not None and (input_ids == -200).any():
                    input_ids_m3id = input_ids[input_ids != -200].view(input_ids.shape[0], -1)
                if attention_mask is not None and attention_mask_m3id is not None:
                    attention_mask_m3id = attention_mask_m3id[:, :input_ids_m3id.shape[1]]

            return _prepare_with_image(
                model_self,
                input_ids_m3id,
                past_key_values=past_key_values,
                attention_mask=attention_mask_m3id,
                inputs_embeds=inputs_embeds,
                image_key=None,
                **model_kwargs,
            )

        try:
            self.model.sample = types.MethodType(avisc_sample.sample, self.model)
            if orig_prepare_method is None:
                self.model.prepare_inputs_for_generation_method = types.MethodType(
                    _prepare_inputs_for_generation_method, self.model
                )
            if orig_prepare_cd is None:
                self.model.prepare_inputs_for_generation_cd = types.MethodType(
                    _prepare_inputs_for_generation_cd, self.model
                )
            if orig_prepare_m3id is None:
                self.model.prepare_inputs_for_generation_m3id = types.MethodType(
                    _prepare_inputs_for_generation_m3id, self.model
                )
            return avisc_sample.generate(self.model, inputs=inputs, mymodel=self, **kwargs)
        finally:
            if orig_sample is not None:
                self.model.sample = orig_sample
            if orig_prepare_method is not None:
                self.model.prepare_inputs_for_generation_method = orig_prepare_method
            elif hasattr(self.model, "prepare_inputs_for_generation_method"):
                delattr(self.model, "prepare_inputs_for_generation_method")
            if orig_prepare_cd is not None:
                self.model.prepare_inputs_for_generation_cd = orig_prepare_cd
            elif hasattr(self.model, "prepare_inputs_for_generation_cd"):
                delattr(self.model, "prepare_inputs_for_generation_cd")
            if orig_prepare_m3id is not None:
                self.model.prepare_inputs_for_generation_m3id = orig_prepare_m3id
            elif hasattr(self.model, "prepare_inputs_for_generation_m3id"):
                delattr(self.model, "prepare_inputs_for_generation_m3id")


class OctopusMitigator(TrainableMitigator):
    """
    Octopus: Dynamic Strategy Selection.

    Uses reference generation loop and MyModel-style policy wrapper.
    """

    name: str = "octopus"

    def __init__(
            self,
            model: nn.Module,
            model_type: str = "llava",
            classifier: Optional[OctopusClassifier] = None,
            noise_step: int = 500,
            layer_gamma: float = 0.5,
            lamb: float = 100.0,
            masking_scheme: str = "zeros",
            n_query: int = 4,
            bt: int = 1,
            **kwargs,
    ):
        super().__init__(model, model_type, **kwargs)
        self.noise_step = noise_step
        self.layer_gamma = layer_gamma
        self.lamb = lamb
        self.masking_scheme = masking_scheme

        if classifier is None:
            d_model = 4096
            if hasattr(model, "config"):
                d_model = getattr(model.config, "hidden_size", d_model)
            classifier = OctopusClassifier(d_model=d_model, n_query=n_query, bt=bt)

        self.classifier = classifier
        self.policy = OctopusPolicy(model, classifier, model_type=model_type)

    def setup(self) -> None:
        device = next(self.model.parameters()).device
        self.policy = self.policy.to(device)

    def cleanup(self) -> None:
        return

    def get_trainable_parameters(self) -> List[nn.Parameter]:
        return list(self.classifier.parameters())

    def compute_loss(
            self,
            hidden_states: torch.Tensor,
            chosen_actions: torch.Tensor,
            rejected_actions: torch.Tensor,
            beta: float = 1.0,
    ) -> torch.Tensor:
        # If hidden_states are already action scores [B, T, C], use them directly
        if hidden_states.dim() == 3 and hidden_states.shape[-1] == 4:
            action_scores = hidden_states
        else:
            action_logits = self.classifier(hidden_states)
            action_scores = action_logits.unsqueeze(1)

        if chosen_actions.dim() == 1:
            chosen_actions = chosen_actions.unsqueeze(1)
        if rejected_actions.dim() == 1:
            rejected_actions = rejected_actions.unsqueeze(1)

        len_action = action_scores.shape[1]
        len_text = min(chosen_actions.shape[1], rejected_actions.shape[1])
        if len_action < len_text:
            chosen_actions = chosen_actions[:, :len_action]
            rejected_actions = rejected_actions[:, :len_action]
        else:
            action_scores = action_scores[:, :len_text, :]
            chosen_actions = chosen_actions[:, :len_text]
            rejected_actions = rejected_actions[:, :len_text]

        chosen_logps = torch.gather(
            action_scores.log_softmax(-1), 2, chosen_actions[:, :, None]
        ).squeeze(2).sum(-1)
        rejected_logps = torch.gather(
            action_scores.log_softmax(-1), 2, rejected_actions[:, :, None]
        ).squeeze(2).sum(-1)

        pi_logratios = chosen_logps - rejected_logps
        logits = (pi_logratios - 0).sum(-1)
        loss = -F.logsigmoid(beta * logits)

        return loss

    def generate(
            self,
            input_ids: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            pixel_values: Optional[torch.Tensor] = None,
            images: Optional[torch.Tensor] = None,
            images_cd: Optional[torch.Tensor] = None,
            is_eval: bool = False,
            output_scores: Optional[bool] = None,
            return_dict_in_generate: Optional[bool] = None,
            **kwargs,
    ):
        if images is None and pixel_values is not None:
            images = pixel_values
        if images is None:
            raise ValueError("Octopus requires images or pixel_values")

        if images_cd is None:
            images_cd = add_diffusion_noise(images, noise_step=self.noise_step)

        gen_kwargs = {
            "do_sample": self.config.do_sample,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "top_k": self.config.top_k,
            "max_new_tokens": self.config.max_new_tokens,
            "use_cache": True,
            "use_avisc": True,
            "use_m3id": True,
            "layer_gamma": self.layer_gamma,
            "masking_scheme": self.masking_scheme,
            "lamb": self.lamb,
            "is_eval": is_eval,
        }
        gen_kwargs.update(kwargs)

        if output_scores is not None:
            gen_kwargs["output_scores"] = output_scores
        if return_dict_in_generate is not None:
            gen_kwargs["return_dict_in_generate"] = return_dict_in_generate
        if attention_mask is not None:
            gen_kwargs["attention_mask"] = attention_mask

        if "model_name" not in gen_kwargs:
            if self.model_type in ("llava", "llava_next"):
                gen_kwargs["model_name"] = "llava"
            else:
                gen_kwargs["model_name"] = self.model_type
        gen_kwargs.setdefault("model_type", self.model_type)
        gen_kwargs.setdefault("model_config", getattr(self.model, "config", None))

        return self.policy.generate(
            inputs=input_ids,
            images=images,
            images_cd=images_cd,
            **gen_kwargs,
        )

    def save_pretrained(self, path: str) -> None:
        payload = {
            "query": self.classifier.queries.data,
            "cls": self.classifier.cls_token.data,
            "mlp_state_dict": self.classifier.mlp.state_dict(),
            "transformer": self.classifier.transformer.state_dict(),
        }
        torch.save(payload, path)

    @classmethod
    def from_pretrained(
            cls,
            model: nn.Module,
            path: str,
            model_type: str = "llava",
            **kwargs,
    ) -> "OctopusMitigator":
        checkpoint = torch.load(path, map_location="cpu")

        d_model = checkpoint["cls"].shape[-1]
        n_query = checkpoint["query"].shape[0]

        classifier = OctopusClassifier(d_model=d_model, n_query=n_query)
        classifier.cls_token.data = checkpoint["cls"]
        classifier.mlp.load_state_dict(checkpoint["mlp_state_dict"])
        classifier.transformer.load_state_dict(checkpoint["transformer"])
        classifier.queries.data = checkpoint["query"]

        return cls(model, model_type=model_type, classifier=classifier, **kwargs)
