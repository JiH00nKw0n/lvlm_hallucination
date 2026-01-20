"""
VTI direction computation utilities.

Reference: VTI/vti_utils/utils.py

Direction formula: correct - hallucinating (각 layer의 last token)
"""

import json
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .pca import PCA

HiddenStatesTuple = Tuple[torch.Tensor, ...]
InputDict = Dict[str, torch.Tensor]


@dataclass
class VTIArgs:
    """Arguments for VTI direction computation."""
    num_demos: int = 5
    num_trials: int = 1
    mask_ratio: float = 0.5
    data_file: str = ""  # COCO base path
    conv_mode: str = "llava_v1"


def process_image(image_processor, image_raw: Image.Image) -> torch.Tensor:
    """
    Process image using model's image processor.

    Reference: VTI/vti_utils/utils.py:26-44
    """
    answer = image_processor(image_raw)

    if 'pixel_values' in answer:
        answer = answer['pixel_values'][0]

    if isinstance(answer, np.ndarray):
        answer = torch.from_numpy(answer)
    elif isinstance(answer, torch.Tensor):
        return answer
    else:
        raise ValueError("Unexpected output format from image_processor.")

    return answer


def mask_patches(
    tensor: torch.Tensor,
    indices: torch.Tensor,
    patch_size: int = 14,
) -> torch.Tensor:
    """
    Creates a new tensor where specified patches are set to the mean.

    Reference: VTI/vti_utils/utils.py:46-81

    Args:
        tensor: Input tensor (C, H, W)
        indices: Indices of patches to modify
        patch_size: Size of square patch
    """
    new_tensor = tensor.clone()
    mean_values = tensor.mean(dim=(1, 2), keepdim=True)
    patches_per_row = tensor.shape[2] // patch_size

    for index in indices:
        row = index // patches_per_row
        col = index % patches_per_row
        start_x = col * patch_size
        start_y = row * patch_size
        new_tensor[:, start_y:start_y + patch_size, start_x:start_x + patch_size] = \
            mean_values.expand(-1, patch_size, patch_size)

    return new_tensor


def get_prompts(
    args: VTIArgs,
    model,
    tokenizer,
    data_demos: List[dict],
    question: str,
    model_is_llava: bool = True,
) -> Tuple:
    """
    Create positive/negative input pairs.

    Reference: VTI/vti_utils/utils.py:84-148

    Returns:
        tuple of (negative_input, positive_input) pairs
    """
    if model_is_llava:
        from llava.constants import (
            IMAGE_TOKEN_INDEX,
            DEFAULT_IMAGE_TOKEN,
            DEFAULT_IM_START_TOKEN,
            DEFAULT_IM_END_TOKEN,
        )
        from llava.conversation import conv_templates
        from llava.mm_utils import tokenizer_image_token

        qs_pos = question
        qs_neg = question

        if hasattr(model.config, 'mm_use_im_start_end'):
            if model.config.mm_use_im_start_end:
                qs_pos = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs_pos
                qs_neg = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs_neg
            else:
                qs_pos = DEFAULT_IMAGE_TOKEN + '\n' + qs_pos
                qs_neg = DEFAULT_IMAGE_TOKEN + '\n' + qs_neg

            conv_pos = conv_templates[args.conv_mode].copy()
            conv_pos.append_message(conv_pos.roles[0], qs_pos)
            conv_pos.append_message(conv_pos.roles[1], None)

            conv_neg = conv_templates[args.conv_mode].copy()
            conv_neg.append_message(conv_neg.roles[0], qs_neg)
            conv_neg.append_message(conv_neg.roles[1], None)

            prompts_positive = [conv_pos.get_prompt() + k['value'] for k in data_demos]
            prompts_negative = [conv_neg.get_prompt() + k['h_value'] for k in data_demos]

            input_ids_positive = [
                tokenizer_image_token(p, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
                for p in prompts_positive
            ]
            input_ids_negative = [
                tokenizer_image_token(p, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
                for p in prompts_negative
            ]
        else:
            raise ValueError("InstructBLIP not supported in this implementation")

        inputs = [
            (input_ids_negative[demo_id], input_ids_positive[demo_id])
            for demo_id in range(len(input_ids_negative))
        ]
        inputs = tuple(inputs)

    else:
        # Qwen-VL style
        prompts_positive = []
        prompts_negative = []

        for k in data_demos:
            image_path = os.path.join(args.data_file, 'train2014', k['image'])
            prompts_positive.append(
                tokenizer.from_list_format([{'image': image_path}, {'text': question + k['value']}])
            )
            prompts_negative.append(
                tokenizer.from_list_format([{'image': image_path}, {'text': question + k['h_value']}])
            )

        input_ids_positive = [tokenizer(p, return_tensors='pt').to(model.device) for p in prompts_positive]
        input_ids_negative = [tokenizer(p, return_tensors='pt').to(model.device) for p in prompts_negative]
        inputs = [
            (input_ids_negative[demo_id], input_ids_positive[demo_id])
            for demo_id in range(len(input_ids_negative))
        ]
        inputs = tuple(inputs)

    return inputs


def get_demos(
    args: VTIArgs,
    image_processor: Any,
    model: Any,
    tokenizer: Any,
    patch_size: int = 14,
    file_path: Optional[str] = None,
    model_is_llava: bool = True,
) -> Tuple[List[List[torch.Tensor]], Tuple[Tuple[InputDict, InputDict], ...]]:
    """
    Load demonstration data from JSONL.

    Reference: VTI/vti_utils/utils.py:150-181

    Returns:
        (inputs_images, input_ids)
    """
    if file_path is None:
        file_path = os.path.join(os.path.dirname(__file__), 'hallucination_vti_demos.jsonl')

    data = []
    with open(file_path, 'r') as file:
        for line in file:
            json_object = json.loads(line.strip())
            data.append(json_object)

    data_demos = random.sample(data, min(args.num_demos, len(data)))

    inputs_images = []
    for i in range(len(data_demos)):
        question = data_demos[i]['question']
        image_path = os.path.join(args.data_file, 'train2014', data_demos[i]['image'])
        image_raw = Image.open(image_path).convert("RGB")
        image_tensor = process_image(image_processor, image_raw)
        image_tensor_cd_all_trials = []

        for t in range(args.num_trials):
            token_numbers = image_tensor.shape[-1] * image_tensor.shape[-2] / patch_size ** 2
            mask_index = torch.randperm(int(token_numbers))[:int(args.mask_ratio * token_numbers)]
            image_tensor_cd = mask_patches(image_tensor, mask_index, patch_size=patch_size)
            image_tensor_cd_all_trials.append(image_tensor_cd)

        inputs_images.append([image_tensor_cd_all_trials, image_tensor])

    input_ids = get_prompts(args, model, tokenizer, data_demos, question, model_is_llava=model_is_llava)

    return inputs_images, input_ids


def get_hiddenstates(
    model: Any,
    inputs: Tuple[Tuple[InputDict, InputDict], ...],
    image_tensor: Optional[Sequence[Sequence[torch.Tensor]]],
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Extract hidden states for textual VTI.

    Reference: VTI/vti_utils/utils.py:184-210
    """
    h_all = []
    with torch.no_grad():
        for example_id in range(len(inputs)):
            embeddings_for_all_styles = []
            for style_id in range(len(inputs[example_id])):
                if image_tensor is None:
                    h = model(
                        **inputs[example_id][style_id],
                        output_hidden_states=True,
                        return_dict=True
                    ).hidden_states
                else:
                    h = model(
                        inputs[example_id][style_id],
                        images=image_tensor[example_id][-1].unsqueeze(0).half(),
                        use_cache=False,
                        output_hidden_states=True,
                        return_dict=True
                    ).hidden_states

                embedding_token = []
                for layer in range(len(h)):
                    embedding_token.append(h[layer][:, -1].detach().cpu())

                embedding_token = torch.cat(embedding_token, dim=0).cpu().clone()
                embeddings_for_all_styles.append(embedding_token)
            h_all.append(tuple(embeddings_for_all_styles))
    return h_all


def obtain_textual_vti(model, inputs, image_tensor, rank: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute textual VTI direction.

    Reference: VTI/vti_utils/utils.py:212-230

    Direction: correct - hallucinating (각 layer의 last token)

    Returns:
        (direction, reading_direction)
    """
    hidden_states = get_hiddenstates(model, inputs, image_tensor)
    hidden_states_all = []
    num_demonstration = len(hidden_states)
    neg_all = []
    pos_all = []

    for demonstration_id in range(num_demonstration):
        # h = positive - negative = correct - hallucinating
        h = hidden_states[demonstration_id][1].view(-1) - hidden_states[demonstration_id][0].view(-1)
        hidden_states_all.append(h)
        neg_all.append(hidden_states[demonstration_id][0].view(-1))
        pos_all.append(hidden_states[demonstration_id][1].view(-1))

    fit_data = torch.stack(hidden_states_all)
    pca = PCA(n_components=rank).to(fit_data.device).fit(fit_data.float())
    eval_data = pca.transform(fit_data.float())
    h_pca = pca.inverse_transform(eval_data)

    direction = (pca.components_.sum(dim=1, keepdim=True) + pca.mean_).mean(0).view(
        hidden_states[demonstration_id][0].size(0),
        hidden_states[demonstration_id][0].size(1)
    )
    reading_direction = fit_data.mean(0).view(
        hidden_states[demonstration_id][0].size(0),
        hidden_states[demonstration_id][0].size(1)
    )

    return direction, reading_direction


def _average_tuples(tuples: List[Tuple[torch.Tensor, ...]]) -> Tuple[torch.Tensor, ...]:
    """Average tensors at each position across tuples."""
    if not tuples:
        raise ValueError("The input list of tuples is empty.")

    n = len(tuples[0])
    if not all(len(t) == n for t in tuples):
        raise ValueError("All tuples must have the same length.")

    averaged_tensors = []
    for i in range(n):
        tensors_at_i = torch.stack([t[i].detach().cpu() for t in tuples])
        averaged_tensor = tensors_at_i.mean(dim=0)
        averaged_tensors.append(averaged_tensor)

    return tuple(averaged_tensors)


def get_visual_hiddenstates(
    model: Any,
    image_tensor: Sequence[Sequence[torch.Tensor]],
    model_is_llava: bool = True,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Extract hidden states from vision encoder.

    Reference: VTI/vti_utils/utils.py:257-307
    """
    h_all = []
    with torch.no_grad():
        if model_is_llava:
            try:
                vision_model = model.model.vision_tower.vision_tower.vision_model
            except:
                vision_model = model.vision_model
        else:
            vision_model = model.transformer.visual
            model.transformer.visual.output_hidden_states = True

        for example_id in range(len(image_tensor)):
            embeddings_for_all_styles = []
            for style_id in range(len(image_tensor[example_id])):
                if isinstance(image_tensor[example_id][style_id], list):
                    h = []
                    for image_tensor_ in image_tensor[example_id][style_id]:
                        if model_is_llava:
                            h_ = vision_model(
                                image_tensor_.unsqueeze(0).half().cuda(),
                                output_hidden_states=True,
                                return_dict=True
                            ).hidden_states
                        else:
                            _, h_ = vision_model(image_tensor_.unsqueeze(0).cuda())
                        h.append(h_)
                    h = _average_tuples(h)
                else:
                    if model_is_llava:
                        h = vision_model(
                            image_tensor[example_id][style_id].unsqueeze(0).cuda(),
                            output_hidden_states=True,
                            return_dict=True
                        ).hidden_states
                    else:
                        _, h = vision_model(image_tensor[example_id][style_id].unsqueeze(0).cuda())

                embedding_token = []
                for layer in range(len(h)):
                    embedding_token.append(h[layer][:, :].detach().cpu())
                embedding_token = torch.cat(embedding_token, dim=0)
                embeddings_for_all_styles.append(embedding_token)
            h_all.append(tuple(embeddings_for_all_styles))

        if not model_is_llava:
            model.transformer.visual.output_hidden_states = False

    return h_all


def obtain_visual_vti(
    model: Any,
    image_tensor: Sequence[Sequence[torch.Tensor]],
    rank: int = 1,
    model_is_llava: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute visual VTI direction.

    Reference: VTI/vti_utils/utils.py:309-325

    Direction: masked - clean (vision encoder에서)

    Returns:
        (direction, reading_direction)
    """
    hidden_states = get_visual_hiddenstates(model, image_tensor, model_is_llava=model_is_llava)
    n_layers, n_tokens, feat_dim = hidden_states[0][0].shape
    num_demonstration = len(hidden_states)

    hidden_states_all = []
    for demonstration_id in range(num_demonstration):
        # h = masked - clean
        h = hidden_states[demonstration_id][0].reshape(n_tokens, -1) - \
            hidden_states[demonstration_id][1].reshape(n_tokens, -1)
        hidden_states_all.append(h)

    fit_data = torch.stack(hidden_states_all, dim=1)[:]  # n_token x n_demos x D
    pca = PCA(n_components=rank).to(fit_data.device).fit(fit_data.float())
    direction = (pca.components_.sum(dim=1, keepdim=True) + pca.mean_).mean(1).view(n_layers, n_tokens, -1)
    reading_direction = fit_data.mean(1).view(n_layers, n_tokens, -1)

    return direction, reading_direction
