"""
Analyze attention weights at each decoding step (Figure 3a style).
Computes average attention to image_info_tokens vs prompt_tokens during generation.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import List, Tuple, Dict


def compute_decoding_attention_weights(
    attentions: torch.Tensor,
    info_token_range: Tuple[int, int],
    original_input_length: int,
    last_layer_idx: int = 31
) -> Tuple[List[float], List[float]]:
    """
    Compute total attention weights to info tokens and prompt tokens at each decoding step.

    Args:
        attentions: Attention tensor (batch, layers, heads, seq_len, seq_len)
        info_token_range: (start, end) of info tokens (vision or caption)
        original_input_length: Length of original input (before generation)
        last_layer_idx: Which layer to analyze (default 31)

    Returns:
        Tuple of (info_weights, prompt_weights) lists for each generation step.
        Each value is the sum (not average) of attention to the respective token range.
    """
    info_start, info_end = info_token_range

    # Extract last layer attention: (heads, seq_len, seq_len)
    last_layer_attn = attentions[0, last_layer_idx, :, :, :]

    # Average over heads: (seq_len, seq_len)
    avg_attn = last_layer_attn.mean(dim=0).cpu().numpy()

    info_weights = []
    prompt_weights = []

    # Iterate over generated tokens (starting from original_input_length)
    total_length = avg_attn.shape[0]
    for step_idx in range(original_input_length, total_length):
        # Attention from current generated token to all previous tokens
        attn_from_step = avg_attn[step_idx, :]  # (seq_len,)

        # Total attention to info tokens (fixed range)
        info_attn = attn_from_step[info_start:info_end].sum()
        info_weights.append(float(info_attn))

        # Total attention to prompt tokens (from info_end to current step)
        # This changes at each step - includes all tokens from info_end to step_idx
        prompt_attn = attn_from_step[info_end:step_idx].sum()
        prompt_weights.append(float(prompt_attn))

    return info_weights, prompt_weights


def analyze_sample_decoding(
    result: Dict,
    mode: str = 'image'
) -> Dict[str, List[float]]:
    """
    Analyze decoding attention for a single sample.

    Args:
        result: Result dictionary from inference
        mode: 'image' or 'text'

    Returns:
        Dictionary with info_weights and prompt_weights lists
    """
    mode_key = f'{mode}_mode'

    if mode_key not in result or result[mode_key]['attentions'] is None:
        return None

    data = result[mode_key]
    attentions = data['attentions']
    original_input_length = data['input_ids'].shape[1]

    if mode == 'image':
        info_token_range = data.get('vision_token_range')
        if info_token_range is None:
            return None
    else:  # text mode
        info_token_range = data.get('prompt_token_range')  # This is caption range
        if info_token_range is None:
            return None

    info_weights, prompt_weights = compute_decoding_attention_weights(
        attentions,
        info_token_range,
        original_input_length
    )

    return {
        'info_weights': info_weights,
        'prompt_weights': prompt_weights
    }


def plot_attention_weights(
    image_data: Dict[str, List[List[float]]],
    text_data: Dict[str, List[List[float]]],
    output_dir: str
):
    """
    Generate 3 plots: image only, text only, and combined.

    Args:
        image_data: Dict with 'info_weights' and 'prompt_weights' (each sample is a list)
        text_data: Dict with 'info_weights' and 'prompt_weights' (each sample is a list)
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)

    # Average across samples (pad to max length)
    def average_weights(weights_list: List[List[float]]) -> np.ndarray:
        if not weights_list:
            return np.array([])
        max_len = max(len(w) for w in weights_list)
        # Pad with NaN and compute nanmean
        padded = np.full((len(weights_list), max_len), np.nan)
        for i, w in enumerate(weights_list):
            padded[i, :len(w)] = w
        return np.nanmean(padded, axis=0)

    # 1. Image only plot
    if image_data['info_weights']:
        avg_image_info = average_weights(image_data['info_weights'])
        avg_image_prompt = average_weights(image_data['prompt_weights'])

        fig, ax = plt.subplots(figsize=(10, 6))
        steps = np.arange(len(avg_image_info))
        ax.plot(steps, avg_image_info, 'b-', linewidth=2, label='Image Token')
        ax.plot(steps, avg_image_prompt, 'r-', linewidth=2, label='Text Token')
        ax.set_xlabel('Generated Tokens', fontsize=12)
        ax.set_ylabel('Attention Weight', fontsize=12)
        ax.set_title('Image Mode: Attention Weights at Each Decoding Step', fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'image_attention_weights_decoding_steps.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: image_attention_weights_decoding_steps.png")

    # 2. Text only plot
    if text_data['info_weights']:
        avg_text_info = average_weights(text_data['info_weights'])
        avg_text_prompt = average_weights(text_data['prompt_weights'])

        fig, ax = plt.subplots(figsize=(10, 6))
        steps = np.arange(len(avg_text_info))
        ax.plot(steps, avg_text_info, 'b-', linewidth=2, label='Caption Token')
        ax.plot(steps, avg_text_prompt, 'r-', linewidth=2, label='Text Token')
        ax.set_xlabel('Generated Tokens', fontsize=12)
        ax.set_ylabel('Attention Weight', fontsize=12)
        ax.set_title('Text Mode: Attention Weights at Each Decoding Step', fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'text_attention_weights_decoding_steps.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: text_attention_weights_decoding_steps.png")

    # 3. Combined plot
    if image_data['info_weights'] and text_data['info_weights']:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

        # Left: Image mode
        steps_img = np.arange(len(avg_image_info))
        ax1.plot(steps_img, avg_image_info, 'b-', linewidth=2, label='Image Token')
        ax1.plot(steps_img, avg_image_prompt, 'r-', linewidth=2, label='Text Token')
        ax1.set_xlabel('Generated Tokens', fontsize=12)
        ax1.set_ylabel('Attention Weight', fontsize=12)
        ax1.set_title('(a) Image Mode', fontsize=14)
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)

        # Right: Text mode
        steps_txt = np.arange(len(avg_text_info))
        ax2.plot(steps_txt, avg_text_info, 'b-', linewidth=2, label='Caption Token')
        ax2.plot(steps_txt, avg_text_prompt, 'r-', linewidth=2, label='Text Token')
        ax2.set_xlabel('Generated Tokens', fontsize=12)
        ax2.set_ylabel('Attention Weight', fontsize=12)
        ax2.set_title('(b) Text Mode', fontsize=14)
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'combined_attention_weights_decoding_steps.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: combined_attention_weights_decoding_steps.png")


def main():
    # Configuration
    results_path = "inference_results/all_results.pt"
    output_dir = "analysis/averaged"

    os.makedirs(output_dir, exist_ok=True)

    # Load results
    print(f"Loading results from {results_path}...")
    all_results = torch.load(results_path)
    print(f"Loaded {len(all_results)} results")

    # Collect data from all samples
    image_info_weights_all = []
    image_prompt_weights_all = []
    text_info_weights_all = []
    text_prompt_weights_all = []

    print("\nAnalyzing decoding attention patterns...")
    for idx, result in enumerate(tqdm(all_results, desc="Processing samples")):
        # Analyze image mode
        image_result = analyze_sample_decoding(result, mode='image')
        if image_result:
            image_info_weights_all.append(image_result['info_weights'])
            image_prompt_weights_all.append(image_result['prompt_weights'])

        # Analyze text mode
        text_result = analyze_sample_decoding(result, mode='text')
        if text_result:
            text_info_weights_all.append(text_result['info_weights'])
            text_prompt_weights_all.append(text_result['prompt_weights'])

    print(f"\nCollected data from:")
    print(f"  - Image mode: {len(image_info_weights_all)} samples")
    print(f"  - Text mode: {len(text_info_weights_all)} samples")

    # Prepare data for plotting
    image_data = {
        'info_weights': image_info_weights_all,
        'prompt_weights': image_prompt_weights_all
    }

    text_data = {
        'info_weights': text_info_weights_all,
        'prompt_weights': text_prompt_weights_all
    }

    # Generate plots
    print("\nGenerating plots...")
    plot_attention_weights(image_data, text_data, output_dir)

    print(f"\nAnalysis complete! Results saved to {output_dir}/")


if __name__ == "__main__":
    main()