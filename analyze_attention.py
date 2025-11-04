"""
Analyze attention patterns and logit contributions from inference results.
Based on: https://github.com/ZhangqiJiang07/middle_layers_indicating_hallucinations

Performs:
1. Attention map plot - Heatmap visualization of attention weights
2. Attention sublayers contribution plot - Layer-wise contribution to final logits
3. Retrieve text token via logit lens - Predict tokens from intermediate hidden states
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import spacy
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel


def load_spacy_model():
    """Load spacy model for noun extraction."""
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("Downloading spacy model...")
        os.system("python -m spacy download en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")
    return nlp


def extract_nouns(text: str, nlp) -> List[str]:
    """Extract nouns from text using spacy."""
    doc = nlp(text)
    nouns = [token.text for token in doc if token.pos_ in ["NOUN", "PROPN"]]
    return nouns


def find_token_positions(tokens: List[str], target_word: str) -> List[int]:
    """Find positions of target word in token list."""
    positions = []
    target_lower = target_word.lower()

    for idx, token in enumerate(tokens):
        token_clean = token.replace('▁', '').lower()
        if target_lower in token_clean or token_clean in target_lower:
            positions.append(idx)

    return positions


def plot_attention_distribution_sorted_heads(
    attentions: torch.Tensor,
    object_token_idx: int,
    vision_token_range: Tuple[int, int],
    output_path: str = None,
    title: str = "Attention Distribution"
):
    """
    Plot attention distribution across layers and heads (sorted by attention ratio).
    Recreates Figure 2(a) from the paper.

    Args:
        attentions: Attention tensor (batch, layers, heads, seq_len, seq_len)
        object_token_idx: Index of the object token to analyze
        vision_token_range: (start, end) indices of vision tokens
        output_path: Path to save plot
        title: Plot title
    """
    vision_start, vision_end = vision_token_range

    # Extract attention from object token to vision tokens
    # Shape: (layers, heads, vision_tokens)
    attn_to_vision = attentions[
        0,  # batch
        :,  # all layers
        :,  # all heads
        object_token_idx,  # from object token
        vision_start:vision_end  # to vision tokens
    ]

    # Sum over vision tokens to get attention ratio per (layer, head)
    # Shape: (layers, heads)
    attn_ratio = attn_to_vision.sum(dim=-1).cpu().numpy()

    num_layers, num_heads = attn_ratio.shape

    # Sort heads by attention ratio within each layer
    sorted_attn = np.zeros_like(attn_ratio)
    for layer_idx in range(num_layers):
        sorted_indices = np.argsort(attn_ratio[layer_idx])
        sorted_attn[layer_idx] = attn_ratio[layer_idx, sorted_indices]

    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(sorted_attn, aspect='auto', cmap='Blues', origin='lower', vmin=0, vmax=0.75)

    ax.set_xlabel('Sorted Heads', fontsize=12)
    ax.set_ylabel('Layers', fontsize=12)
    ax.set_title(title, fontsize=14)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Visual Attention Ratio', fontsize=11)

    # Mark middle layers (e.g., 5-18 for LLaVA-1.5-7B)
    middle_start = 5
    middle_end = 18
    ax.axhline(y=middle_start - 0.5, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax.axhline(y=middle_end + 0.5, color='red', linestyle='--', linewidth=1, alpha=0.5)

    # Add text annotation for middle layers
    ax.text(
        num_heads * 0.7, (middle_start + middle_end) / 2,
        'high attn. in\nmiddle layers',
        color='red',
        fontsize=10,
        ha='left',
        va='center',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
    )

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_attention_map_compressed(
    attentions: torch.Tensor,
    tokens: List[str],
    layer_idx: int,
    head_idx: Optional[int] = None,
    vision_token_range: Optional[Tuple[int, int]] = None,
    prompt_token_range: Optional[Tuple[int, int]] = None,
    input_length: int = 0,
    output_path: str = None,
    title: str = "Attention Map (Compressed)"
):
    """
    Plot attention map with image/caption tokens compressed into single tokens.

    Args:
        attentions: Attention tensor (batch, layers, heads, seq_len, seq_len)
        tokens: List of token strings
        layer_idx: Which layer to visualize
        head_idx: Which head to visualize (if None, average over all heads)
        vision_token_range: Optional (start, end) of vision tokens
        prompt_token_range: Optional (start, end) of caption tokens
        input_length: Length of input (non-generated) tokens
        output_path: Path to save plot
        title: Plot title
    """
    # Extract attention for specified layer
    if head_idx is not None:
        attn_weights = attentions[0, layer_idx, head_idx, :, :].cpu().numpy()
        title = f"{title} - Layer {layer_idx}, Head {head_idx}"
    else:
        attn_weights = attentions[0, layer_idx, :, :, :].mean(dim=0).cpu().numpy()
        title = f"{title} - Layer {layer_idx} (avg heads)"

    # Compress attention matrix
    compressed_attn = []
    compressed_tokens = []

    # For each query position (row)
    for q_idx in range(len(tokens)):
        row = []

        # Add image token attention (sum)
        if vision_token_range is not None:
            v_start, v_end = vision_token_range
            row.append(attn_weights[q_idx, v_start:v_end].sum())

        # Add caption token attention (sum)
        if prompt_token_range is not None:
            p_start, p_end = prompt_token_range
            row.append(attn_weights[q_idx, p_start:p_end].sum())

        # Add generated token attentions (individual)
        if input_length < len(tokens):
            row.extend(attn_weights[q_idx, input_length:])

        compressed_attn.append(row)

    compressed_attn = np.array(compressed_attn)

    # Compress query dimension (rows) similarly
    final_attn = []

    # Image tokens row (sum)
    if vision_token_range is not None:
        v_start, v_end = vision_token_range
        final_attn.append(compressed_attn[v_start:v_end].sum(axis=0))
        compressed_tokens.append("<IMAGE>")

    # Caption tokens row (sum)
    if prompt_token_range is not None:
        p_start, p_end = prompt_token_range
        final_attn.append(compressed_attn[p_start:p_end].sum(axis=0))
        compressed_tokens.append("<CAPTION>")

    # Generated tokens rows (individual)
    if input_length < len(tokens):
        for i in range(input_length, len(tokens)):
            final_attn.append(compressed_attn[i])
            token = tokens[i][:15] if len(tokens[i]) > 15 else tokens[i]
            compressed_tokens.append(token)

    final_attn = np.array(final_attn)

    # Create key tokens labels (columns)
    key_tokens = []
    if vision_token_range is not None:
        key_tokens.append("<IMAGE>")
    if prompt_token_range is not None:
        key_tokens.append("<CAPTION>")
    if input_length < len(tokens):
        for i in range(input_length, len(tokens)):
            token = tokens[i][:15] if len(tokens[i]) > 15 else tokens[i]
            key_tokens.append(token)

    # Create heatmap
    num_tokens = len(compressed_tokens)
    fig_height = max(8, num_tokens * 0.3)
    fig, ax = plt.subplots(figsize=(14, fig_height))

    sns.heatmap(
        final_attn,
        xticklabels=key_tokens,
        yticklabels=compressed_tokens,
        cmap='viridis',
        cbar_kws={'label': 'Attention Weight'},
        ax=ax
    )

    ax.set_xlabel('Key Tokens', fontsize=12)
    ax.set_ylabel('Query Tokens', fontsize=12)
    ax.set_title(title, fontsize=14)

    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_attention_map(
    attentions: torch.Tensor,
    tokens: List[str],
    layer_idx: int,
    head_idx: Optional[int] = None,
    input_length: int = 0,
    output_path: str = None,
    title: str = "Attention Map"
):
    """
    Plot attention map heatmap for generated tokens only.

    Args:
        attentions: Attention tensor (batch, layers, heads, seq_len, seq_len)
        tokens: List of token strings
        layer_idx: Which layer to visualize
        head_idx: Which head to visualize (if None, average over all heads)
        input_length: Length of input (non-generated) tokens
        output_path: Path to save plot
        title: Plot title
    """
    # Debug: Print shape and input_length
    print(f"  [DEBUG] Attention shape: {attentions.shape}, input_length: {input_length}, total tokens: {len(tokens)}")

    # Extract attention for specified layer
    if head_idx is not None:
        # Single head: (seq_len, seq_len)
        attn_weights = attentions[0, layer_idx, head_idx, :, :].cpu().numpy()
        title = f"{title} - Layer {layer_idx}, Head {head_idx}"
    else:
        # Average over heads: (seq_len, seq_len)
        attn_weights = attentions[0, layer_idx, :, :, :].mean(dim=0).cpu().numpy()
        title = f"{title} - Layer {layer_idx} (avg heads)"

    print(f"  [DEBUG] Before slicing - attn_weights shape: {attn_weights.shape}")

    # Extract only generated tokens (both query and key)
    if input_length < len(tokens):
        attn_weights = attn_weights[input_length:, input_length:]
        tokens_display = tokens[input_length:]
        print(f"  [DEBUG] After slicing - attn_weights shape: {attn_weights.shape}, tokens_display: {len(tokens_display)}")
    else:
        tokens_display = tokens
        print(f"  [DEBUG] No slicing - using all tokens")

    # Limit tokens for faster rendering
    max_tokens = min(100, len(tokens_display))
    attn_weights = attn_weights[:max_tokens, :max_tokens]
    tokens_display = tokens_display[:max_tokens]

    # Truncate long tokens
    tokens_display = [t[:15] if len(t) > 15 else t for t in tokens_display]

    # Dynamic figure size based on number of tokens
    num_tokens = len(tokens_display)
    fig_size = max(10, int(num_tokens * 0.2))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))

    # Use 'hot' colormap and adjust vmin/vmax for better visibility
    sns.heatmap(
        attn_weights,
        xticklabels=tokens_display,
        yticklabels=tokens_display,
        cmap='hot',
        vmin=0,
        vmax=0.5,  # Lower maximum for brighter visualization
        cbar_kws={'label': 'Attention Weight'},
        ax=ax
    )

    ax.set_xlabel('Key Tokens (Generated)', fontsize=12)
    ax.set_ylabel('Query Tokens (Generated)', fontsize=12)
    ax.set_title(title, fontsize=14)

    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def compute_attention_contribution(
    model,
    input_ids: torch.Tensor,
    pixel_values: Optional[torch.Tensor],
    target_token_idx: int,
    device: str = 'cuda'
):
    """
    Compute layer-wise attention sublayer contribution to final logits.

    This performs forward passes with intervention at each layer to measure
    how much each attention sublayer contributes to the final prediction.

    Args:
        model: LLaVA model with EvalLlama
        input_ids: Input token ids (batch, seq_len)
        pixel_values: Image pixel values (optional)
        target_token_idx: Index of target token to analyze
        device: Device to run on (ignored if model uses device_map)

    Returns:
        Dictionary with layer contributions and predicted tokens
    """
    model.eval()

    # Get model's device (for device_map="auto", use first parameter's device)
    model_device = next(model.parameters()).device

    input_ids = input_ids.to(model_device)
    if pixel_values is not None:
        pixel_values = pixel_values.to(model_device)

    # Get full forward pass outputs
    with torch.no_grad():
        if pixel_values is not None:
            outputs = model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                output_hidden_states=True,
                output_attentions=True,
                use_cache=False
            )
        else:
            outputs = model(
                input_ids=input_ids,
                output_hidden_states=True,
                output_attentions=True,
                use_cache=False
            )

        # Get hidden states from all layers
        # hidden_states: tuple of (batch, seq_len, hidden_dim) for each layer
        hidden_states = outputs.hidden_states

        # Compute final logits using lm_head directly from last hidden state
        # This avoids IndexError from logits_to_keep slicing
        lm_head = model.lm_head
        final_hidden = hidden_states[-1][0, target_token_idx, :]  # (hidden_dim,)
        final_logits = lm_head(final_hidden)  # (vocab_size,)
        final_probs = torch.softmax(final_logits, dim=-1)

    # Compute contribution of each layer
    layer_contributions = []
    predicted_tokens = []

    lm_head = model.lm_head

    for layer_idx in range(len(hidden_states)):
        # Get hidden state at target token position from this layer
        hidden = hidden_states[layer_idx][0, target_token_idx, :]  # (hidden_dim,)

        # Project through lm_head to get logits
        with torch.no_grad():
            logits = lm_head(hidden)  # (vocab_size,)
            probs = torch.softmax(logits, dim=-1)

        # Compute KL divergence or probability difference
        # Contribution = how close this layer's prediction is to final prediction
        contribution = torch.nn.functional.kl_div(
            torch.log_softmax(logits, dim=-1),
            final_probs,
            reduction='sum'
        ).item()

        layer_contributions.append(contribution)

        # Get top predicted token
        top_token_id = torch.argmax(logits).item()
        predicted_tokens.append(top_token_id)

    return {
        'layer_contributions': layer_contributions,
        'predicted_tokens': predicted_tokens,
        'final_token_id': torch.argmax(final_logits).item()
    }


def plot_layer_contributions(
    contributions: List[float],
    layer_range: Optional[Tuple[int, int]] = None,
    output_path: str = None,
    title: str = "Attention Sublayers Contribution"
):
    """
    Plot layer-wise contribution to final output.

    Args:
        contributions: List of contribution scores per layer
        layer_range: Optional (start, end) to highlight specific layers
        output_path: Path to save plot
        title: Plot title
    """
    num_layers = len(contributions)
    layers = np.arange(num_layers)

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot contributions
    bars = ax.bar(layers, contributions, color='steelblue', alpha=0.7)

    # Highlight specific layer range if provided
    if layer_range:
        start, end = layer_range
        for i in range(start, min(end, num_layers)):
            bars[i].set_color('coral')
            bars[i].set_alpha(0.9)

    ax.set_xlabel('Layer Index', fontsize=12)
    ax.set_ylabel('KL Divergence to Final Output', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def retrieve_tokens_via_logit_lens(
    predicted_token_ids: List[int],
    tokenizer,
    output_path: str = None,
    title: str = "Token Prediction via Logit Lens"
):
    """
    Visualize predicted tokens from each layer.

    Args:
        predicted_token_ids: List of predicted token IDs from each layer
        tokenizer: Tokenizer for decoding
        output_path: Path to save plot
        title: Plot title
    """
    # Decode tokens
    tokens = [tokenizer.decode([tid]) for tid in predicted_token_ids]

    # Create visualization
    num_layers = len(tokens)
    layers = np.arange(num_layers)

    fig, ax = plt.subplots(figsize=(14, 8))

    # Plot token changes
    for i, (layer, token) in enumerate(zip(layers, tokens)):
        ax.text(layer, 0, token, ha='center', va='center',
                fontsize=10, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

    ax.set_xlim(-0.5, num_layers - 0.5)
    ax.set_ylim(-0.5, 0.5)
    ax.set_xlabel('Layer Index', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_yticks([])
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return tokens


def analyze_single_sample(
    result: Dict,
    model,
    tokenizer,
    nlp,
    output_dir: str,
    sample_idx: int,
    image_token_id: int = 32000,
    device: str = 'cuda'
) -> Dict:
    """
    Perform all three analyses on a single sample.

    Args:
        result: Result dictionary from inference
        model: LLaVA model
        tokenizer: Tokenizer
        nlp: Spacy model
        output_dir: Directory to save outputs
        sample_idx: Sample index
        image_token_id: Image token ID
        device: Device

    Returns:
        Dictionary with intermediate data for averaging
    """
    sample_dir = os.path.join(output_dir, f"sample_{sample_idx}")
    os.makedirs(sample_dir, exist_ok=True)

    # Store intermediate data for averaging
    sample_data = {
        'image_mode': {},
        'text_mode': {}
    }

    print(f"\n{'='*80}")
    print(f"Analyzing Sample {sample_idx}")
    print(f"{'='*80}")
    print(f"Question: {result['question']}")

    # ========== Image Mode Analysis ==========
    if 'image_mode' in result and result['image_mode']['attentions'] is not None:
        print("\n--- Image Mode ---")
        img_data = result['image_mode']
        generated_ids = img_data['generated_ids'][0]
        attentions = img_data['attentions']
        vision_token_range = img_data.get('vision_token_range')

        tokens = tokenizer.convert_ids_to_tokens(generated_ids)
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        print(f"Generated: {generated_text}")

        # Extract only the generated part (exclude input prompt)
        input_len = img_data['input_ids'].shape[1]
        generated_tokens_only = generated_ids[input_len:]
        generated_text_only = tokenizer.decode(generated_tokens_only, skip_special_tokens=True)

        # Extract nouns from generated text only
        nouns = extract_nouns(generated_text_only, nlp)
        print(f"Nouns (from generated text): {nouns}")

        if nouns and vision_token_range:
            # Find all noun positions in generated text
            all_noun_positions = []
            for noun in nouns:
                positions = find_token_positions(tokens, noun)
                # Only keep positions in generated part
                positions = [p for p in positions if p >= input_len]
                all_noun_positions.extend(positions)

            # Use all noun positions for sorted_head calculation
            # (No limit for comprehensive analysis)

            if all_noun_positions:
                print(f"Analyzing {len(all_noun_positions)} noun positions (averaged)")

                # 0. Attention Distribution (Sorted Heads) - Average over all nouns
                print("  Generating attention distribution (sorted heads)...")
                vision_start, vision_end = vision_token_range

                # Average attention from all noun positions
                all_attn_ratios = []
                for target_idx in all_noun_positions:
                    attn_to_vision = attentions[
                        0, :, :, target_idx, vision_start:vision_end
                    ]
                    attn_ratio = attn_to_vision.sum(dim=-1).cpu().numpy()  # (layers, heads)
                    all_attn_ratios.append(attn_ratio)

                # Average over all nouns
                avg_attn_ratio = np.mean(all_attn_ratios, axis=0)  # (layers, heads)

                sorted_attn = np.zeros_like(avg_attn_ratio)
                for layer_idx in range(avg_attn_ratio.shape[0]):
                    sorted_indices = np.argsort(avg_attn_ratio[layer_idx])
                    sorted_attn[layer_idx] = avg_attn_ratio[layer_idx, sorted_indices]

                sample_data['image_mode']['sorted_attention'] = sorted_attn

                # Use first noun position for visualization title
                first_noun = nouns[0] if nouns else "objects"
                plot_attention_distribution_sorted_heads(
                    attentions,
                    all_noun_positions[0],  # Use first position for plot
                    vision_token_range,
                    output_path=os.path.join(sample_dir, "image_attn_distribution_sorted_heads.png"),
                    title=f"Image Mode Attn. Distribution - avg over {len(all_noun_positions)} nouns"
                )

                # 1. Attention Map Plot (layers 0, 15, 31) - Use first noun for visualization
                print("  Generating attention maps for layers 0, 15, 31...")
                input_len = img_data['input_ids'].shape[1]
                first_noun_idx = all_noun_positions[0]
                first_noun_str = nouns[0] if nouns else "noun"

                for layer_idx in [0, 15, 31]:
                    # Generated tokens only attention map
                    plot_attention_map(
                        attentions,
                        tokens,
                        layer_idx=layer_idx,
                        input_length=input_len,
                        output_path=os.path.join(sample_dir, f"image_attention_map_layer_{layer_idx}.png"),
                        title=f"Image Mode Attention Map Layer {layer_idx} - '{first_noun_str}'"
                    )
                    # Compressed map (image/prompt compressed)
                    plot_attention_map_compressed(
                        attentions,
                        tokens,
                        layer_idx=layer_idx,
                        vision_token_range=vision_token_range,
                        input_length=input_len,
                        output_path=os.path.join(sample_dir, f"image_attention_map_compressed_layer_{layer_idx}.png"),
                        title=f"Image Mode Compressed Attention Layer {layer_idx} - '{first_noun_str}'"
                    )

                # 2. Attention Sublayers Contribution - Average over all nouns (COMMENTED OUT)
                # print("  Computing layer contributions (averaged over all nouns)...")
                # all_layer_contribs = []
                # for target_idx in all_noun_positions:
                #     try:
                #         contribution_result = compute_attention_contribution(
                #             model,
                #             img_data['generated_ids'],  # Use full generated sequence
                #             img_data['pixel_values'],
                #             target_idx,
                #             device=device
                #         )
                #         all_layer_contribs.append(contribution_result['layer_contributions'])
                #     except (IndexError, RuntimeError) as e:
                #         print(f"    Warning: Failed to compute contribution for token {target_idx}: {e}")
                #         continue
                #
                # # Average over all nouns (if any succeeded)
                # if all_layer_contribs:
                #     avg_layer_contribs = np.mean(all_layer_contribs, axis=0).tolist()
                #
                #     # Store layer contributions for averaging
                #     sample_data['image_mode']['layer_contributions'] = avg_layer_contribs
                #
                #     plot_layer_contributions(
                #         avg_layer_contribs,
                #         layer_range=(5, 18),
                #         output_path=os.path.join(sample_dir, "image_layer_contributions.png"),
                #         title=f"Image Mode Layer Contributions - avg over {len(all_layer_contribs)} nouns"
                #     )
                # else:
                #     print(f"    Warning: No layer contributions computed for image mode")

                # 3. Logit Lens Token Retrieval (DISABLED)
                # print("  Retrieving tokens via logit lens...")
                # predicted_tokens = retrieve_tokens_via_logit_lens(
                #     contribution_result['predicted_tokens'],
                #     tokenizer,
                #     output_path=os.path.join(sample_dir, "image_logit_lens.png"),
                #     title=f"Image Mode Logit Lens - '{noun}'"
                # )
                # print(f"  Final predicted token: {tokenizer.decode([contribution_result['final_token_id']])}")

    # ========== Text Mode Analysis ==========
    if 'text_mode' in result and result['text_mode']['attentions'] is not None:
        print("\n--- Text Mode ---")
        txt_data = result['text_mode']
        generated_ids = txt_data['generated_ids'][0]
        attentions = txt_data['attentions']
        prompt_token_range = txt_data.get('prompt_token_range')

        tokens = tokenizer.convert_ids_to_tokens(generated_ids)
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        print(f"Generated: {generated_text}")

        # Extract only the generated part (exclude input prompt)
        input_len_text = txt_data['input_ids'].shape[1]
        generated_tokens_only = generated_ids[input_len_text:]
        generated_text_only = tokenizer.decode(generated_tokens_only, skip_special_tokens=True)

        # Extract nouns from generated text only
        nouns = extract_nouns(generated_text_only, nlp)
        print(f"Nouns (from generated text): {nouns}")

        if nouns and prompt_token_range:
            # Find all noun positions in generated text
            all_noun_positions = []
            for noun in nouns:
                positions = find_token_positions(tokens, noun)
                # Only keep positions in generated part
                positions = [p for p in positions if p >= input_len_text]
                all_noun_positions.extend(positions)

            # Use all noun positions for sorted_head calculation
            # (No limit for comprehensive analysis)

            if all_noun_positions:
                print(f"Analyzing {len(all_noun_positions)} noun positions (averaged)")

                # 0. Attention Distribution (Sorted Heads) - Average over all nouns
                print("  Generating attention distribution (sorted heads)...")
                prompt_start, prompt_end = prompt_token_range

                # Average attention from all noun positions
                all_attn_ratios = []
                for target_idx in all_noun_positions:
                    attn_to_prompt = attentions[
                        0, :, :, target_idx, prompt_start:prompt_end
                    ]
                    attn_ratio_text = attn_to_prompt.sum(dim=-1).cpu().numpy()  # (layers, heads)
                    all_attn_ratios.append(attn_ratio_text)

                # Average over all nouns
                avg_attn_ratio_text = np.mean(all_attn_ratios, axis=0)  # (layers, heads)

                sorted_attn_text = np.zeros_like(avg_attn_ratio_text)
                for layer_idx in range(avg_attn_ratio_text.shape[0]):
                    sorted_indices = np.argsort(avg_attn_ratio_text[layer_idx])
                    sorted_attn_text[layer_idx] = avg_attn_ratio_text[layer_idx, sorted_indices]

                sample_data['text_mode']['sorted_attention'] = sorted_attn_text

                plot_attention_distribution_sorted_heads(
                    attentions,
                    all_noun_positions[0],  # Use first position for plot
                    prompt_token_range,
                    output_path=os.path.join(sample_dir, "text_attn_distribution_sorted_heads.png"),
                    title=f"Text Mode Attn. Distribution - avg over {len(all_noun_positions)} nouns"
                )

                # 1. Attention Map Plot (layers 0, 15, 31) - Use first noun for visualization
                print("  Generating attention maps for layers 0, 15, 31...")
                first_noun_idx = all_noun_positions[0]
                first_noun_str = nouns[0] if nouns else "noun"

                for layer_idx in [0, 15, 31]:
                    # Generated tokens only attention map
                    plot_attention_map(
                        attentions,
                        tokens,
                        layer_idx=layer_idx,
                        input_length=input_len_text,
                        output_path=os.path.join(sample_dir, f"text_attention_map_layer_{layer_idx}.png"),
                        title=f"Text Mode Attention Map Layer {layer_idx} - '{first_noun_str}'"
                    )
                    # Compressed map (caption compressed)
                    plot_attention_map_compressed(
                        attentions,
                        tokens,
                        layer_idx=layer_idx,
                        prompt_token_range=prompt_token_range,
                        input_length=input_len_text,
                        output_path=os.path.join(sample_dir, f"text_attention_map_compressed_layer_{layer_idx}.png"),
                        title=f"Text Mode Compressed Attention Layer {layer_idx} - '{first_noun_str}'"
                    )

                # 2. Attention Sublayers Contribution - Average over all nouns (COMMENTED OUT)
                # print("  Computing layer contributions (averaged over all nouns)...")
                # all_layer_contribs = []
                # for target_idx in all_noun_positions:
                #     try:
                #         contribution_result = compute_attention_contribution(
                #             model,
                #             txt_data['generated_ids'],  # Use full generated sequence
                #             None,  # No pixel values for text mode
                #             target_idx,
                #             device=device
                #         )
                #         all_layer_contribs.append(contribution_result['layer_contributions'])
                #     except (IndexError, RuntimeError) as e:
                #         print(f"    Warning: Failed to compute contribution for token {target_idx}: {e}")
                #         continue
                #
                # # Average over all nouns (if any succeeded)
                # if all_layer_contribs:
                #     avg_layer_contribs = np.mean(all_layer_contribs, axis=0).tolist()
                #
                #     # Store layer contributions for averaging
                #     sample_data['text_mode']['layer_contributions'] = avg_layer_contribs
                #
                #     plot_layer_contributions(
                #         avg_layer_contribs,
                #         layer_range=(5, 18),
                #         output_path=os.path.join(sample_dir, "text_layer_contributions.png"),
                #         title=f"Text Mode Layer Contributions - avg over {len(all_layer_contribs)} nouns"
                #     )
                # else:
                #     print(f"    Warning: No layer contributions computed for text mode")

                # 3. Logit Lens Token Retrieval (DISABLED)
                # print("  Retrieving tokens via logit lens...")
                # predicted_tokens = retrieve_tokens_via_logit_lens(
                #     contribution_result['predicted_tokens'],
                #     tokenizer,
                #     output_path=os.path.join(sample_dir, "text_logit_lens.png"),
                #     title=f"Text Mode Logit Lens - '{noun}'"
                # )
                # print(f"  Final predicted token: {tokenizer.decode([contribution_result['final_token_id']])}")

    return sample_data


def main():
    # Configuration
    results_path = "inference_results/all_results.pt"
    output_dir = "analysis"
    model_name = "llava-hf/llava-1.5-7b-hf"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Analyze all samples in the dataset
    num_samples_to_analyze = None  # None = all samples

    os.makedirs(output_dir, exist_ok=True)

    # Load results
    print(f"Loading results from {results_path}...")
    all_results = torch.load(results_path)
    print(f"Loaded {len(all_results)} results")

    # Load tokenizer and spacy
    print("Loading tokenizer and spacy model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    nlp = load_spacy_model()

    # Load model (need it for contribution analysis)
    print("Loading model for contribution analysis...")
    from src.models.llava.modeling_llava import CustomLlavaForConditionalGeneration
    from transformers import LlavaConfig

    # Load config with EvalLlama
    config = LlavaConfig.from_pretrained(model_name)
    config.text_config.auto_map = {
        "AutoModel": "src.models.eval_llama.modeling_eval_llama.EvalLlamaModel",
        "AutoModelForCausalLM": "src.models.eval_llama.modeling_eval_llama.EvalLlamaForCausalLM"
    }

    model = CustomLlavaForConditionalGeneration.from_pretrained(
        model_name,
        config=config,
        torch_dtype=torch.float16,
        device_map="auto",
        attn_implementation="eager"  # Required for output_attentions=True
    )
    model.eval()

    # Analyze samples and collect data for averaging
    num_samples = len(all_results) if num_samples_to_analyze is None else min(num_samples_to_analyze, len(all_results))
    print(f"\nAnalyzing {num_samples} samples...")

    all_sample_data = []
    for idx in tqdm(range(num_samples), desc="Processing samples"):
        result = all_results[idx]
        sample_data = analyze_single_sample(
            result,
            model,
            tokenizer,
            nlp,
            output_dir,
            result['sample_idx'],
            device=str(device)
        )
        if sample_data:
            all_sample_data.append(sample_data)

    # ========== Generate Averaged Results ==========
    print(f"\nGenerating averaged results from {len(all_sample_data)} samples...")
    averaged_dir = os.path.join(output_dir, "averaged")
    os.makedirs(averaged_dir, exist_ok=True)

    # Collect data for averaging
    image_sorted_attns = []
    # image_layer_contribs = []  # COMMENTED OUT
    text_sorted_attns = []
    # text_layer_contribs = []  # COMMENTED OUT

    for sample_data in all_sample_data:
        # Image mode data
        if 'sorted_attention' in sample_data['image_mode']:
            image_sorted_attns.append(sample_data['image_mode']['sorted_attention'])
        # if 'layer_contributions' in sample_data['image_mode']:  # COMMENTED OUT
        #     image_layer_contribs.append(sample_data['image_mode']['layer_contributions'])

        # Text mode data
        if 'sorted_attention' in sample_data['text_mode']:
            text_sorted_attns.append(sample_data['text_mode']['sorted_attention'])
        # if 'layer_contributions' in sample_data['text_mode']:  # COMMENTED OUT
        #     text_layer_contribs.append(sample_data['text_mode']['layer_contributions'])

    # Average and plot image mode results
    if image_sorted_attns:
        print(f"Averaging image mode attention distribution ({len(image_sorted_attns)} samples)...")
        avg_image_sorted_attn = np.mean(image_sorted_attns, axis=0)

        # Create dummy attentions tensor for plotting (we just need the averaged sorted data)
        # We'll create the plot directly without using plot_attention_distribution_sorted_heads
        num_layers, num_heads = avg_image_sorted_attn.shape

        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(avg_image_sorted_attn, aspect='auto', cmap='Blues', origin='lower', vmin=0, vmax=0.75)
        ax.set_xlabel('Sorted Heads', fontsize=12)
        ax.set_ylabel('Layers', fontsize=12)
        ax.set_title('Averaged Image Mode Attn. Distribution', fontsize=14)
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Visual Attention Ratio', fontsize=11)

        # Mark middle layers (5-18)
        middle_start, middle_end = 5, 18
        ax.axhline(y=middle_start - 0.5, color='red', linestyle='--', linewidth=1, alpha=0.5)
        ax.axhline(y=middle_end + 0.5, color='red', linestyle='--', linewidth=1, alpha=0.5)
        ax.text(
            num_heads * 0.7, (middle_start + middle_end) / 2,
            'high attn. in\nmiddle layers',
            color='red', fontsize=10, ha='left', va='center',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
        )

        plt.tight_layout()
        plt.savefig(os.path.join(averaged_dir, "image_attn_distribution_sorted_heads.png"),
                    dpi=150, bbox_inches='tight')
        plt.close()

    # if image_layer_contribs:  # COMMENTED OUT
    #     print(f"Averaging image mode layer contributions ({len(image_layer_contribs)} samples)...")
    #     avg_image_contribs = np.mean(image_layer_contribs, axis=0)
    #     plot_layer_contributions(
    #         avg_image_contribs.tolist(),
    #         layer_range=(5, 18),
    #         output_path=os.path.join(averaged_dir, "image_layer_contributions.png"),
    #         title="Averaged Image Mode Layer Contributions"
    #     )

    # Average and plot text mode results
    if text_sorted_attns:
        print(f"Averaging text mode attention distribution ({len(text_sorted_attns)} samples)...")
        avg_text_sorted_attn = np.mean(text_sorted_attns, axis=0)

        num_layers, num_heads = avg_text_sorted_attn.shape

        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(avg_text_sorted_attn, aspect='auto', cmap='Blues', origin='lower', vmin=0, vmax=0.75)
        ax.set_xlabel('Sorted Heads', fontsize=12)
        ax.set_ylabel('Layers', fontsize=12)
        ax.set_title('Averaged Text Mode Attn. Distribution', fontsize=14)
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Prompt Attention Ratio', fontsize=11)

        # Mark middle layers (5-18)
        middle_start, middle_end = 5, 18
        ax.axhline(y=middle_start - 0.5, color='red', linestyle='--', linewidth=1, alpha=0.5)
        ax.axhline(y=middle_end + 0.5, color='red', linestyle='--', linewidth=1, alpha=0.5)
        ax.text(
            num_heads * 0.7, (middle_start + middle_end) / 2,
            'high attn. in\nmiddle layers',
            color='red', fontsize=10, ha='left', va='center',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
        )

        plt.tight_layout()
        plt.savefig(os.path.join(averaged_dir, "text_attn_distribution_sorted_heads.png"),
                    dpi=150, bbox_inches='tight')
        plt.close()

    # if text_layer_contribs:  # COMMENTED OUT
    #     print(f"Averaging text mode layer contributions ({len(text_layer_contribs)} samples)...")
    #     avg_text_contribs = np.mean(text_layer_contribs, axis=0)
    #     plot_layer_contributions(
    #         avg_text_contribs.tolist(),
    #         layer_range=(5, 18),
    #         output_path=os.path.join(averaged_dir, "text_layer_contributions.png"),
    #         title="Averaged Text Mode Layer Contributions"
    #     )

    print(f"\nAnalysis complete!")
    print(f"  - Per-sample results: {output_dir}/sample_N/")
    print(f"  - Averaged results: {averaged_dir}/")


if __name__ == "__main__":
    main()