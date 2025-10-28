"""
Inference script for LLaVA with EvalLlama backend.
Processes dataset_question/test with both image+question and caption+question.
Saves input_ids, attention_mask, attention_weights, and model outputs.
"""

import os
import argparse
import torch
from datasets import load_from_disk
from transformers import AutoProcessor, LlavaConfig
from tqdm import tqdm

from src.models.llava.modeling_llava import CustomLlavaForConditionalGeneration
from src.models.eval_llama.modeling_eval_llama import EvalLlamaForCausalLM


def modify_llava_config_for_eval_llama(model_name: str = "llava-hf/llava-1.5-7b-hf"):
    """
    Load LLaVA config and modify it to use EvalLlamaForCausalLM instead of LlamaForCausalLM.

    Args:
        model_name: HuggingFace model name

    Returns:
        Modified LlavaConfig with EvalLlamaForCausalLM as text model
    """
    config = LlavaConfig.from_pretrained(model_name)

    # Modify text_config to use EvalLlamaForCausalLM
    config.text_config.auto_map = {
        "AutoModel": "src.models.eval_llama.modeling_eval_llama.EvalLlamaModel",
        "AutoModelForCausalLM": "src.models.eval_llama.modeling_eval_llama.EvalLlamaForCausalLM"
    }

    return config


def create_chat_messages(question: str, include_image: bool = True):
    """
    Create chat messages for the model.

    Args:
        question: Question text
        include_image: Whether to include image placeholder

    Returns:
        List of chat messages
    """
    if include_image:
        content = [
            {"type": "image"},
            {"type": "text", "text": question}
        ]
    else:
        content = question

    return [
        {"role": "user", "content": content}
    ]


def process_single_sample(
    model,
    processor,
    image,
    caption,
    question,
    device,
    max_new_tokens=512
):
    """
    Process a single sample with both image+question and caption+question.

    Args:
        model: LLaVA model with EvalLlama backend
        processor: LLaVA processor
        image: PIL Image
        caption: Caption text
        question: Question text
        device: torch device
        max_new_tokens: Maximum number of tokens to generate

    Returns:
        Dictionary containing results for both image and caption modes
    """
    results = {}

    # ========== Image + Question ==========
    messages_with_image = create_chat_messages(question, include_image=True)
    prompt_with_image = processor.apply_chat_template(
        messages_with_image,
        add_generation_prompt=True
    )

    inputs_with_image = processor(
        images=image,
        text=prompt_with_image,
        return_tensors="pt"
    ).to(device)

    # Generate with greedy decoding
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs_with_image,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # greedy decoding
            use_cache=False,
        )

        # Forward pass with full generated sequence to get attentions
        # Create new inputs with generated sequence
        full_outputs = model(
            input_ids=generated_ids,
            pixel_values=inputs_with_image['pixel_values'],
            output_attentions=True,
            use_cache=False,
        )

    # Find vision token range (image token ID is typically 32000 for LLaVA)
    image_token_id = 32000
    vision_token_positions = [i for i, token_id in enumerate(generated_ids[0]) if token_id.item() == image_token_id]
    if vision_token_positions:
        vision_token_range = (min(vision_token_positions), max(vision_token_positions) + 1)
    else:
        vision_token_range = None

    # attentions from EvalLlama: (batch, num_layers, num_heads, seq_len, seq_len)
    results['image_mode'] = {
        'input_ids': inputs_with_image['input_ids'].cpu(),
        'attention_mask': inputs_with_image['attention_mask'].cpu(),
        'pixel_values': inputs_with_image['pixel_values'].cpu(),
        'generated_ids': generated_ids.cpu(),
        'attentions': full_outputs.attentions.cpu() if full_outputs.attentions is not None else None,
        'vision_token_range': vision_token_range,
    }

    # ========== Caption + Question (Text-only) ==========
    # Tokenize caption separately to find its range
    caption_only_ids = processor.tokenizer.encode(caption, add_special_tokens=False)
    caption_length = len(caption_only_ids)

    messages_text_only = create_chat_messages(caption + "\n\n" + question, include_image=False)
    prompt_text_only = processor.apply_chat_template(
        messages_text_only,
        add_generation_prompt=True
    )

    inputs_text_only = processor(
        text=prompt_text_only,
        return_tensors="pt"
    ).to(device)

    # Generate with greedy decoding
    with torch.no_grad():
        generated_ids_text = model.generate(
            **inputs_text_only,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # greedy decoding
            use_cache=False,
        )

        # Forward pass with full generated sequence to get attentions
        full_outputs_text = model(
            input_ids=generated_ids_text,
            output_attentions=True,
            use_cache=False,
        )

    # Find caption token range by searching for the caption token sequence
    input_ids_list = generated_ids_text[0].tolist()
    caption_start_idx = None

    # Search for caption token sequence in the input
    for i in range(len(input_ids_list) - len(caption_only_ids) + 1):
        if input_ids_list[i:i+len(caption_only_ids)] == caption_only_ids:
            caption_start_idx = i
            break

    if caption_start_idx is not None:
        prompt_token_range = (caption_start_idx, caption_start_idx + len(caption_only_ids))
    else:
        # Fallback: approximate based on caption length
        prompt_length = inputs_text_only['input_ids'].shape[1]
        prompt_token_range = (0, min(caption_length + 10, prompt_length))

    # attentions from EvalLlama: (batch, num_layers, num_heads, seq_len, seq_len)
    results['text_mode'] = {
        'input_ids': inputs_text_only['input_ids'].cpu(),
        'attention_mask': inputs_text_only['attention_mask'].cpu(),
        'generated_ids': generated_ids_text.cpu(),
        'attentions': full_outputs_text.attentions.cpu() if full_outputs_text.attentions is not None else None,
        'prompt_token_range': prompt_token_range,
        'caption_length': caption_length,
    }

    return results


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Run inference with LLaVA and save results")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Number of samples to process (default: all samples)")
    parser.add_argument("--model_name", type=str, default="llava-hf/llava-1.5-7b-hf",
                        help="Model name or path")
    parser.add_argument("--dataset_path", type=str, default="dataset_question/test",
                        help="Path to dataset")
    parser.add_argument("--output_dir", type=str, default="inference_results",
                        help="Output directory")
    args = parser.parse_args()

    # Configuration
    model_name = args.model_name
    dataset_path = args.dataset_path
    output_dir = args.output_dir
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    # Load dataset
    print(f"Loading dataset from {dataset_path}...")
    dataset = load_from_disk(dataset_path)
    total_samples = len(dataset)
    print(f"Dataset loaded: {total_samples} samples")

    # Limit samples if specified
    if args.num_samples is not None:
        num_samples = min(args.num_samples, total_samples)
        dataset = dataset.select(range(num_samples))
        print(f"Processing {num_samples} samples (out of {total_samples})")
    else:
        num_samples = total_samples
        print(f"Processing all {num_samples} samples")

    print(f"Columns: {dataset.column_names}")

    # Load processor
    print(f"Loading processor from {model_name}...")
    processor = AutoProcessor.from_pretrained(model_name)

    # Load config and modify for EvalLlama
    print(f"Loading and modifying config for EvalLlama...")
    config = modify_llava_config_for_eval_llama(model_name)

    # Load model with modified config
    print(f"Loading LLaVA model with EvalLlama backend...")
    model = CustomLlavaForConditionalGeneration.from_pretrained(
        model_name,
        config=config,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model.eval()

    print(f"Model loaded successfully")
    print(f"Language model type: {type(model.language_model)}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Process each sample
    all_results = []
    print(f"\nProcessing {len(dataset)} samples...")

    for idx, sample in enumerate(tqdm(dataset, desc="Inference")):
        image = sample['image']
        caption = sample['caption']
        question = sample['question']

        try:
            results = process_single_sample(
                model=model,
                processor=processor,
                image=image,
                caption=caption,
                question=question,
                device=device,
                max_new_tokens=512
            )

            # Add sample metadata
            results['sample_idx'] = idx
            results['question'] = question
            results['caption'] = caption[:200]  # Store truncated caption for reference

            all_results.append(results)

            # Save intermediate results every 10 samples
            if (idx + 1) % 10 == 0:
                checkpoint_path = os.path.join(output_dir, f"checkpoint_{idx + 1}.pt")
                torch.save(all_results, checkpoint_path)
                print(f"Saved checkpoint at sample {idx + 1}")

        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            continue

    # Save final results
    final_path = os.path.join(output_dir, "all_results.pt")
    torch.save(all_results, final_path)
    print(f"\nAll results saved to {final_path}")
    print(f"Total samples processed: {len(all_results)}")


if __name__ == "__main__":
    main()