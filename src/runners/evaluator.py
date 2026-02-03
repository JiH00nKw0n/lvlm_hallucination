import asyncio
import json
import logging
import os
import sys
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch
from accelerate import PartialState
from accelerate.utils import gather_object
from datasets import tqdm
from pydantic import Field
from torch.utils.data import DataLoader
from trl.data_utils import apply_chat_template

from src.common.registry import registry
from src.runners.base import BaseEvaluator
from src.integrations.saebench_autointerp import (
    AutoInterpConfig,
    AutoInterpRunner,
    get_feature_activation_sparsity_hidden,
    load_tokenized_dataset_cached,
)
from src.integrations.saebench_core_eval import (
    compute_l0,
    compute_loss_recovered,
    load_tokenized_text_dataset,
)

logger = logging.getLogger(__name__)

__all__ = [
    "LVLMEvaluator",
    "POPEEvaluator",
    "MMEEvaluator",
    "TextAutoInterpEvaluator",
    "ImageAutoInterpEvaluator",
    "L0Evaluator",
    "LossRecoveredEvaluator",
]


def default_collate_fn(batch: List[Dict]) -> List[Dict]:
    """
    Default collate function that returns the batch as-is.

    This is used to bypass PyTorch's default collation which expects tensors.
    The actual batching and processing happens in _prepare_batch.

    Args:
        batch: List of raw dataset samples

    Returns:
        The input batch unchanged
    """
    return batch


class LVLMEvaluator(BaseEvaluator):
    """
    Abstract base class for evaluating Large Vision-Language Models (LVLMs).

    This class provides a unified interface for LVLM inference and evaluation,
    handling chat template formatting, batch processing, and answer generation.

    Uses Accelerate's PartialState for distributed inference across multiple GPUs.

    Attributes:
        generation_config (Optional[Dict]): Configuration for model.generate() method.
            Common parameters:
                - max_new_tokens (int): Maximum number of tokens to generate
                - temperature (float): Sampling temperature (0.0 for greedy)
                - do_sample (bool): Whether to use sampling
                - top_p (float): Nucleus sampling parameter
                - top_k (int): Top-k sampling parameter
        decode_fn (Optional[Callable]): Custom decoding function for generation outputs.
            If None, uses default greedy decoding with processor.batch_decode.
        distributed_state (PartialState): Accelerate distributed state (singleton).
            Initialized once and reused across all methods.

    Note:
        The processor is accessed via self.data_collator.processor, which is passed
        from the evaluation task.

    Methods to implement in subclasses:
        _prepare_conversation(sample: dict) -> list:
            Convert a dataset sample to chat template format.

        _parse_answer(generated_text: str, sample: dict) -> str:
            Parse the generated text to extract the answer.

        _compute_metrics(results: List[Dict]) -> dict:
            Compute evaluation metrics from generation results.

    Example:
        >>> class MyEvaluator(LVLMEvaluator):
        ...     def _prepare_conversation(self, sample):
        ...         return [{
        ...             "role": "user",
        ...             "content": [
        ...                 {"type": "image"},
        ...                 {"type": "text", "text": sample["question"]}
        ...             ]
        ...         }]
        ...
        ...     def _parse_answer(self, generated_text, sample):
        ...         return generated_text.strip().lower()
        ...
        ...     def _compute_metrics(self, results):
        ...         correct = sum(r["parsed_answer"] == r["answer"] for r in results)
        ...         return {"accuracy": correct / len(results)}
    """

    generation_config: Optional[Dict] = Field(
        default_factory=lambda: {
            "max_new_tokens": 10,
            "temperature": 0.0,
            "do_sample": False,
        }
    )
    decoding_config: Optional[Dict] = None
    decode_fn: Optional["Callable"] = None  # type: ignore
    distributed_state: Optional[PartialState] = None  # type: ignore

    def model_post_init(self, __context):
        """Initialize distributed state after model creation."""
        super().model_post_init(__context)
        # Initialize PartialState once for this evaluator instance
        self.distributed_state = PartialState()

    def _prepare_conversation(self, sample: dict) -> list:
        """
        Prepare a single sample as a conversation for the chat template.

        Args:
            sample: A dictionary containing dataset sample with keys like 'image', 'question', etc.

        Returns:
            A conversation list in the format expected by processor.apply_chat_template:
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": "Question text"}
                    ]
                }
            ]

        Raises:
            NotImplementedError: Must be implemented by subclass.
        """
        raise NotImplementedError("Subclasses must implement _prepare_conversation")

    def _prepare_batch(self, batch: List[dict]) -> Dict[str, torch.Tensor]:
        """
        Prepare a batch of samples for model inference.

        This method:
        1. Converts each sample to conversation format
        2. Applies chat template using TRL's apply_chat_template utility
        3. Tokenizes formatted text with images using processor
        4. Returns batched tensors ready for model.generate()

        Args:
            batch: List of raw dataset samples

        Returns:
            Dictionary of batched input tensors (input_ids, attention_mask, pixel_values, etc.)
        """

        # Access processor through data_collator
        processor = self.data_collator.processor

        # Prepare conversations for each sample and apply chat template
        formatted_texts = []
        for sample in batch:
            conversation = self._prepare_conversation(sample)
            # Use TRL's apply_chat_template
            formatted = apply_chat_template(
                {"messages": conversation},
                processor,
                add_generation_prompt=True
            )
            formatted_texts.append(formatted["text"])

        # Tokenize the batch
        inputs = processor(
            text=formatted_texts,
            images=[sample["image"] for sample in batch],
            return_tensors="pt",
            padding=True
        )

        # Move to model device
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        return inputs

    def _build_mitigator(self):
        """
        Build a decoding mitigator from config, if provided.

        Expected decoding_config format:
            {
                "name": "VCDMitigator",
                "model_type": "llava-hf/llava-1.5-7b-hf",
                "config": { ... }  # mitigator kwargs
            }
        """
        if not self.decoding_config:
            return None

        # Ensure mitigators are registered
        import src.decoding  # noqa: F401

        name = self.decoding_config.get("name")
        if not name:
            raise ValueError("decoding_config requires 'name'")

        mitigator_cls = registry.get_mitigator_class(name)
        if mitigator_cls is None and name == "GreedyMitigator":
            from src.decoding.greedy import GreedyMitigator
            registry.register_mitigator("GreedyMitigator")(GreedyMitigator)
            mitigator_cls = registry.get_mitigator_class(name)
        if mitigator_cls is None:
            available = ", ".join(registry.list_mitigators())
            raise ValueError(f"Unknown mitigator: {name}. Available: {available}")

        model_type_raw = self.decoding_config.get(
            "model_type",
            self.decoding_config.get("config", {}).get("model_type", "llava-hf/llava-1.5-7b-hf"),
        )
        model_type = self._resolve_model_type(model_type_raw)
        mitigator_kwargs = dict(self.decoding_config.get("config", {}))
        mitigator_kwargs["model_type"] = model_type

        # Map generation_config to mitigator config defaults when not specified
        gen_cfg = self.generation_config or {}
        for key in ("max_new_tokens", "temperature", "top_p", "top_k", "do_sample"):
            if key not in mitigator_kwargs and key in gen_cfg:
                mitigator_kwargs[key] = gen_cfg[key]

        return mitigator_cls(self.model, **mitigator_kwargs)

    @staticmethod
    def _resolve_model_type(model_type: str) -> str:
        """
        Resolve internal model_type from a HuggingFace repo id.

        Examples:
            llava-hf/llava-1.5-7b-hf -> llava
            llava-hf/llava-next-7b   -> llava_next
            Qwen/Qwen2-VL-7B-Instruct -> qwen2_vl
            Qwen/Qwen2.5-VL-7B-Instruct -> qwen2_5_vl
        """
        model_type_norm = model_type.lower()

        if "llava-next" in model_type_norm or "llava_next" in model_type_norm:
            return "llava_next"
        if "llava" in model_type_norm:
            return "llava"
        if "qwen2.5-vl" in model_type_norm or "qwen2_5_vl" in model_type_norm:
            return "qwen2_5_vl"
        if "qwen2-vl" in model_type_norm or "qwen2_vl" in model_type_norm:
            return "qwen2_vl"

        # Allow internal model_type directly
        if model_type_norm in ("llava", "llava_next", "qwen2_vl", "qwen2_5_vl"):
            return model_type_norm

        raise ValueError(
            "Unsupported model_type. Use a HuggingFace repo id such as "
            "'llava-hf/llava-1.5-7b-hf' or 'Qwen/Qwen2-VL-7B-Instruct'."
        )

    def _decode_outputs(self, output_ids: torch.Tensor, input_ids: torch.Tensor) -> List[str]:
        """
        Decode model outputs to text.

        Args:
            output_ids: Generated token IDs from model.generate() [batch_size, seq_len]
            input_ids: Input token IDs [batch_size, input_len]

        Returns:
            List of generated text strings
        """
        processor = self.data_collator.processor

        if self.decode_fn is not None:
            # Use custom decoding function
            return self.decode_fn(output_ids, input_ids, processor)

        # Default: remove input tokens and decode
        # output_ids includes input_ids, so we slice them out
        generated_ids = output_ids[:, input_ids.shape[1]:]

        generated_texts = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )

        return generated_texts

    def _parse_answer(self, generated_text: str, sample: dict) -> str:
        """
        Parse generated text to extract the answer.

        Args:
            generated_text: The text generated by the model
            sample: The original dataset sample (for context if needed)

        Returns:
            Parsed answer string

        Raises:
            NotImplementedError: Must be implemented by subclass.
        """
        raise NotImplementedError("Subclasses must implement _parse_answer")

    def _compute_metrics(self, results: List[Dict]) -> dict:
        """
        Compute evaluation metrics from generation results.

        Args:
            results: List of dictionaries containing:
                - All original sample fields
                - 'generated': Raw generated text
                - 'parsed_answer': Parsed answer

        Returns:
            Dictionary of metrics (e.g., {"accuracy": 0.85, "f1": 0.82})

        Raises:
            NotImplementedError: Must be implemented by subclass.
        """
        raise NotImplementedError("Subclasses must implement _compute_metrics")

    @torch.no_grad()
    def _generate_answers(self, batch_size: int = 32) -> List[Dict]:
        """
        Generate answers for all samples in the evaluation dataset using distributed inference.

        Uses Accelerate's PartialState to split dataset across available GPUs for parallel processing.

        Args:
            batch_size: Number of samples to process in parallel per GPU

        Returns:
            List of result dictionaries containing original sample data plus:
                - 'generated': Raw generated text
                - 'parsed_answer': Parsed answer

        Note:
            Uses apply_padding=True to ensure all processes have equal-length data for gathering.
            Duplicate samples from padding are identified by _sample_idx and removed after gathering.
        """
        # Add index to track original order and identify padding duplicates
        batch_size = self.batch_size if batch_size is not None else batch_size

        # Lazy indexing: use HuggingFace Dataset.map() instead of list comprehension
        # This avoids loading all images into memory upfront
        indexed_dataset = self.evaluate_dataset.map(
            lambda example, idx: {"_sample_idx": idx, **example},
            with_indices=True,
            desc="Adding sample indices (lazy)",
        )

        # Split indexed dataset across GPUs with padding
        with self.distributed_state.split_between_processes(
                indexed_dataset,
                apply_padding=True  # Pad to make lengths equal for gathering
        ) as process_dataset:
            results = []
            mitigator = self._build_mitigator()

            # Create dataloader for batch processing with parallel loading
            dataloader = DataLoader(
                process_dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=default_collate_fn,
                num_workers=4,
                prefetch_factor=2,
                pin_memory=True,
            )

            def run_batch(batch_inputs, batch_samples):
                if mitigator is None:
                    output_ids = self.model.generate(
                        **batch_inputs,
                        **self.generation_config
                    )
                else:
                    output_ids = mitigator.generate(**batch_inputs)

                generated_texts = self._decode_outputs(output_ids, batch_inputs['input_ids'])

                for sample, generated_text in zip(batch_samples, generated_texts):
                    parsed_answer = self._parse_answer(generated_text, sample)

                    results.append(
                        {
                            **sample,
                            'generated': generated_text,
                            'parsed_answer': parsed_answer
                        }
                    )

            if mitigator is None:
                for batch in tqdm(
                        dataloader,
                        desc=f"Generating answers for {self.dataset_name} (GPU {self.distributed_state.process_index})",
                        disable=not self.distributed_state.is_main_process
                ):
                    inputs = self._prepare_batch(batch)
                    run_batch(inputs, batch)
            else:
                with mitigator:
                    for batch in tqdm(
                            dataloader,
                            desc=f"Generating answers for {self.dataset_name} (GPU {self.distributed_state.process_index})",
                            disable=not self.distributed_state.is_main_process
                    ):
                        inputs = self._prepare_batch(batch)
                        run_batch(inputs, batch)

        # Gather results from all processes
        all_results = gather_object(results)

        # Remove duplicates by tracking seen indices (from padding)
        seen_indices = set()
        unique_results = []
        for result in all_results:
            idx = result.get("_sample_idx")
            if idx is not None and idx not in seen_indices:
                seen_indices.add(idx)
                # Remove internal tracking field
                result_clean = {k: v for k, v in result.items() if k != "_sample_idx"}
                unique_results.append((idx, result_clean))

        # Sort by original index to maintain dataset order (important for MME!)
        unique_results.sort(key=lambda x: x[0])

        # Extract just the results
        final_results = [result for _, result in unique_results]

        return final_results

    def evaluate(self, batch_size: int = 32):
        """
        Run the complete evaluation pipeline.

        Args:
            batch_size: Number of samples to process in parallel

        Returns:
            None. Results are saved to output_dir.
        """
        # Skip if results already exist and overwrite is disabled
        if self.output_dir is not None:
            result_path = os.path.join(self.output_dir, f'{self.dataset_name}.json')
            if not self.overwrite_results and os.path.exists(result_path):
                logger.info(f"Skipping {self.dataset_name} - results already exist at {result_path}")
                return

        # Generate answers for all samples
        results = self._generate_answers(batch_size)

        # Compute metrics
        metrics = self._compute_metrics(results)

        # Log metrics
        logger.info(f"Evaluation results for {self.dataset_name}:")
        for metric_name, metric_value in metrics.items():
            logger.info(f"  {metric_name}: {metric_value}")

        # Save results
        self._save_result(metrics)


@registry.register_evaluator("POPEEvaluator")
class POPEEvaluator(LVLMEvaluator):
    """
    Evaluator for the POPE (Polling-based Object Probing Evaluation) benchmark.

    POPE evaluates object hallucination in LVLMs through yes/no questions about
    object presence in images. The benchmark includes three sampling strategies:
    - random: Random negative object sampling
    - popular: Popular objects as negative samples
    - adversarial: Adversarial negative sampling (most challenging)

    Metrics computed:
        - Accuracy: Overall correctness
        - Precision: TP / (TP + FP)
        - Recall: TP / (TP + FN)
        - F1-score: Harmonic mean of precision and recall
        - Yes ratio: Proportion of "yes" predictions

    Example:
        >>> from src.datasets.pope import POPEDatasetBuilder
        >>> from transformers import AutoProcessor, AutoModel
        >>> from src.common.registry import registry
        >>>
        >>> # Load dataset
        >>> dataset = POPEDatasetBuilder().build_dataset()
        >>>
        >>> # Load model and processor
        >>> model = AutoModel.from_pretrained("llava-hf/llava-1.5-7b-hf", device_map="auto")
        >>> processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
        >>>
        >>> # Create collator
        >>> collator_cls = registry.get_collator_class("DummyImageCollator")
        >>> collator = collator_cls(processor=processor)
        >>>
        >>> # Evaluate
        >>> evaluator = POPEEvaluator(
        ...     model=model,
        ...     data_collator=collator,
        ...     evaluate_dataset=dataset,
        ...     output_dir="./results"
        ... )
        >>> evaluator.evaluate(batch_size=8)
    """
    dataset_name: Optional[str] = "POPE"

    def _prepare_conversation(self, sample: dict) -> list:
        """
        Prepare a POPE sample as a conversation.

        Args:
            sample: Dictionary with keys 'image' and 'question'

        Returns:
            Conversation list for chat template
        """
        question_with_instruction = f"{sample['question']} Please answer this question with one word."
        
        return [{
            "role": "user",
            "content": [
                {"type": "image", "image": sample["image"]},
                {"type": "text", "text": question_with_instruction}
            ]
        }]

    def _parse_answer(self, generated_text: str, sample: dict) -> str:
        """
        Parse generated text to extract yes/no answer following POPE official logic.

        Official POPE evaluation:
        1. Keep only the first sentence (before '.')
        2. Remove commas and split into words
        3. If "No", "not", or "no" in words -> "no"
        4. Otherwise -> "yes"

        Reference: https://github.com/AoiDragon/POPE

        Args:
            generated_text: Raw generated text from model
            sample: Original sample (unused but required by interface)

        Returns:
            "yes" or "no" (no "other" category in official POPE)
        """
        text = generated_text.strip()

        # Only keep the first sentence
        if '.' in text:
            text = text.split('.')[0]

        # Remove commas and split into words
        text = text.replace(',', '')
        words = text.split(' ')

        # Check for negative indicators
        if 'No' in words or 'not' in words or 'no' in words:
            return "no"
        else:
            return "yes"

    def _compute_metrics(self, results: List[Dict]) -> dict:
        """
        Compute POPE evaluation metrics following the official calculation method.

        Official POPE metrics:
        - Accuracy: (TP + TN) / Total
        - Precision: TP / (TP + FP)  [yes=positive]
        - Recall: TP / (TP + FN)
        - F1: 2 * Precision * Recall / (Precision + Recall)
        - Yes ratio: predicted "yes" count / total count

        Reference: https://github.com/AoiDragon/POPE

        Args:
            results: List of result dicts with 'parsed_answer', 'answer', 'category'

        Returns:
            Dictionary with overall and per-category metrics
        """
        # Group results by category
        category_results = {"all": results}
        for category in ["random", "popular", "adversarial"]:
            category_results[category] = [
                r for r in results if r.get("category") == category
            ]

        # Compute metrics for each category
        metrics = {}
        for category_name, category_data in category_results.items():
            if len(category_data) == 0:
                continue

            # Convert to binary: yes=1 (positive), no=0 (negative)
            label_map = {"yes": 1, "no": 0}
            y_true = [label_map.get(r["answer"], 1) for r in category_data]
            y_pred = [label_map.get(r["parsed_answer"], 1) for r in category_data]

            # Calculate confusion matrix components
            TP = TN = FP = FN = 0
            for pred, label in zip(y_pred, y_true):
                if pred == 1 and label == 1:
                    TP += 1
                elif pred == 1 and label == 0:
                    FP += 1
                elif pred == 0 and label == 0:
                    TN += 1
                elif pred == 0 and label == 1:
                    FN += 1

            # Calculate metrics (with zero division handling)
            total = TP + TN + FP + FN
            acc = (TP + TN) / total if total > 0 else 0.0
            prec = TP / (TP + FP) if (TP + FP) > 0 else 0.0
            rec = TP / (TP + FN) if (TP + FN) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            yes_ratio = sum(y_pred) / len(y_pred) if len(y_pred) > 0 else 0.0

            metrics[category_name] = {
                "accuracy": float(acc),
                "precision": float(prec),
                "recall": float(rec),
                "f1": float(f1),
                "yes_ratio": float(yes_ratio),
                "TP": TP,
                "FP": FP,
                "TN": TN,
                "FN": FN,
                "total_count": len(category_data)
            }

        return metrics


@registry.register_evaluator("MMEEvaluator")
class MMEEvaluator(LVLMEvaluator):
    """
    Evaluator for the MME (Multimodal Evaluation) benchmark.

    MME evaluates both perception and cognition abilities through yes/no questions.

    Task categories:
        Perception: existence, count, position, color, posters, celebrity,
                   scene, landmark, artwork, OCR
        Cognition: commonsense_reasoning, numerical_calculation,
                  text_translation, code_reasoning

    Scoring:
        - Each image has 2 questions
        - acc: Standard accuracy per question
        - acc_plus: Both questions for an image must be correct
        - Total score per category: (acc + acc_plus) * 100

    Example:
        >>> from transformers import AutoProcessor, AutoModel

from src.common.registry import registry
from src.datasets.mme import MMEDatasetBuilder, EVAL_TYPE_DICT
        >>>
        >>> # Load dataset
        >>> dataset = MMEDatasetBuilder().build_dataset()
        >>>
        >>> # Load model
        >>> model = AutoModel.from_pretrained("llava-hf/llava-1.5-7b-hf", device_map="auto")
        >>> processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
        >>>
        >>> # Create collator
        >>> collator_cls = registry.get_collator_class("DummyImageCollator")
        >>> collator = collator_cls(processor=processor)
        >>>
        >>> # Evaluate
        >>> evaluator = MMEEvaluator(
        ...     model=model,
        ...     data_collator=collator,
        ...     evaluate_dataset=dataset,
        ...     output_dir="./results"
        ... )
        >>> evaluator.evaluate(batch_size=8)
    """
    dataset_name: Optional[str] = "MME"

    def _prepare_conversation(self, sample: dict) -> list:
        """
        Prepare an MME sample as a conversation.

        Args:
            sample: Dictionary with keys 'image' and 'question'

        Returns:
            Conversation list for chat template
        """
        return [{
            "role": "user",
            "content": [
                {"type": "image", "image": sample["image"]},
                {"type": "text", "text": sample["question"]}
            ]
        }]

    def _parse_answer(self, generated_text: str, sample: dict) -> str:
        """
        Parse generated text to extract yes/no answer.

        Args:
            generated_text: Raw generated text from model
            sample: Original sample (unused)

        Returns:
            "yes", "no", or "other"
        """
        text_lower = generated_text.strip().lower()

        if text_lower in ["yes", "no"]:
            return text_lower

        prefix = text_lower[:4]
        if "yes" in prefix:
            return "yes"
        elif "no" in prefix:
            return "no"
        else:
            return "other"

    def _compute_metrics(self, results: List[Dict]) -> dict:
        """
        Compute MME evaluation metrics following the official calculation method.

        MME groups questions sequentially (every 2 consecutive questions belong to the same image)
        and computes:
        - acc: Per-question accuracy
        - acc_plus: Accuracy where both questions per image are correct
        - precision: TP / (TP + FP) - excludes "other" responses
        - recall: TP / (TP + FN) - excludes "other" responses
        - Total score per category: (acc + acc_plus) * 100

        Args:
            results: List of result dicts with 'parsed_answer', 'answer', 'category'
                     Results MUST be in the original dataset order

        Returns:
            Dictionary with per-category and per-task-type scores
        """
        from collections import defaultdict
        from sklearn.metrics import precision_score, recall_score, confusion_matrix
        from src.datasets.mme import EVAL_TYPE_DICT

        # Group results by category
        category_groups = defaultdict(list)
        for result in results:
            category = result.get("category")
            if category:
                category_groups[category].append(result)

        # Compute metrics for each category
        category_scores = {}

        for category, category_results in category_groups.items():
            # MME: Every 2 consecutive questions belong to the same image
            # Following official calculation.py logic
            total_questions = len(category_results)
            correct_questions = 0
            correct_images = 0
            total_images = total_questions // 2  # Each image has 2 questions

            # Collect all predictions and ground truths for precision/recall
            gts = []
            preds = []

            # Process in chunks of 2 (one image = 2 questions)
            for i in range(0, total_questions, 2):
                # Handle potential odd number of questions
                if i + 1 >= total_questions:
                    # Single question remaining
                    result = category_results[i]
                    gts.append(result["answer"])
                    preds.append(result["parsed_answer"])

                    if result["parsed_answer"] == result["answer"]:
                        correct_questions += 1
                    break

                # Get 2 questions for this image
                result1 = category_results[i]
                result2 = category_results[i + 1]

                gts.append(result1["answer"])
                gts.append(result2["answer"])
                preds.append(result1["parsed_answer"])
                preds.append(result2["parsed_answer"])

                image_correct_count = 0

                # Check first question
                if result1["parsed_answer"] == result1["answer"]:
                    correct_questions += 1
                    image_correct_count += 1

                # Check second question
                if result2["parsed_answer"] == result2["answer"]:
                    correct_questions += 1
                    image_correct_count += 1

                # acc_plus: both questions must be correct
                if image_correct_count == 2:
                    correct_images += 1

            # Calculate basic metrics
            acc = correct_questions / total_questions if total_questions > 0 else 0
            acc_plus = correct_images / total_images if total_images > 0 else 0

            # MME score: (acc + acc_plus) * 100
            score = (acc + acc_plus) * 100

            # Compute precision, recall, and confusion matrix (following calculation.py)
            # Convert to binary labels: yes=1, no=0, other=-1
            label_map = {"yes": 1, "no": 0, "other": -1}
            gts_binary = [label_map[x] for x in gts]
            preds_binary = [label_map[x] for x in preds]

            # Filter out "other" predictions for precision/recall calculation
            clean_gts = []
            clean_preds = []
            other_num = 0

            for gt, pred in zip(gts_binary, preds_binary):
                if pred == -1:
                    other_num += 1
                    continue
                clean_gts.append(gt)
                clean_preds.append(pred)

            # Calculate precision, recall, and confusion matrix
            precision = 0.0
            recall = 0.0
            tp = fn = fp = tn = 0

            if len(clean_gts) > 0:
                precision = precision_score(clean_gts, clean_preds, average='binary', zero_division=0)
                recall = recall_score(clean_gts, clean_preds, average='binary', zero_division=0)

                conf_mat = confusion_matrix(clean_gts, clean_preds, labels=[1, 0])
                tp, fn = conf_mat[0]
                fp, tn = conf_mat[1]

            category_scores[category] = {
                "acc": float(acc),
                "acc_plus": float(acc_plus),
                "score": float(score),
                "precision": float(precision),
                "recall": float(recall),
                "TP": int(tp),
                "FN": int(fn),
                "FP": int(fp),
                "TN": int(tn),
                "other_num": int(other_num),
                "total_questions": total_questions,
                "total_images": total_images
            }

        # Aggregate by task type (Perception vs Cognition)
        task_type_scores = {}
        for task_type, categories in EVAL_TYPE_DICT.items():
            type_total_score = 0
            type_category_scores = {}

            for category in categories:
                if category in category_scores:
                    type_total_score += category_scores[category]["score"]
                    type_category_scores[category] = category_scores[category]

            task_type_scores[task_type] = {
                "total_score": float(type_total_score),
                "categories": type_category_scores
            }

        return {
            "task_types": task_type_scores,
            "all_categories": category_scores
        }


@registry.register_evaluator("TextAutoInterpEvaluator")
class TextAutoInterpEvaluator(BaseEvaluator):
    """
    Text-only auto-interpretation evaluator reimplementing SAEBench autointerp
    with hidden_states from a HuggingFace model.
    """

    dataset_name: Optional[str] = "TextAutoInterp"
    text_column: str = "text"

    # Tokenization / sampling
    ctx_len: int = 128
    total_tokens: int = 2_000_000
    llm_batch_size: int = 32
    buffer: int = 10
    no_overlap: bool = True
    act_threshold_frac: float = 0.01

    # Latent selection
    n_latents: int = 1000
    override_latents: Optional[List[int]] = None
    dead_latent_threshold: float = 15
    random_seed: int = 42

    # Prompting / scoring
    openai_api_key: Optional[str] = None
    openai_api_key_path: Optional[str] = None
    openai_model: str = "gpt-4o-mini"
    scoring: bool = True
    max_tokens_in_explanation: int = 30
    use_demos_in_explanation: bool = True
    n_top_ex_for_generation: int = 10
    n_iw_sampled_ex_for_generation: int = 5
    n_top_ex_for_scoring: int = 2
    n_random_ex_for_scoring: int = 10
    n_iw_sampled_ex_for_scoring: int = 2

    # SAE / model interface
    sae_model_cls: str = "TopKSAE"
    sae_path: Optional[str] = None
    sae_dtype: str = "float16"
    sae_device: Optional[str] = None
    hidden_state_layer: int = 24
    hidden_state_index: Optional[int] = None
    tokenizer_name_or_path: Optional[str] = None

    # Caching
    artifacts_dir: Optional[str] = "./artifacts"

    def _resolve_openai_key(self) -> str:
        if self.openai_api_key:
            return self.openai_api_key
        if self.openai_api_key_path:
            with open(self.openai_api_key_path, "r") as f:
                return f.read().strip()
        raise ValueError("OpenAI API key missing (openai_api_key or openai_api_key_path).")

    def _resolve_hidden_state_index(self) -> int:
        if self.hidden_state_index is not None:
            return self.hidden_state_index
        return self.hidden_state_layer + 1

    def _load_tokenizer(self):
        from transformers import AutoTokenizer

        name_or_path = self.tokenizer_name_or_path
        if name_or_path is None:
            name_or_path = getattr(self.model.config, "_name_or_path", None) or getattr(
                self.model, "name_or_path", None
            )
        if name_or_path is None:
            raise ValueError("tokenizer_name_or_path is required when it cannot be inferred.")
        tokenizer = AutoTokenizer.from_pretrained(name_or_path, use_fast=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def _load_sae(self):
        from src import models  # noqa: F401
        from src.common.registry import registry as _registry

        if not self.sae_path:
            raise ValueError("sae_path is required for TextAutoInterpEvaluator.")
        sae_cls = _registry.get_model_class(self.sae_model_cls)
        if sae_cls is None:
            raise ValueError(f"Unknown SAE model class: {self.sae_model_cls}")

        dtype = getattr(torch, self.sae_dtype, None)
        if dtype is None:
            raise ValueError(f"Unsupported sae_dtype: {self.sae_dtype}")
        sae = sae_cls.from_pretrained(self.sae_path, torch_dtype=dtype)
        device = self.sae_device or str(self.model.device)
        sae = sae.to(device=device).eval()
        return sae

    def evaluate(self, batch_size: int = 32):
        if self.output_dir is None:
            raise ValueError("output_dir is required for TextAutoInterpEvaluator.")
        if self.output_dir is not None:
            result_path = os.path.join(self.output_dir, f"{self.dataset_name.upper()}.json")
            if not self.overwrite_results and os.path.exists(result_path):
                logger.info(f"Skipping {self.dataset_name} - results already exist at {result_path}")
                return

        random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)

        openai_key = self._resolve_openai_key()
        tokenizer = self._load_tokenizer()
        hidden_state_index = self._resolve_hidden_state_index()

        cache_path = None
        if self.artifacts_dir:
            safe_name = self._dataset_id().replace("/", "_")
            cache_dir = os.path.join(self.artifacts_dir, "text_autointerp")
            cache_path = os.path.join(
                cache_dir, f"{safe_name}_{self.total_tokens}_tokens_{self.ctx_len}_ctx.pt"
            )

        tokenized_dataset = load_tokenized_dataset_cached(
            self.evaluate_dataset,
            tokenizer=tokenizer,
            ctx_len=self.ctx_len,
            total_tokens=self.total_tokens,
            text_column=self.text_column,
            cache_path=cache_path,
            device=self.model.device,
        )

        sae = self._load_sae()

        sparsity = get_feature_activation_sparsity_hidden(
            tokenized_dataset,
            self.model,
            sae,
            self.llm_batch_size,
            hidden_state_index,
            tokenizer,
            mask_bos_pad_eos_tokens=True,
        )

        cfg = AutoInterpConfig(
            model_name=self._dataset_id(),
            openai_api_key=openai_key,
            openai_model=self.openai_model,
            n_latents=self.n_latents,
            override_latents=self.override_latents,
            dead_latent_threshold=self.dead_latent_threshold,
            random_seed=self.random_seed,
            dataset_name=self._dataset_id(),
            llm_context_size=self.ctx_len,
            llm_batch_size=self.llm_batch_size,
            buffer=self.buffer,
            no_overlap=self.no_overlap,
            act_threshold_frac=self.act_threshold_frac,
            total_tokens=self.total_tokens,
            scoring=self.scoring,
            max_tokens_in_explanation=self.max_tokens_in_explanation,
            use_demos_in_explanation=self.use_demos_in_explanation,
            n_top_ex_for_generation=self.n_top_ex_for_generation,
            n_iw_sampled_ex_for_generation=self.n_iw_sampled_ex_for_generation,
            n_top_ex_for_scoring=self.n_top_ex_for_scoring,
            n_random_ex_for_scoring=self.n_random_ex_for_scoring,
            n_iw_sampled_ex_for_scoring=self.n_iw_sampled_ex_for_scoring,
        )

        runner = AutoInterpRunner(
            cfg=cfg,
            model=self.model,
            sae=sae,
            tokenized_dataset=tokenized_dataset,
            sparsity=sparsity,
            tokenizer=tokenizer,
            hidden_state_index=hidden_state_index,
        )
        results = asyncio.run(runner.run())

        scores = [r["score"] for r in results.values() if "score" in r]
        scores_tensor = torch.tensor(scores) if scores else torch.tensor([0.0])
        summary = {
            "autointerp_score": scores_tensor.mean().item(),
            "autointerp_std_dev": scores_tensor.std().item(),
            "details": results,
        }

        self._save_result(summary)

    def _dataset_id(self) -> str:
        info = getattr(self.evaluate_dataset, "info", None)
        name = getattr(info, "dataset_name", None)
        return name or (self.dataset_name or "dataset")


@registry.register_evaluator("ImageAutoInterpEvaluator")
class ImageAutoInterpEvaluator(BaseEvaluator):
    """
    Image auto-interpretation evaluator wrapping multimodal-sae sae_auto_interp scorers.
    """

    dataset_name: Optional[str] = "ImageAutoInterp"
    dataset_split: str = "train"

    model_name: str = "llava-hf/llama3-llava-next-8b-hf"
    explanation_dir: Optional[str] = None
    activation_dir: Optional[str] = None
    selected_layer: str = "model.layers.24"
    width: int = 131072
    n_splits: int = 128
    filters_path: Optional[str] = None

    # Pipeline switches
    run_cache: bool = False
    run_explain: bool = False

    # Cache parameters
    sae_path: Optional[str] = None
    cache_save_dir: Optional[str] = None
    cache_batch_size: int = 1
    cache_ctx_len: int = 64
    cache_hf_token: Optional[str] = None

    # Explain parameters
    explain_save_dir: Optional[str] = None
    explainer_model: str = "lmms-lab/llava-onevision-qwen2-72b-ov"
    explainer_tp: int = 8
    explainer_base_url: str = "http://localhost:12345"
    explain_max_examples: int = 5
    explain_selected_layers: Optional[List[int]] = None
    explain_train_type: str = "top"
    explain_n_examples_train: int = 10
    explain_n_quantiles: int = 10

    # Segment scoring
    run_segment: bool = True
    segment_eval_type: str = "default"
    detector: str = "IDEA-Research/grounding-dino-base"
    segmentor: str = "facebook/sam-vit-huge"
    refine_cache: Optional[str] = None
    save_refine_path: Optional[str] = None
    save_segment_score_path: Optional[str] = None

    # CLIP scoring
    run_clip: bool = True
    clip_eval_type: str = "default"
    clip_model_name_or_path: str = "openai/clip-vit-base-patch16"
    clip_k: int = 5
    clip_random_runs: int = 30
    save_clip_score_path: Optional[str] = None
    refine_clip: bool = False

    def _ensure_sae_auto_interp_on_path(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        sae_root = repo_root / "multimodal-sae"
        if str(sae_root) not in sys.path:
            sys.path.append(str(sae_root))

    def _resolve_dataset(self):
        if self.evaluate_dataset is not None:
            return self.evaluate_dataset
        raise ValueError("evaluate_dataset must be provided by the dataset builder.")

    def evaluate(self, batch_size: int = 32):
        if self.output_dir is None:
            raise ValueError("output_dir is required for ImageAutoInterpEvaluator.")
        summary: Dict[str, Any] = {}
        os.makedirs(self.output_dir, exist_ok=True)

        self._ensure_sae_auto_interp_on_path()
        from sae_auto_interp.agents.explainers import ImageExplainer
        from sae_auto_interp.agents.scorers import LabelRefiner, RandomSegmentScorer, SegmentScorer
        from sae_auto_interp.agents.scorers.clip import ClipScorer
        from sae_auto_interp.clients import SRT
        from sae_auto_interp.config import ExperimentConfig, FeatureConfig
        from sae_auto_interp.features import (
            FeatureDataset,
            FeatureImageCache,
            pool_max_activations_windows_image,
            sample,
        )
        from sae_auto_interp.pipeline import Pipeline, process_wrapper
        from sae_auto_interp.utils import load_filter, load_saes, maybe_load_llava_model
        from transformers import AutoProcessor

        tokens = self._resolve_dataset()
        processor = AutoProcessor.from_pretrained(self.model_name)

        filters = None
        if self.filters_path is not None:
            filters = load_filter(self.filters_path)[self.selected_layer].cpu()

        if self.run_cache:
            if not self.sae_path:
                raise ValueError("sae_path is required when run_cache is True.")
            if not self.cache_save_dir:
                raise ValueError("cache_save_dir is required when run_cache is True.")

            if torch.cuda.is_bf16_supported():
                dtype = torch.bfloat16
            else:
                dtype = "auto"

            model, _ = maybe_load_llava_model(
                self.model_name, 0, dtype, self.cache_hf_token
            )
            submodule_dict = load_saes(self.sae_path, filters=filters, device=model.device)

            from transformers import AutoTokenizer

            cache = FeatureImageCache(
                model=model,
                tokenizer=AutoTokenizer.from_pretrained(self.model_name, token=self.cache_hf_token),
                submodule_dict=submodule_dict,
                batch_size=self.cache_batch_size,
                shard_size=0,
                processor=processor,
                filters=filters,
            )
            cache.run(self.cache_ctx_len, tokens)
            cache.save_splits(n_splits=self.n_splits, save_dir=self.cache_save_dir, rank=0)
            cache.concate_safetensors(n_splits=self.n_splits, save_dir=self.cache_save_dir)
            summary["cache_dir"] = self.cache_save_dir

        if self.run_explain:
            if not self.explain_save_dir:
                raise ValueError("explain_save_dir is required when run_explain is True.")
            activation_dir = self.cache_save_dir or self.activation_dir
            if not activation_dir:
                raise ValueError("activation_dir (or cache_save_dir) is required for explain.")

            modules = os.listdir(activation_dir)
            feature_cfg = FeatureConfig(
                width=self.width,
                max_examples=self.explain_max_examples,
                n_splits=self.n_splits,
            )
            experiment_cfg = ExperimentConfig(
                model=self.model_name,
                dataset=self._dataset_id(),
                split=self.dataset_split,
                save_dir=activation_dir,
                explanation_dir=self.explain_save_dir,
                filters_path=self.filters_path,
                selected_layers=self.explain_selected_layers or [],
                train_type=self.explain_train_type,
                n_examples_train=self.explain_n_examples_train,
                n_quantiles=self.explain_n_quantiles,
            )

            if self.filters_path is not None:
                filters = load_filter(self.filters_path, device="cpu")
            else:
                filters = None

            if filters is not None:
                modules = [mod for mod in modules if mod in filters]
            elif self.explain_selected_layers:
                modules = [
                    mod for idx, mod in enumerate(modules) if idx in self.explain_selected_layers
                ]

            dataset = FeatureDataset(
                raw_dir=activation_dir,
                cfg=feature_cfg,
                modules=modules,
                features=filters,
            )

            loader = partial(
                dataset.load,
                constructor=partial(
                    pool_max_activations_windows_image,
                    tokens=tokens,
                    cfg=feature_cfg,
                    processor=processor,
                ),
                sampler=partial(sample, cfg=experiment_cfg),
            )

            os.makedirs(os.path.expanduser(self.explain_save_dir), exist_ok=True)

            client = SRT(
                model=self.explainer_model,
                tp=self.explainer_tp,
                base_url=self.explainer_base_url,
            )

            def explainer_postprocess(result):
                content, reps, result = result
                record = result.record
                images = [train.image for train in record.train]
                masks = [train.mask for train in record.train]
                activated_images = [train.activation_image for train in record.train]
                module_name = result.record.feature.module_name.replace(".", "_")
                image_output_dir = (
                    f"{self.explain_save_dir}/images/{module_name}/{result.record.feature}"
                )
                os.makedirs(image_output_dir, exist_ok=True)
                output_path = f"{self.explain_save_dir}/{module_name}.json"
                if os.path.exists(output_path):
                    output_file = json.load(open(output_path, "r"))
                else:
                    output_file = []

                output_file.append({f"{result.record.feature}": f"{result.explanation}"})

                with open(output_path, "w") as f:
                    json.dump(output_file, f, indent=4, ensure_ascii=False)

                idx = 0
                os.makedirs(f"{image_output_dir}/images", exist_ok=True)
                os.makedirs(f"{image_output_dir}/activated_images", exist_ok=True)
                os.makedirs(f"{image_output_dir}/masks", exist_ok=True)
                for image, activated_image, mask in zip(images, activated_images, masks):
                    image.save(f"{image_output_dir}/images/top_{idx}.png")
                    activated_image.save(
                        f"{image_output_dir}/activated_images/top{idx}_activated.jpg"
                    )
                    mask.save(f"{image_output_dir}/masks/{idx}_mask.jpg")
                    idx += 1

                return result

            explainer_pipe = process_wrapper(
                ImageExplainer(
                    client=client,
                    verbose=True,
                ),
                postprocess=explainer_postprocess,
            )

            pipeline = Pipeline(loader, explainer_pipe)
            asyncio.run(pipeline.run(max_processes=os.cpu_count() // 4))
            client.clean()
            summary["explanation_dir"] = self.explain_save_dir

        if self.run_segment:
            if not self.explanation_dir or not self.activation_dir:
                raise ValueError("explanation_dir and activation_dir are required for segment scoring.")

            scorer_cls = SegmentScorer if self.segment_eval_type == "default" else RandomSegmentScorer
            scorer = scorer_cls(
                explanation_dir=self.explanation_dir,
                activation_dir=self.activation_dir,
                tokens=tokens,
                processor=processor,
                selected_layer=self.selected_layer,
                width=self.width,
                n_splits=self.n_splits,
                detector=self.detector,
                segmentor=self.segmentor,
                device=str(self.model.device),
                filters=filters,
            )

            if self.refine_cache is None and self.save_refine_path is not None:
                client = SRT(model="meta-llama/Llama-3.1-8B-Instruct", tp=2)
                refiner = LabelRefiner(client, scorer.filtered_explanation)
                scorer.refine(refiner, save_path=self.save_refine_path)
                client.clean()
            elif self.refine_cache is not None:
                with open(self.refine_cache, "r") as f:
                    scorer.explanation = json.load(f)

            scorer.load_model()
            segment_scores = scorer()

            segment_path = self.save_segment_score_path or os.path.join(
                self.output_dir, "segment_scores.json"
            )
            with open(segment_path, "w") as f:
                json.dump(segment_scores, f, indent=2)
            summary["segment_scores_path"] = segment_path

        if self.run_clip:
            if not self.explanation_dir:
                raise ValueError("explanation_dir is required for clip scoring.")

            clip_scorer = ClipScorer(
                explanation_dir=self.explanation_dir,
                dataset_path=self._dataset_id(),
                dataset_split=self.dataset_split,
                k=self.clip_k,
                evaluation_type=self.clip_eval_type,
                clip_model_name_or_path=self.clip_model_name_or_path,
                device=str(self.model.device),
                random_runs=self.clip_random_runs,
            )

            if self.refine_clip and self.save_refine_path:
                client = SRT(model="meta-llama/Llama-3.1-8B-Instruct", tp=2)
                refiner = LabelRefiner(client, clip_scorer.explanations)
                clip_scorer.refine(refiner, save_path=self.save_refine_path)
                client.clean()

            clip_scores = clip_scorer.run()
            clip_path = self.save_clip_score_path or os.path.join(
                self.output_dir, "clip_scores.json"
            )
            with open(clip_path, "w") as f:
                json.dump(clip_scores, f, indent=2)
            summary["clip_scores_path"] = clip_path

        self._save_result(summary)

    def _dataset_id(self) -> str:
        info = getattr(self.evaluate_dataset, "info", None)
        name = getattr(info, "dataset_name", None)
        return name or (self.dataset_name or "dataset")


@registry.register_evaluator("L0Evaluator")
class L0Evaluator(BaseEvaluator):
    """
    Compute average L0 (active features per token) using SAE activations.
    """

    dataset_name: Optional[str] = "L0"
    text_column: str = "text"

    ctx_len: int = 128
    total_tokens: int = 2_000_000
    batch_size_tokens: int = 8

    sae_model_cls: str = "TopKSAE"
    sae_path: Optional[str] = None
    sae_dtype: str = "float16"
    sae_device: Optional[str] = None

    hidden_state_layer: int = 24
    hidden_state_index: Optional[int] = None
    tokenizer_name_or_path: Optional[str] = None
    artifacts_dir: Optional[str] = "./artifacts"

    def _resolve_hidden_state_index(self) -> int:
        if self.hidden_state_index is not None:
            return self.hidden_state_index
        return self.hidden_state_layer + 1

    def _load_tokenizer(self):
        from transformers import AutoTokenizer

        name_or_path = self.tokenizer_name_or_path
        if name_or_path is None:
            name_or_path = getattr(self.model.config, "_name_or_path", None) or getattr(
                self.model, "name_or_path", None
            )
        if name_or_path is None:
            raise ValueError("tokenizer_name_or_path is required when it cannot be inferred.")
        tokenizer = AutoTokenizer.from_pretrained(name_or_path, use_fast=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def _load_sae(self):
        from src import models  # noqa: F401
        from src.common.registry import registry as _registry

        if not self.sae_path:
            raise ValueError("sae_path is required for L0Evaluator.")
        sae_cls = _registry.get_model_class(self.sae_model_cls)
        if sae_cls is None:
            raise ValueError(f"Unknown SAE model class: {self.sae_model_cls}")
        dtype = getattr(torch, self.sae_dtype, None)
        if dtype is None:
            raise ValueError(f"Unsupported sae_dtype: {self.sae_dtype}")
        sae = sae_cls.from_pretrained(self.sae_path, torch_dtype=dtype)
        device = self.sae_device or str(self.model.device)
        return sae.to(device=device).eval()

    def evaluate(self, batch_size: int = 32):
        if self.output_dir is None:
            raise ValueError("output_dir is required for L0Evaluator.")
        result_path = os.path.join(self.output_dir, f"{self.dataset_name.upper()}.json")
        if not self.overwrite_results and os.path.exists(result_path):
            logger.info(f"Skipping {self.dataset_name} - results already exist at {result_path}")
            return

        tokenizer = self._load_tokenizer()
        hidden_state_index = self._resolve_hidden_state_index()

        cache_path = None
        if self.artifacts_dir:
            safe_name = self._dataset_id().replace("/", "_")
            cache_dir = os.path.join(self.artifacts_dir, "l0")
            cache_path = os.path.join(
                cache_dir, f"{safe_name}_{self.total_tokens}_tokens_{self.ctx_len}_ctx.pt"
            )

        tokens = load_tokenized_text_dataset(
            self.evaluate_dataset,
            tokenizer=tokenizer,
            ctx_len=self.ctx_len,
            total_tokens=self.total_tokens,
            text_column=self.text_column,
            cache_path=cache_path,
            device=self.model.device,
        )
        sae = self._load_sae()

        l0_value = compute_l0(
            tokens,
            self.model,
            sae,
            tokenizer,
            batch_size=self.batch_size_tokens,
            hidden_state_index=hidden_state_index,
        )
        self._save_result({"l0": l0_value})

    def _dataset_id(self) -> str:
        info = getattr(self.evaluate_dataset, "info", None)
        name = getattr(info, "dataset_name", None)
        return name or (self.dataset_name or "dataset")


@registry.register_evaluator("LossRecoveredEvaluator")
class LossRecoveredEvaluator(BaseEvaluator):
    """
    Compute CE loss recovered using SAE reconstruction vs ablation.
    """

    dataset_name: Optional[str] = "LossRecovered"
    text_column: str = "text"

    ctx_len: int = 128
    total_tokens: int = 2_000_000
    batch_size_tokens: int = 4

    sae_model_cls: str = "TopKSAE"
    sae_path: Optional[str] = None
    sae_dtype: str = "float16"
    sae_device: Optional[str] = None

    hidden_state_layer: int = 24
    hidden_state_index: Optional[int] = None
    hook_module_path: Optional[str] = None
    tokenizer_name_or_path: Optional[str] = None
    artifacts_dir: Optional[str] = "./artifacts"
    exclude_special_tokens: bool = True

    def _resolve_hidden_state_index(self) -> int:
        if self.hidden_state_index is not None:
            return self.hidden_state_index
        return self.hidden_state_layer + 1

    def _infer_hook_module_path(self) -> str:
        if self.hook_module_path:
            return self.hook_module_path
        if hasattr(self.model, "language_model"):
            return f"language_model.model.layers.{self.hidden_state_layer}"
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return f"model.layers.{self.hidden_state_layer}"
        raise ValueError("hook_module_path is required for this model.")

    def _load_tokenizer(self):
        from transformers import AutoTokenizer

        name_or_path = self.tokenizer_name_or_path
        if name_or_path is None:
            name_or_path = getattr(self.model.config, "_name_or_path", None) or getattr(
                self.model, "name_or_path", None
            )
        if name_or_path is None:
            raise ValueError("tokenizer_name_or_path is required when it cannot be inferred.")
        tokenizer = AutoTokenizer.from_pretrained(name_or_path, use_fast=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def _load_sae(self):
        from src import models  # noqa: F401
        from src.common.registry import registry as _registry

        if not self.sae_path:
            raise ValueError("sae_path is required for LossRecoveredEvaluator.")
        sae_cls = _registry.get_model_class(self.sae_model_cls)
        if sae_cls is None:
            raise ValueError(f"Unknown SAE model class: {self.sae_model_cls}")
        dtype = getattr(torch, self.sae_dtype, None)
        if dtype is None:
            raise ValueError(f"Unsupported sae_dtype: {self.sae_dtype}")
        sae = sae_cls.from_pretrained(self.sae_path, torch_dtype=dtype)
        device = self.sae_device or str(self.model.device)
        return sae.to(device=device).eval()

    def evaluate(self, batch_size: int = 32):
        if self.output_dir is None:
            raise ValueError("output_dir is required for LossRecoveredEvaluator.")
        result_path = os.path.join(self.output_dir, f"{self.dataset_name.upper()}.json")
        if not self.overwrite_results and os.path.exists(result_path):
            logger.info(f"Skipping {self.dataset_name} - results already exist at {result_path}")
            return

        tokenizer = self._load_tokenizer()
        hidden_state_index = self._resolve_hidden_state_index()
        hook_module_path = self._infer_hook_module_path()

        cache_path = None
        if self.artifacts_dir:
            safe_name = self._dataset_id().replace("/", "_")
            cache_dir = os.path.join(self.artifacts_dir, "loss_recovered")
            cache_path = os.path.join(
                cache_dir, f"{safe_name}_{self.total_tokens}_tokens_{self.ctx_len}_ctx.pt"
            )

        tokens = load_tokenized_text_dataset(
            self.evaluate_dataset,
            tokenizer=tokenizer,
            ctx_len=self.ctx_len,
            total_tokens=self.total_tokens,
            text_column=self.text_column,
            cache_path=cache_path,
            device=self.model.device,
        )
        sae = self._load_sae()

        metrics = compute_loss_recovered(
            tokens=tokens,
            model=self.model,
            sae=sae,
            tokenizer=tokenizer,
            batch_size=self.batch_size_tokens,
            hidden_state_index=hidden_state_index,
            hook_module_path=hook_module_path,
            exclude_special_tokens=self.exclude_special_tokens,
        )
        self._save_result(metrics)

    def _dataset_id(self) -> str:
        info = getattr(self.evaluate_dataset, "info", None)
        name = getattr(info, "dataset_name", None)
        return name or (self.dataset_name or "dataset")
