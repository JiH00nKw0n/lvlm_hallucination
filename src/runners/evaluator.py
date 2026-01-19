import logging
from typing import Callable, Dict, List, Optional

import torch
from accelerate import PartialState
from accelerate.utils import gather_object
from datasets import tqdm
from pydantic import Field
from torch.utils.data import DataLoader
from trl.data_utils import apply_chat_template

from src.common.registry import registry
from src.runners.base import BaseEvaluator

logger = logging.getLogger(__name__)

__all__ = ["LVLMEvaluator", "POPEEvaluator", "MMEEvaluator"]


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
        if mitigator_cls is None:
            available = ", ".join(registry.list_mitigators())
            raise ValueError(f"Unknown mitigator: {name}. Available: {available}")

        model_type_raw = self.decoding_config.get("model_type", "llava-hf/llava-1.5-7b-hf")
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
        
        indexed_dataset = [
            {"_sample_idx": idx, **sample}
            for idx, sample in enumerate(self.evaluate_dataset)
        ]

        # Split indexed dataset across GPUs with padding
        with self.distributed_state.split_between_processes(
                indexed_dataset,
                apply_padding=True  # Pad to make lengths equal for gathering
        ) as process_dataset:
            results = []
            mitigator = self._build_mitigator()

            # Create dataloader for batch processing
            dataloader = DataLoader(
                process_dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=default_collate_fn
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

        Looks for "yes" or "no" in the first few characters of the generated text.

        Args:
            generated_text: Raw generated text from model
            sample: Original sample (unused but required by interface)

        Returns:
            "yes", "no", or "other" if neither is found
        """
        text_lower = generated_text.strip().lower()

        # Check if answer is exactly "yes" or "no"
        if text_lower in ["yes", "no"]:
            return text_lower

        # Check first 4 characters for yes/no
        prefix = text_lower[:4]
        if "yes" in prefix:
            return "yes"
        elif "no" in prefix:
            return "no"
        else:
            return "other"

    def _compute_metrics(self, results: List[Dict]) -> dict:
        """
        Compute POPE evaluation metrics.

        Metrics include overall statistics and per-category breakdown
        (random, popular, adversarial).

        Args:
            results: List of result dicts with 'parsed_answer', 'answer', 'category'

        Returns:
            Dictionary with overall and per-category metrics
        """
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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

            # Extract labels and predictions
            y_true = [r["answer"] for r in category_data]
            y_pred = [r["parsed_answer"] for r in category_data]

            # Convert to binary (yes=1, no=0), filter out "other"
            label_map = {"yes": 1, "no": 0}

            valid_indices = [
                i for i, pred in enumerate(y_pred)
                if pred in label_map
            ]

            y_true_binary = [label_map[y_true[i]] for i in valid_indices]
            y_pred_binary = [label_map[y_pred[i]] for i in valid_indices]

            # Count "other" responses
            other_count = len(y_pred) - len(valid_indices)

            # Compute metrics
            if len(y_true_binary) > 0:
                acc = accuracy_score(y_true_binary, y_pred_binary)
                prec = precision_score(y_true_binary, y_pred_binary, zero_division=0)
                rec = recall_score(y_true_binary, y_pred_binary, zero_division=0)
                f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)
                yes_ratio = sum(y_pred_binary) / len(y_pred_binary)

                metrics[category_name] = {
                    "accuracy": float(acc),
                    "precision": float(prec),
                    "recall": float(rec),
                    "f1": float(f1),
                    "yes_ratio": float(yes_ratio),
                    "other_count": other_count,
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
