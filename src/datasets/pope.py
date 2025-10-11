from typing import Optional

from datasets import load_dataset, Dataset

from src.common import registry
from src.datasets.base import BaseBuilder


@registry.register_builder('POPEDatasetBuilder')
class POPEDatasetBuilder(BaseBuilder):
    """
    Dataset builder for the POPE (Polling-based Object Probing Evaluation) benchmark.

    POPE evaluates object hallucination in Large Vision-Language Models (LVLMs) through
    yes/no questions about object presence in images.

    The dataset contains three different sampling strategies in the 'category' column:
    - random: Random negative object sampling
    - popular: Popular objects as negative samples
    - adversarial: Adversarial negative sampling (most challenging)

    Dataset structure:
        - id: Unique identifier (string)
        - question_id: Question identifier (string)
        - question: Yes/no question about object presence (string)
        - answer: Ground truth answer ("yes" or "no")
        - image_source: Source of the image (e.g., "coco_val2014_000000000042.jpg")
        - image: PIL Image object
        - category: Sampling strategy ("random", "popular", or "adversarial")

    Note: All 9,000 samples (3,000 per category) are loaded. Category-based filtering
    should be performed at the evaluator level.

    Args:
        split (Optional[str]): Dataset split to load. Defaults to "test".

    Example:
        >>> builder = POPEDatasetBuilder()
        >>> dataset = builder.build_dataset()
        >>> print(len(dataset))  # 9000
        >>> print(dataset[0])
        {
            'id': '...',
            'question_id': '...',
            'question': 'Is there a dog in the image?',
            'answer': 'yes',
            'image_source': 'coco_val2014_...',
            'image': <PIL.Image>,
            'category': 'adversarial'
        }
    """
    split: Optional[str] = 'test'

    def build_dataset(self) -> Dataset:
        """
        Builds and returns the POPE dataset from HuggingFace.

        Returns:
            Dataset: A HuggingFace Dataset object containing the POPE evaluation data.
        """
        # Load the default subset's test split
        dataset = load_dataset(
            path="lmms-lab/POPE",
            split=self.split
        )

        # Ensure answer is lowercase for consistency
        dataset = dataset.map(
            lambda example: {'answer': example['answer'].lower()},
            desc="Normalizing answers to lowercase"
        )

        return dataset
