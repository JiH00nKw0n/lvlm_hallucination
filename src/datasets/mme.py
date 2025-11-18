from typing import Optional

from datasets import load_dataset, Dataset

from src.common import registry
from src.datasets.base import BaseBuilder


# MME task categories organized by evaluation type
EVAL_TYPE_DICT = {
    "Perception": [
        "existence", "count", "position", "color", "posters",
        "celebrity", "scene", "landmark", "artwork", "OCR"
    ],
    "Cognition": [
        "commonsense_reasoning", "numerical_calculation",
        "text_translation", "code_reasoning"
    ]
}


@registry.register_builder('MMEDatasetBuilder')
class MMEDatasetBuilder(BaseBuilder):
    """
    Dataset builder for the MME (Multimodal Evaluation) benchmark.

    MME is a comprehensive evaluation benchmark for multimodal large language models,
    covering both perception and cognition abilities through yes/no questions.

    The dataset evaluates models across multiple task categories:

    Perception tasks:
        - existence: Object existence detection
        - count: Object counting
        - position: Spatial position understanding
        - color: Color recognition
        - posters: Poster/text understanding
        - celebrity: Celebrity recognition
        - scene: Scene recognition
        - landmark: Landmark identification
        - artwork: Artwork recognition
        - OCR: Optical character recognition

    Cognition tasks:
        - commonsense_reasoning: Common sense reasoning
        - numerical_calculation: Numerical computation
        - text_translation: Text translation
        - code_reasoning: Code understanding

    Dataset structure:
        - question_id: Question identifier (string)
        - image: PIL Image object
        - question: Yes/no question (string)
        - answer: Ground truth answer ("Yes" or "No")
        - category: Task category (string)

    Note: All samples across all categories are loaded. Category-based and task type
    filtering should be performed at the evaluator level using the EVAL_TYPE_DICT
    constant defined in this module.

    Args:
        split (Optional[str]): Dataset split to load. Defaults to "test".

    Example:
        >>> builder = MMEDatasetBuilder()
        >>> dataset = builder.build_dataset()
        >>> print(dataset[0])
        {
            'question_id': 'artwork/10002',
            'image': <PIL.Image>,
            'question': 'Does this artwork exist in the form of painting? Please answer yes or no.',
            'answer': 'yes',
            'category': 'artwork'
        }
    """
    split: Optional[str] = 'test'

    def build_dataset(self) -> Dataset:
        """
        Builds and returns the MME dataset from HuggingFace.

        Returns:
            Dataset: A HuggingFace Dataset object containing the MME evaluation data.
        """
        # Load the dataset
        dataset = load_dataset(
            path="darkyarding/MME",
            split=self.split
        )

        # Normalize answer format (Yes/No -> yes/no)
        dataset = dataset.map(
            lambda example: {'answer': example['answer'].lower()},
            desc="Normalizing answers to lowercase"
        )

        return dataset
