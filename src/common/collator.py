import asyncio
from dataclasses import dataclass
from typing import Dict, List, Optional

import PIL
import aiohttp
from PIL import Image
from tqdm.asyncio import tqdm_asyncio
from transformers.utils import add_end_docstrings, logging
from trl.trainer.dpo_trainer import DataCollatorForPreference
from trl.trainer.sft_trainer import DataCollatorForVisionLanguageModeling

from src.common.registry import registry
from src.utils import is_url
from .base import BASE_COLLATOR_DOCSTRING, BaseCollator

logger = logging.get_logger(__name__)

__all__ = [
    "ImageCollator",
    "ImageURLCollator",
    "RLHFVForDPOImageCollator",
    "DummyImageCollator",
]


@add_end_docstrings(BASE_COLLATOR_DOCSTRING)
@dataclass
@registry.register_collator('ImageCollator')
class ImageCollator(BaseCollator):
    """
    A collator class for processing dictionaries containing image and text data. The 'images' key in the input
    dictionaries must hold `PIL.Image` objects, which are converted to RGB format, and the 'text' key must hold
    strings. This class handles dynamic padding, truncation, and tensor conversion before passing the processed
    data to the processor.

    Raises:
        TypeError:
            - If the 'images' key contains objects that are not `PIL.Image` instances.
            - If the 'text' key contains values that are not `str` instances.
        ValueError:
            - If the input dictionaries contain keys other than 'images' and 'text'.
    """

    def _convert_images(self, examples: List[Dict]) -> List[Dict]:
        """
        Convert images in each example to the specified color model.

        Args:
            examples: List of dictionaries containing 'images' key with PIL.Image values.

        Returns:
            List of dictionaries with 'images' converted to the specified color model (e.g., RGB).

        Raises:
            ValueError: If 'images' key is missing or contains non-PIL.Image objects.
        """

        def _convert_single_image(example: Dict) -> Dict:
            value = example.get("images")

            if value is None:
                raise ValueError("'images' key is missing.")

            if isinstance(value, list):
                # Validate each element in the list is a PIL Image
                for idx, img in enumerate(value):
                    if not isinstance(img, PIL.Image.Image):
                        raise TypeError(
                            f"'images' list element at index {idx} must be PIL.Image.Image, "
                            f"got {type(img).__name__}."
                        )
                return {**example, "images": [v.convert(self.color_model) for v in value]}

            if not isinstance(value, PIL.Image.Image):
                raise TypeError(
                    f"'images' must be PIL.Image.Image or list of PIL.Image.Image, "
                    f"got {type(value).__name__}."
                )

            return {**example, "images": value.convert(self.color_model)}

        return list(map(_convert_single_image, examples))


@add_end_docstrings(BASE_COLLATOR_DOCSTRING)
@dataclass
@registry.register_collator('ImageURLCollator')
class ImageURLCollator(BaseCollator):
    """
    Collator for processing batches of image URLs and text data.

    Fetches images from URLs asynchronously (with semaphore limit of 32) and converts them
    to the specified color model (default RGB).

    Raises:
        ValueError: If 'images' contains invalid URLs or 'text' contains non-string values.
    """

    async def _fetch_single_image(
            self,
            session: aiohttp.ClientSession,
            url: str,
            semaphore: asyncio.Semaphore
    ) -> Optional[PIL.Image.Image]:
        """Fetch a single image from URL with semaphore control."""
        async with semaphore:
            try:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as response:
                    if response.status == 200:
                        image_bytes = await response.read()
                        image = Image.open(PIL.Image.BytesIO(image_bytes))
                        return image.convert(self.color_model)
            except Exception as e:
                logger.warning(f"Failed to fetch image from {url}: {e}")
        return None

    async def _fetch_images_batch(self, urls: List[str]) -> List[Optional[PIL.Image.Image]]:
        """Fetch multiple images asynchronously with semaphore limit of 32 and progress bar."""
        semaphore = asyncio.Semaphore(32)
        async with aiohttp.ClientSession() as session:
            tasks = [self._fetch_single_image(session, url, semaphore) for url in urls]
            return await tqdm_asyncio.gather(*tasks, desc="Fetching images")

    def _convert_images(self, examples: List[Dict]) -> List[Dict]:
        """
        Convert image URLs in examples to RGB PIL.Image objects.

        Args:
            examples: List of dictionaries containing 'images' (URL) and 'text' keys.

        Returns:
            List of dictionaries with 'images' converted to PIL.Image (valid entries only).
        """

        def _validate_and_extract(example: Dict) -> Optional[tuple[str, str]]:
            url = example.get("images")
            text = example.get("text")

            if url is None:
                raise ValueError("'images' key is missing.")
            if not is_url(url):
                raise ValueError(f"'images' must be a valid URL, got {url}.")
            if not isinstance(text, str):
                raise ValueError(f"'text' must be str, got {type(text).__name__}.")

            return (url, text)

        valid_pairs = list(map(_validate_and_extract, examples))
        urls, texts = zip(*valid_pairs)

        # Fetch images asynchronously
        images = asyncio.run(self._fetch_images_batch(list(urls)))

        # Filter successful fetches
        return [
            {"images": img, "text": txt}
            for img, txt in zip(images, texts)
            if img is not None
        ]


@registry.register_collator('RLHFVForDPOImageCollator')
class RLHFVForDPOImageCollator(ImageCollator, DataCollatorForPreference):
    pass


@registry.register_collator('LRVInstructForSFTImageCollator')
class LRVInstructForSFTImageCollator(ImageCollator, DataCollatorForVisionLanguageModeling):
    completion_only_loss: bool = True
    image_field: str = "images"
    prompt_field: str = "prompt"
    completion_field: str = "completion"
    prompt_role: str = "user"
    completion_role: str = "assistant"

    def _process_example(self, example: Dict) -> Dict:
        image = example.get(self.image_field)
        prompt = example.get(self.prompt_field)
        completion = example.get(self.completion_field)

        processed_input = {
            "images": [image],
            "prompt": [{"role": self.prompt_role, "content": prompt}],
            "completion": [{"role": self.completion_role, "content": completion}],
        }
        return processed_input

    def __call__(self, features: List[Dict], return_tensors: Optional[str] = None):
        features = self._convert_images(features)
        features = list(map(self._process_example, features))

        return super().__call__(features, return_tensors=return_tensors)


@add_end_docstrings(BASE_COLLATOR_DOCSTRING)
@dataclass
@registry.register_collator('DummyImageCollator')
class DummyImageCollator(BaseCollator):
    """
    A dummy collator that accepts a processor but performs no operations.

    This collator is used for evaluators that handle their own batch processing
    (e.g., LVLMEvaluator) and don't require data collation. It exists to satisfy
    the collator requirement in the evaluation task structure while allowing
    the evaluator to access the processor directly.

    The collator simply returns the input features as-is without any processing.
    """

    def __call__(self, features: List[Dict], return_tensors: Optional[str] = None):
        """
        Returns the input features without any processing.

        Args:
            features: List of feature dictionaries.
            return_tensors: Ignored (for API compatibility).

        Returns:
            The input features unchanged.
        """
        return features
