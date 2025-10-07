import asyncio
from dataclasses import dataclass
from typing import Dict, List, Optional

import PIL
import aiohttp
from PIL import Image
from tqdm.asyncio import tqdm_asyncio
from transformers import BatchEncoding
from transformers.utils import add_end_docstrings, logging
from trl.trainer.dpo_trainer import DataCollatorForPreference

from src.common.registry import registry
from src.utils import is_url
from .base import BASE_COLLATOR_DOCSTRING, BaseCollator

logger = logging.get_logger(__name__)

__all__ = [
    "ImageCollator",
    "ImageURLCollator",
    "RLHFVImageForDPOCollator",
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
            if not isinstance(value, PIL.Image.Image):
                raise ValueError(f"'images' must be PIL.Image.Image, got {type(value).__name__}.")

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


@registry.register_collator('RLHFVImageForDPOCollator')
class RLHFVImageForDPOCollator(DataCollatorForPreference, ImageCollator):

    def _process_example(self, example: Dict) -> BatchEncoding:
        image = example.get("images")
        data = example.get("text")
        question = data.get("question")
        chosen = data.get("chosen")
        rejected = data.get("rejected")

        conversation = [
            {

                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question},
                ],
            },
        ]

        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)

        prompt = self.processor(
            images=[image],
            text=prompt,
            return_tensors=self.return_tensors,
            padding=self.padding,
            max_length=self.max_length,
            truncation=self.truncation
        )
        chosen = self.processor(
            text=chosen,
            return_tensors=self.return_tensors,
            padding=self.padding,
            max_length=self.max_length,
            truncation=self.truncation,
            add_special_tokens=False
        )
        rejected = self.processor(
            text=rejected,
            return_tensors=self.return_tensors,
            padding=self.padding,
            max_length=self.max_length,
            truncation=self.truncation,
            add_special_tokens=False
        )

        inputs = {
            "prompt_input_ids": prompt.input_ids[0],
            "prompt_attention_mask": prompt.attention_mask[0],
            "pixel_values": prompt.pixel_values[0],
            "chosen_input_ids": chosen.input_ids[0],
            "rejected_input_ids": rejected.input_ids[0],
        }
        return BatchEncoding(inputs)

    def __call__(self, features: List[Dict], return_tensors: Optional[str] = None):
        features = self._convert_images(features)
        features = list(map(self._process_example, features))

        return super().__call__(features, return_tensors=return_tensors)
