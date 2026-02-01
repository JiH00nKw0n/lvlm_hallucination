import asyncio
from dataclasses import dataclass
from typing import Dict, List, Optional

import PIL
import aiohttp
import torch
from PIL import Image
from tqdm.asyncio import tqdm_asyncio
from transformers.utils import add_end_docstrings, logging
from transformers import LlavaNextForConditionalGeneration
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
    "LlavaNextSAECollator",
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


@add_end_docstrings(BASE_COLLATOR_DOCSTRING)
@dataclass
@registry.register_collator("LlavaNextSAECollator")
class LlavaNextSAECollator(BaseCollator):
    model_name_or_path: str = ""
    layer_index: int = 24
    attn_implementation: Optional[str] = "flash_attention_3"
    torch_dtype: Optional[str] = "bfloat16"
    device: Optional[str] = None

    def __post_init__(self):
        super().__post_init__()
        if not self.model_name_or_path:
            raise ValueError("model_name_or_path must be provided for LlavaNextSAECollator.")
        dtype = getattr(torch, self.torch_dtype) if isinstance(self.torch_dtype, str) else self.torch_dtype
        model = LlavaNextForConditionalGeneration.from_pretrained(
            self.model_name_or_path,
            torch_dtype=dtype,
            attn_implementation=self.attn_implementation,
        )
        device = (
            torch.device(self.device)
            if self.device is not None
            else torch.device(
                f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"
            )
        )
        model.to(device)
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        self.model = model
        self.device = device

    def _build_prompt(self, conversations: List[Dict]) -> str:
        messages = []
        for turn in conversations:
            role = turn.get("from", turn.get("role", "user"))
            if role == "human":
                role = "user"
            elif role == "gpt":
                role = "assistant"
            content = turn.get("value", turn.get("content", ""))
            messages.append({"role": role, "content": content})
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

    def __call__(self, features: List[Dict], return_tensors: Optional[str] = None):
        texts = []
        images = []
        for example in features:
            conversations = example.get("conversations")
            if conversations is None:
                raise ValueError("Expected 'conversations' field in dataset examples.")
            texts.append(self._build_prompt(conversations))
            image = example.get("image", example.get("images"))
            if isinstance(image, list):
                image = [img.convert(self.color_model) for img in image]
            elif isinstance(image, PIL.Image.Image):
                image = image.convert(self.color_model)
            images.append(image)

        inputs = self.processor(
            images=images,
            text=texts,
            padding=self.padding,
            truncation=self.truncation,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors or self.return_tensors,
            return_mm_token_type_ids=True,
        )

        with torch.no_grad():
            outputs = self.model(
                input_ids=inputs["input_ids"].to(self.device),
                attention_mask=inputs.get("attention_mask", None).to(self.device)
                if inputs.get("attention_mask", None) is not None
                else None,
                pixel_values=inputs.get("pixel_values", None).to(self.device)
                if inputs.get("pixel_values", None) is not None
                else None,
                image_sizes=inputs.get("image_sizes", None),
                output_hidden_states=True,
                use_cache=False,
            )

        hidden_states = outputs.hidden_states[self.layer_index]
        attention_mask = inputs.get("attention_mask", None)
        visual_mask = inputs.get("mm_token_type_ids", None)
        if visual_mask is None:
            raise ValueError("mm_token_type_ids missing from processor output.")
        visual_mask = visual_mask.bool()

        return {
            "hidden_states": hidden_states,
            "attention_mask": attention_mask,
            "visual_mask": visual_mask,
        }
