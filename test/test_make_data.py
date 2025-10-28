"""
Script to load and iterate through Mayfull/Long-DCI-QA dataset examples.
"""

import asyncio
import base64
import io
import os

from datasets import load_dataset
from dotenv import load_dotenv
from openai import AsyncOpenAI
from PIL import Image
from pydantic import BaseModel
from tqdm import tqdm
from tqdm.asyncio import tqdm as tqdm_asyncio

load_dotenv()
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM_PROMPT = """
You are a helpful assistant that generates relevant questions based on the given context.
Generate a short and clear question within 10 words that briefly mention a specific element in the provided information and asks for a detailed description of it.
Follow the given examples.

### Examples
Examine this painting, detailing its composition, color scheme, and the artist's technique.

Analyze this art image, describing its spatial arrangement, interactive elements, and conceptual message.

Provide an in-depth description of the image, centering on the text and its context.

How do the elements in the image relate to each other in terms of positioning or composition?
###
""".strip()


class QuestionOutput(BaseModel):
    question: str


def encode_image(image: Image.Image) -> str:
    """Convert PIL Image to base64 string."""
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


async def generate_question(
        image: Image.Image,
        model_name: str = "gpt-5-2025-08-07",
        reasoning_effort: str = "minimal",
) -> str:
    """
    Generate a question based on the given image using GPT-4o with minimal reasoning.

    Args:
        image: PIL Image to generate a question from
        model_name: The GPT model to use (default: gpt-4o)
        reasoning_effort: Reasoning effort level (default: minimal)

    Returns:
        Generated question as a string
    """
    # Convert image to base64
    base64_image = encode_image(image)

    response = await client.beta.chat.completions.parse(
        model=model_name,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Generate a short, clear question within 10 words that briefly mention a specific element and asks for detailed descriptions of it."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            },
        ],
        response_format=QuestionOutput,
        reasoning_effort=reasoning_effort,
        seed=2025,
    )

    return response.choices[0].message.parsed.question


async def main():
    # Load the Long-DCI-QA dataset
    print("Loading Mayfull/Long-DCI-QA dataset...")
    dataset = load_dataset("Mayfull/Long-DCI-QA")

    # Use test split
    split = dataset['test']
    print(f"\n{'=' * 80}")
    print(f"Processing test split")
    print(f"{'=' * 80}")
    print(f"Number of examples: {len(split)}")
    print(f"Columns: {split.column_names}\n")

    # Show first example structure
    print("First example structure:")
    print("-" * 80)
    example = split[0]
    for key, value in example.items():
        print(f"\n[{key}]")
        if isinstance(value, str) and len(value) > 200:
            print(f"{value[:200]}... (truncated)")
        else:
            print(value)
    print("\n" + "=" * 80)

    # Generate questions for all examples
    print(f"\nGenerating questions for all {len(split)} examples...")

    # Create all tasks
    tasks = []
    for example in split:
        image = example['image']
        tasks.append(generate_question(image))

    # Run all tasks with progress bar
    all_questions = await tqdm_asyncio.gather(*tasks, desc="Generating questions")

    # Add question column to dataset
    split = split.add_column("question", all_questions)

    # Print all generated questions
    print(f"\n{'=' * 80}")
    print(f"All Generated Questions ({len(all_questions)} total):")
    print(f"{'=' * 80}")
    for idx, question in enumerate(all_questions):
        print(f"{idx}: {question}")

    # Save dataset with questions
    output_dir = "../dataset_question"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "test")

    print(f"\n{'=' * 80}")
    print(f"Saving dataset to {output_path}...")
    split.save_to_disk(output_path)
    print(f"Successfully saved test split with {len(split)} examples")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    asyncio.run(main())
