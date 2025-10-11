"""
Test CustomDPOTrainer's _prepare_dataset with RLHF-V dataset
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from transformers import AutoProcessor
from trl import DPOConfig

from src.datasets.rlhf_v import RLHFVDatasetBuilder

# Import CustomDPOTrainer directly to avoid __init__.py issues
import importlib.util
spec = importlib.util.spec_from_file_location("trainer", project_root / "src" / "runners" / "trainer.py")
trainer_module = importlib.util.module_from_spec(spec)
sys.modules["trainer"] = trainer_module
spec.loader.exec_module(trainer_module)
CustomDPOTrainer = trainer_module.CustomDPOTrainer


def test_prepare_dataset():
    print("="*80)
    print("Testing CustomDPOTrainer._prepare_dataset with RLHF-V dataset")
    print("="*80)

    # Load dataset
    print("\n1. Loading RLHF-V dataset...")
    builder = RLHFVDatasetBuilder(split="train[:5]")
    dataset = builder.build_dataset()

    print(f"   Dataset size: {len(dataset)}")
    print(f"   Dataset columns: {dataset.column_names}")
    print(f"   Dataset features: {dataset.features}")

    # Show first example before processing
    print("\n2. First example (before processing):")
    example = dataset[0]
    print(f"   Images: {type(example['images'])}, length: {len(example['images'])}")
    print(f"   Prompt: {example['prompt']}")
    print(f"   Chosen: {example['chosen']}")
    print(f"   Rejected: {example['rejected']}")

    # Load processor
    print("\n3. Loading processor...")
    model_name = "llava-hf/llava-1.5-7b-hf"
    processor = AutoProcessor.from_pretrained(model_name)
    print(f"   Loaded processor: {type(processor).__name__}")

    # Create DPO config
    print("\n4. Creating DPOConfig...")
    args = DPOConfig(
        output_dir="./test_output",
        dataset_num_proc=1,
        max_prompt_length=2048,
        max_completion_length=2048,
    )
    print(f"   Config created")

    # Create trainer instance (minimal setup for testing _prepare_dataset)
    print("\n5. Creating CustomDPOTrainer instance (mocked)...")
    try:
        from unittest.mock import Mock
        from trl.trainer.dpo_trainer import DPOTrainer

        # Create a mock trainer with necessary attributes
        trainer = CustomDPOTrainer.__new__(CustomDPOTrainer)
        trainer.processor = processor
        trainer.is_vision_model = True
        trainer.processing_class = processor

        # Use the real process_row method from DPOTrainer parent class
        # process_row is a static method, so we can use it directly
        trainer.process_row = DPOTrainer.process_row
        trainer.tokenize_row = DPOTrainer.tokenize_row

        print("   Trainer instance created (using real DPOTrainer static methods)")

        # Call _prepare_dataset
        print("\n6. Calling _prepare_dataset...")
        processed_dataset = trainer._prepare_dataset(
            dataset=dataset,
            processing_class=processor,
            args=args,
            dataset_name="train"
        )

        print(f"   Processed dataset size: {len(processed_dataset)}")
        print(f"   Processed dataset columns: {processed_dataset.column_names}")

        # Show first example after processing
        print("\n7. First example (after tokenization):")
        processed_example = processed_dataset[0]
        for key, value in processed_example.items():
            if key in ["prompt_input_ids", "chosen_input_ids", "rejected_input_ids"]:
                print(f"   {key}: shape={len(value) if isinstance(value, list) else 'N/A'}")
                if isinstance(value, list) and len(value) > 0:
                    # Decode first few tokens
                    tokens_to_show = min(2000, len(value))
                    decoded = processor.tokenizer.decode(value[:tokens_to_show])
                    print(f"      First {tokens_to_show} tokens decoded: {decoded}")
            elif isinstance(value, list) and len(value) > 0:
                print(f"   {key}: {type(value)}, length: {len(value)}")
                if isinstance(value[0], dict):
                    print(f"      First item: {value[0]}")
                elif key == "pixel_values":
                    import torch
                    if isinstance(value, list):
                        print(f"      Shape: list of {len(value)} items")
                    elif isinstance(value, torch.Tensor):
                        print(f"      Shape: {value.shape}")
            else:
                if key == "pixel_values":
                    import torch
                    if isinstance(value, torch.Tensor):
                        print(f"   {key}: Tensor shape={value.shape}")
                    else:
                        print(f"   {key}: {type(value)}")
                else:
                    print(f"   {key}: {value}")

        print("\n" + "="*80)
        print("Test completed successfully!")
        print("="*80)

    except Exception as e:
        print(f"\n   Error during processing: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    test_prepare_dataset()
