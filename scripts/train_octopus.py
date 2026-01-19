import argparse
import json
import os
from typing import Any, Dict, Optional

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoProcessor

from src.decoding import OctopusMitigator
from src.decoding.base import ModelHelper


class OctopusTrainDataset(Dataset):
    def __init__(self, json_path: str, image_root: str):
        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.image_root = image_root

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.data[idx]
        image_path = os.path.join(self.image_root, item["image"])
        image = Image.open(image_path).convert("RGB")
        return {
            "image": image,
            "query": item["query"],
            "y_w": torch.tensor(item["y_w"], dtype=torch.long),
            "y_l": torch.tensor(item["y_l"], dtype=torch.long),
        }


def build_prompt(processor: Any, query: str) -> str:
    if hasattr(processor, "apply_chat_template"):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": query},
                ],
            }
        ]
        return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if hasattr(processor, "tokenizer") and hasattr(processor.tokenizer, "apply_chat_template"):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": query},
                ],
            }
        ]
        return processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return query


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Octopus classifier training (batch=1).")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-type", default="llava")
    parser.add_argument("--train-json", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
    )
    model.to(args.device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    mitigator = OctopusMitigator(model, model_type=args.model_type)
    mitigator.setup()
    mitigator.classifier.train()

    optimizer = torch.optim.AdamW(mitigator.get_trainable_parameters(), lr=args.lr)

    dataset = OctopusTrainDataset(args.train_json, args.image_root)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    image_kwarg = ModelHelper.get_image_kwarg_name(args.model_type)

    for epoch in range(args.epochs):
        for batch in dataloader:
            prompt = build_prompt(processor, batch["query"][0])
            inputs = processor(
                text=prompt,
                images=batch["image"][0],
                return_tensors="pt",
            )
            inputs = {k: v.to(args.device) if torch.is_tensor(v) else v for k, v in inputs.items()}

            input_ids = inputs["input_ids"]
            attention_mask = inputs.get("attention_mask")
            pixel_values = inputs.get(image_kwarg)
            image_grid_thw = inputs.get("image_grid_thw")

            outputs = mitigator.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_scores=True,
                return_dict_in_generate=True,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                **{image_kwarg: pixel_values, "image_grid_thw": image_grid_thw},
            )

            action_scores = torch.stack(outputs.scores[1], dim=1)
            y_w = batch["y_w"].to(action_scores.device)
            y_l = batch["y_l"].to(action_scores.device)
            loss = mitigator.compute_loss(action_scores, y_w, y_l, beta=args.beta)
            loss = loss.mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        mitigator.save_pretrained(args.output_path)


if __name__ == "__main__":
    main()
