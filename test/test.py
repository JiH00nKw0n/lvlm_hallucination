import requests
from PIL import Image

from transformers import AutoProcessor

model_id = "llava-hf/llava-1.5-7b-hf"

processor = AutoProcessor.from_pretrained(model_id)

# Define a chat history and use `apply_chat_template` to get correctly formatted prompt
# Each value in "content" has to be a list of dicts with types ("text", "image") 
conversation = [
    {

        "role": "user",
        "content": [
            {"type": "text", "text": "What are these?"},
            {"type": "image"},
        ],
    },
    # {
    # 
    #     "role": "assistant",
    #     "content": [
    #         {"type": "text", "text": "What are these?"},
    #     ],
    # },
]

prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

print(prompt)

image_file = "http://images.cocodataset.org/val2017/000000039769.jpg"
raw_image = Image.open(requests.get(image_file, stream=True).raw)

# Process inputs
inputs = processor(images=[raw_image], text=[prompt, prompt], return_tensors='pt')
print(inputs.keys())
print(inputs["input_ids"])
# Decode the input_ids to see the tokenized text
decoded_text = processor.decode(inputs.input_ids[0], skip_special_tokens=False)
print("Decoded input:")
print(decoded_text)
