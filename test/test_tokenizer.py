import requests
from PIL import Image
from transformers import AutoProcessor


chat = "USER: <image>\nWhat are the things I should be cautious about when I visit this place\nASSISTANT:"
chat2 = "USER: <image>\nWhat are the things I should be cautious about when I visit this place What are the things I should be cautious about when I visit this place\nASSISTANT:"
model_name = "llava-hf/llava-1.5-7b-hf"

image_file = "http://images.cocodataset.org/val2017/000000039769.jpg"
raw_image = Image.open(requests.get(image_file, stream=True).raw)

processor = AutoProcessor.from_pretrained(model_name)

print(processor(
    text=[chat, chat2], 
    images=[raw_image, raw_image],
    return_tensors="pt",
    padding=True
).input_ids)