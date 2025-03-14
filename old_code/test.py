# Use a pipeline as a high-level helper
from transformers import pipeline
import torch

messages = [
    {"role": "user", "content": "Who are you?"},
]
ins = [messages,messages]
pipe = pipeline("text-generation", model="google/gemma-2-2b-it",device_map="auto",model_kwargs={"torch_dtype": torch.bfloat16})
output = pipe(ins)
print(output[0][0]["generated_text"]["content"])