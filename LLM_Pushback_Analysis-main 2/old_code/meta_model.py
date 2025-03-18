# Use a pipeline as a high-level helper
from transformers import pipeline
import torch
messages = [
    {"role": "user", "content": "Who are you?"},
]
pipe = pipeline("text-generation", model="meta-llama/Meta-Llama-3-8B",torch_dtype=torch.bfloat16)
pipe(messages)