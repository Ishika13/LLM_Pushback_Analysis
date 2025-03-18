from SingleConvo import SingleConvo
from transformers import AutoTokenizer, AutoModelForCausalLM
# Use a pipeline as a high-level helper
from transformers import pipeline
import torch

class llama3Instruct3(SingleConvo):
    def __init__(self):
        # Load model directly
        self.pipe = pipeline("text-generation", model="meta-llama/Llama-3.1-8B-Instruct",device_map="auto",model_kwargs={"torch_dtype": torch.bfloat16})
    def predict(self, query):
        messages = [
            {"role": "user", "content": query},
        ]
        # Use the pipeline to generate text
        response = self.pipe(messages,penalty_alpha=0.6,do_sample = True,
      top_k=5,temperature=0.05,repetition_penalty=1.2,
      max_new_tokens=500,
      truncation=True,
      pad_token_id=self.pipe.tokenizer.eos_token_id,)
        # print("RESPONSE",response,"\n")
        return response[0]["generated_text"][-1]["content"]


