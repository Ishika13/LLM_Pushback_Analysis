from SingleConvo import SingleConvo
from transformers import AutoTokenizer, AutoModelForCausalLM
# Use a pipeline as a high-level helper
from transformers import pipeline
from accelerate import Accelerator
import torch

# Initialize Accelerator for multi-GPU usage

accelerator = Accelerator()

class llama3(SingleConvo):
    def __init__(self):
        # Load model directly
        self.pipe = pipeline("text-generation", model="meta-llama/Llama-3.1-8B",model_kwargs={"torch_dtype": torch.bfloat16},device_map="auto")
        self.terminators = [
            self.pipe.tokenizer.eos_token_id
        ]
    def predict(self, query):
        # Use the pipeline to generate text
        query = query + " Replying to the above, I think "
        return self.pipe(query,
                         penalty_alpha=0.6,do_sample = True,
      top_k=5,temperature=0.05,repetition_penalty=1.2,
      max_new_tokens=500,
      truncation=True,
    pad_token_id=self.pipe.tokenizer.eos_token_id,
      )[0]["generated_text"]

