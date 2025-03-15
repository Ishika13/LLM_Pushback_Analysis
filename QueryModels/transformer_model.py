from transformers import pipeline
import torch
from tqdm import tqdm

class TransformerModel:
    def __init__(self, model_name, model_type):
        self.name = model_name
        self.pipe = pipeline(
            model_type,
            model=model_name,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
            trust_remote_code=True
            #pad_token_id=self.pipe.model.tokenizer.eos_token_id
            
        )
        self.pipe.tokenizer.pad_token_id = self.pipe.tokenizer.eos_token_id # Set pad_token_id to eos_token_id

    def predict_withformatting(self, queries, formatting, batch_size=32, max_tokens=500):
        """
        Accepts a list of queries and a formatting type.
        For "conversational_agent", it returns the generated text.
        For "instructional_agent", it extracts the message content accordingly.
        """
        outputs = [out  for out in tqdm(self.pipe(queries, batch_size=batch_size,penalty_alpha=0.6,do_sample = True,
      top_k=5,temperature=0.05,repetition_penalty=1.2,
      max_new_tokens=500,
      truncation=True,max_length= 750,
      pad_token_id=self.pipe.tokenizer.eos_token_id,), total=len(queries))]
        return [out[0]["generated_text"][-1]["content"] for out in outputs]
        