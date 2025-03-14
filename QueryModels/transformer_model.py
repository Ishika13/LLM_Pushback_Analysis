from transformers import pipeline
import torch

class TransformerModel:
    def __init__(self, model_name, model_type):
        self.name = model_name
        self.pipe = pipeline(
            model_type,
            model=model_name,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
            #pad_token_id=self.pipe.model.tokenizer.eos_token_id
            
        )
        self.pipe.tokenizer.pad_token_id = self.pipe.tokenizer.eos_token_id # Set pad_token_id to eos_token_id

    def predict_withformatting(self, queries, formatting, batch_size=32, max_tokens=500):
        """
        Accepts a list of queries and a formatting type.
        For "conversational_agent", it returns the generated text.
        For "instructional_agent", it extracts the message content accordingly.
        """
        if formatting == "conversational_agent":
            outputs = self.pipe(queries, batch_size=batch_size)
            return [out["generated_text"] for out in outputs]
        elif formatting == "instructional_agent":
            outputs = self.pipe(queries, batch_size=batch_size)
            return [out[0]["generated_text"] for out in outputs]
        else:
            raise ValueError(f"Unknown formatting type: {formatting}")
