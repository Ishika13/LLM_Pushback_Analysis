__all__ = ["query_model"]

from transformers import pipeline
import torch

from openai import OpenAI

def query_model(model_name,transformer_model=None):
    if transformer_model:
        return TransformerModel(model_name,transformer_model)
    else:
        client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key="<OPENROUTER_API_KEY>",
        )
        completion = client.chat.completions.create(
        model="openai/gpt-4o",
        messages=[
            {
            "role": "user",
            "content": "What is the meaning of life?"
            }
        ]
        )



class TransformerModel():
    def __init__(self,name,model_type):
        self.name = name
        self.pipe = pipeline(model_type, model=name,model_kwargs={"torch_dtype": torch.bfloat16},device_map="auto")

    def predict_withformatting(self,query,formatting,batch_size=32):
        if(formatting == "conversational_agent"):
            output = self.pipe(query)
            print(output)
            return [ out["generated_text"] for out in output]
        elif(formatting == "instructional_agent"):
            return [ out[0]["generated_text"][-1]["content"] for out in self.pipe(query)]
        




