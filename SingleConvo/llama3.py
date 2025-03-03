from SingleConvo import SingleConvo
from transformers import AutoTokenizer, AutoModelForCausalLM
# Use a pipeline as a high-level helper
from transformers import pipeline



class llama3(SingleConvo):
    def __init__(self):
        # Load model directly
        self.pipe = pipeline("text-generation", model="meta-llama/Llama-3.1-8B")
    def predict(self, query):
        # Use the pipeline to generate text
        return self.pipe(query)[0]['generated_text']

llama_obj = llama3()
print(llama_obj.predict("What is the meaning of life?"))