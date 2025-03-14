from .transformer_model import TransformerModel
from .openai_model import OpenAIBatchModel

def query_model(model_name, transformer_model=None):
    if transformer_model:
        return TransformerModel(model_name, transformer_model)
    else:
        return OpenAIBatchModel(model_name)
