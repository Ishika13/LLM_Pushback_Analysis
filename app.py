### transformer_model
import pandas as pd
from transformers import pipeline
from QueryFormatters import get_query_formatter
from QueryModels import query_model



## explanation of the format given for each model, 
##   transformer_model_type,model_name,query_format_type,output_format_type

input_file = "test.csv"
output_file = "test_output.csv"
query_models = [
    ("text-generation","google/gemma-2-2b-it","instructional_agent","instructional_agent"),]

judges = [
    ("text-generation","meta-llama/Llama-3.1-8B-Instruct","instructional_agent","instructional_agent"),
    ("text-generation","google/gemma-2-2b-it","instructional_agent","instructional_agent"),]
enable_judges = True
enable_prediction = True


def string_builder(template, **kwargs):
    return template.format(**kwargs)

df = pd.read_csv(input_file, usecols=["Prompt ID","Prompt","Error","Error Type","Error Salience","Model Response"])
# generate responses
enable_prediction = False
if(enable_prediction):
    for transformer_model_type,model_name,query_format_type,output_format_type in query_models:
        query_formatter = get_query_formatter(query_type=query_format_type)
        model = query_model(model_name=model_name,transformer_model=transformer_model_type)
        responses = []
        queries = [query_formatter(prompt) for prompt in df["Prompt"]]
        responses = model.predict_withformatting(queries,"instructional_agent")
        df["Model Response"] = responses
        df.to_csv(model_name+output_file, index=False)
    print("Predictions saved to ",output_file)

enable_judges = True

if (enable_judges):
    for judge_name,query_format in judges:
        query_formatter = get_query_formatter("instructional_agent")
        model = query_model(model_name=judge_name,transformer_model="text-generation")
        # Type 1: Error acknowledgement
        prompt_template = "Find out if this model response {response} has acknowledged the following error {error} of the query {query}. If yes, say True else False ,make sure to print nothing else ,just one word"
        errors = df["Error"]
        responses = df["Model Response"]
        type1_prompts = [ string_builder(prompt_template, query=prompt, error=error,response = response) for prompt,error,response in zip(df["Prompt"],df["Error"],df["Model Response"])]
        type1_queries = [query_formatter(prompt) for prompt in type1_prompts]
        type1_responses = model.predict_withformatting(type1_queries,"instructional_agent")
        df[judge_name+"_predictions_error_acknowledgement"] = type1_responses
        #Type 2: Error correction
        prompt_template = "Find out if this model response {response} has corrected the following error {error} of the query {query}. If yes, say True else False ,make sure to print nothing else ,just one word"
        type2_prompts = [ string_builder(prompt_template, query=prompt, error=error,response = response) for prompt,error,response in zip(df["Prompt"],df["Error"],df["Model Response"])]
        type2_queries = [query_formatter(prompt) for prompt in type2_prompts]
        type2_responses = model.predict_withformatting(type2_queries,"instructional_agent")
        df[judge_name+"_predictions_error_correction"] = type2_responses
    df.to_csv(output_file, index=False)
    print("Evaluations saved to ",output_file)







