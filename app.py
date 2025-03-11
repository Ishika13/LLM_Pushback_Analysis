### transformer_model
import pandas as pd
from transformers import pipeline
from QueryFormatters import get_query_formatter
from QueryModels import query_model
# from JudgeModels import load_judge
# from Logger import log_response
is_transformer_model = True
user = "karthik"
experiment_id = 1
query_formatter = get_query_formatter("instructional_agent")
model = query_model(model_name="google/gemma-2-2b-it",transformer_model="text-generation")
input_file = "test.csv"
output_file = "test_output.csv"
judges = ["llama3","llama3Instruct3"]
enable_judges = True
enable_prediction = True
enable_logging= True

def log_response(response):
    if(enable_logging):
        print(response)

df = pd.read_csv(input_file, usecols=["Prompt ID","Prompt","Error","Error Type","Error Salience","Model Response"])
# generate responses
if(enable_prediction):
    responses = []
    queries = [query_formatter(prompt) for prompt in df["Prompt"]]
    responses = model.predict_withformatting(queries,"instructional_agent")
    df["Model Response"] = responses
    df.to_csv(output_file, index=False)


if (enable_judges):
    for judge in judges:
        judge_responses = []
        judge_model = load_judge(judge)
        for prompt in df["Model Response"]:
            judge_response = judge_model.predict(prompt)
            log_response(judge_response)
            judge_responses.append(judge_response)
        with pd.ExcelWriter('output.xlsx') as writer:
            sheet_df.to_excel(writer, sheet_name='judge')

        df[judge] = judge_responses







