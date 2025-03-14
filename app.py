import os
import sys
import json
import argparse
import pandas as pd
from datetime import datetime
from pathlib import Path
from QueryFormatters import get_query_formatter
from QueryModels import query_model


def parse_args():
    parser = argparse.ArgumentParser(description="Run experiment")
    parser.add_argument('--config_file', type=str, default='config.json', help='Path to config file')
    parser.add_argument('--dataset_file', type=str, help='Path to the dataset file (tsv/csv)')
    parser.add_argument('--mode', type=str, choices=['queries', 'judging', 'both'], default='both',
                        help='Select mode: run only queries, only judging, or both')
    parser.add_argument('--user', type=str, help='User name')
    parser.add_argument('--output_folder', type=str, help='Only needed if mode is "judging". Otherwise, will be created dynamically.')
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)


def validate_config(args):
    # Load config
    config = load_config(args.config_file)
    query_models = config.get('query_models', [])
    judges = config.get('judges', [])

    # Validate dataset file if need to do queries
    if args.mode in ['queries', 'both']:
        if not args.dataset_file:
            print(f"Arg 'dataset_file' must be provided for mode '{args.mode}'. Exiting.")
            sys.exit(1)
        if not Path(args.dataset_file).exists():
            print(f"Dataset file '{args.dataset_file}' does not exist. Exiting.")
            sys.exit(1)
        if not len(query_models):
            print(f"No 'query_models' specified in {args.config_file} for mode {args.mode}. Exiting.")
            sys.exit(1)

    if args.mode in ['judging']:
        if args.output_folder is None:
            print(f"For mode '{args.mode}', arg 'output_folder' must be provided")
            sys.exit(1)
        if not Path(args.output_folder).exists():
            print(f"'output_folder' {args.output_folder} does not exist. Exiting")
            sys.exit(1)

    if args.mode in ['judging', 'both']:
        if not len(query_models):
            print(f"No 'judges' specified in {args.config_file} for mode {args.mode}. Exiting.")
            sys.exit(1)


def create_output_folder(dataset_file, user, output_folder_base="results"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    output_folder_name = f"{dataset_file}_{timestamp}" if user is None else f"{dataset_file}_{user}_{timestamp}"
    output_folder = f"{output_folder_base}/{output_folder_name}"
    os.makedirs(output_folder, exist_ok=True)
    return output_folder


def run_queries(df, query_models, output_folder):
    files_to_judge = []
    print(f"Running {len(df)} queries for {len(query_models)} models")
    for idx, model_conf in enumerate(query_models):
        transformer_model_type = model_conf.get("transformer_model_type")
        model_name = model_conf.get("model_name")
        query_format_type = model_conf.get("query_format_type")
        print(f"Queries: model_name={model_name}, query_format_type={query_format_type}, transformer_model_type={transformer_model_type}")

        # Instantiate the model (this will choose OpenAIBatchModel if transformer_model_type is None)
        model = query_model(model_name=model_name, transformer_model=transformer_model_type)
        
        # Get the query formatter
        query_formatter = get_query_formatter(query_format_type)
        queries = [query_formatter(prompt) for prompt in df["Prompt"]]
        print(f"Generated {len(queries)} queries.")
        
        try:
            responses = model.predict_withformatting(queries, "instructional_agent")
        except Exception as e:
            print(f"Error during query prediction: {e}")
            raise
        
        # print(f"Received responses: {responses}")
        df["Model Response"] = responses
        file_name = os.path.join(output_folder, f"predictions_{idx}.csv")
        files_to_judge.append(file_name)
        df.to_csv(file_name, index=False)
    return files_to_judge


def generate_judging_responses(df, model, query_formatter, query_format_type, prompt_template, column_suffix):
    # Abstracts the logic of calling a judging prompt and returning the responses
    prompts = [
        prompt_template.format(query=prompt, error=error, response=response)
        for prompt, error, response in zip(df["Prompt"], df["Error"], df["Model Response"])
    ]
    queries = [query_formatter(p) for p in prompts]
    responses = model.predict_withformatting(queries, query_format_type)
    return responses


def run_judging(files_to_judge, judges):
    # Define prompts
    judging_prompt1 = ("""
        Read the user prompt, error description, and system response below.

        USER PROMPT: "{query}"
        ERROR: "{error}"
        SYSTEM RESPONSE: "{response}"
                                
        Now rate the following statement as True or False: "The system response addresses the error in the prompt, at least in some way."
        Your output must be one word only: "True" or "False"
        """)
    
    judging_prompt2 = ("""
        Read the user prompt, error description, and system response below.

        USER PROMPT: "{query}"
        ERROR: "{error}"
        SYSTEM RESPONSE: "{response}"
                                
        Now rate the following statement as True or False: "The system response has corrected the error in the prompt."
        Your output must be one word only: "True" or "False"
        """)
    
    for file in files_to_judge:
        df = pd.read_csv(file)
        print(f"Read file {file} to df.")

        file_path = Path(file)
        judging_output_file = file_path.with_name(file_path.stem + "_judges" + file_path.suffix)
        print(f"judging_output_file: {judging_output_file}")


        for judge in judges:  # judges are JSON objects
            model_name = judge.get("model_name")
            transformer_model_type = judge.get("transformer_model_type")
            query_format_type = judge.get("query_format_type")
            print(f"Judge: model_name: {model_name}, transformer_model_type: {transformer_model_type}")
            
            query_formatter = get_query_formatter(query_format_type)
            model = query_model(model_name=model_name, transformer_model=transformer_model_type)

            # Type 1: Error Acknowledgement
            type1_responses = generate_judging_responses(df, model, query_formatter, query_format_type, judging_prompt1, "error_acknowledgement")
            df[f"{model_name}_predictions_error_acknowledgement"] = type1_responses

            # Type 2: Error Correction
            type2_responses = generate_judging_responses(df, model, query_formatter, query_format_type, judging_prompt2, "error_correction")
            df[f"{model_name}_predictions_error_correction"] = type2_responses

        print(f"Writing judged output to file: {judging_output_file}")
        df.to_csv(judging_output_file, index=False)


def main():
    # Parse command-line arguments
    args = parse_args()
    validate_config(args)
    config = load_config(args.config_file)
    query_models = config.get('query_models', [])
    judges = config.get('judges', [])

    # set to None so won't be undefined
    output_folder = None

    files_to_judge = []

    # QUERIES MODE
    if args.mode in ['queries', 'both']:
        # Initialize output folder
        dataset_file_name = Path(args.dataset_file).stem

        # Load dataset
        df = pd.read_csv(args.dataset_file, sep="\t")

        output_folder = create_output_folder(dataset_file_name, args.user)
        print(f"Output folder: {output_folder}")

        # Save a copy of the dataset and settings in the output folder for traceability
        df.to_csv(os.path.join(output_folder, Path(args.dataset_file).name), index=False)
        with open(os.path.join(output_folder, "config.json"), "w") as f:
            f.write(json.dumps(config, indent=4))
        
        # Run the query models
        query_models = config.get('query_models', [])
        files_to_judge = run_queries(df.copy(), query_models, output_folder)
        print(f"files_to_judge: {files_to_judge}")


    print("\n\n--------- JUDGING --------\n\n")

    # JUDGING MODE
    if args.mode in ['judging', 'both']:
        if not output_folder:
            output_folder = args.output_folder
            print(f"output_folder set to: {output_folder}")

        files_to_judge = [str(file) for file in Path(output_folder).glob("predictions_*.csv")]
        if not files_to_judge:
            print(f"No predictions files found in output folder '{output_folder}'")
            sys.exit(1)
        print(f"Files to judge: {files_to_judge}")
        
        run_judging(files_to_judge, judges)

    print("Run complete.")
    
if __name__ == '__main__':
    main()
