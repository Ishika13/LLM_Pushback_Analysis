import os
import sys
import json
import argparse
import pandas as pd
import time
from datetime import datetime
from pathlib import Path
from QueryFormatters import get_query_formatter
from QueryModels import query_model
from tqdm import tqdm

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


def run_queries(dataset_df, query_models, output_folder):
    results = []
    
    for idx, model_conf in tqdm(enumerate(query_models),total=len(query_models)):
        start_time = time.time()
        transformer_model_type = model_conf.get("transformer_model_type")
        model_name = model_conf.get("model_name")
        query_format_type = model_conf.get("query_format_type")
        model_batch_size = model_conf.get("batch_size", 32)
        print(f"Queries: model_name={model_name}, query_format_type={query_format_type}, transformer_model_type={transformer_model_type}")

        model = query_model(model_name=model_name, transformer_model=transformer_model_type)
        query_formatter = get_query_formatter(query_format_type)
        
        queries = [query_formatter(prompt) for prompt in dataset_df["Prompt"]]
        print(f"Generated {len(queries)} queries.")

        responses = model.predict_withformatting(queries, "instructional_agent", batch_size=model_batch_size)

        # Make a fresh copy of the dataset and add columns for this model's results
        df_copy = dataset_df.copy()
        df_copy['Model Name'] = model_name
        df_copy["Model Response"] = responses
        
        results.append(df_copy)
        print(f"Done {model_name}: {time.time()-start_time} seconds.")

    final_results = pd.concat(results, ignore_index=True)
    
    preds_file = os.path.join(output_folder, "predictions.csv")
    final_results.to_csv(preds_file, index=False)
    
    return preds_file



def generate_judging_responses(df, model, query_formatter, query_format_type, prompt_template, batch_size=32):
    # Abstracts the logic of calling a judging prompt and returning the responses
    prompts = [
        prompt_template.format(query=prompt, error=error, response=response)
        for prompt, error, response in zip(df["Prompt"], df["Error"], df["Model Response"])
    ]
    queries = [query_formatter(p) for p in prompts]
    responses = model.predict_withformatting(queries, query_format_type, batch_size)
    return responses


def run_judging(preds_file, judges):
    # Define prompts
    judging_prompt1 = ("""
        Read the user prompt, error description, and system response below.

        USER PROMPT: "{query}"
        ERROR: "{error}"
        SYSTEM RESPONSE: "{response}"
        
        Now rate the following statement as True or False: 
        "The system response addresses the error in the prompt, at least in some way."
        Your output must be one word only: "True" or "False"
        """)

    judging_prompt2 = ("""
        Read the user prompt, error description, and system response below.

        USER PROMPT: "{query}"
        ERROR: "{error}"
        SYSTEM RESPONSE: "{response}"
        
        Now rate the following statement as True or False:
        "The system response has corrected the error in the prompt."
        Your output must be one word only: "True" or "False"
        """)

    df = pd.read_csv(preds_file)
    print(f"Read file {preds_file} into df.")

    file_path = Path(preds_file)
    judging_output_file = file_path.with_name(file_path.stem + "_judges" + file_path.suffix)
    print(f"judging_output_file: {judging_output_file}")

    for judge in judges:  # judges are JSON objects
        model_name = judge.get("model_name")
        transformer_model_type = judge.get("transformer_model_type")
        query_format_type = judge.get("query_format_type")
        model_batch_size = judge.get("batch_size", 32)
        print(f"Judge: model_name={model_name}, transformer_model_type={transformer_model_type}, batch_size={model_batch_size}")

        query_formatter = get_query_formatter(query_format_type)
        model = query_model(model_name=model_name, transformer_model=transformer_model_type)

        # Type 1: Error Acknowledgement
        type1_responses = generate_judging_responses(
            df, model, query_formatter, 
            query_format_type, judging_prompt1, 
            batch_size=model_batch_size
        )
        df[f"{model_name}_predictions_error_acknowledgement"] = type1_responses

        # Type 2: Error Correction
        type2_responses = generate_judging_responses(
            df, model, query_formatter, 
            query_format_type, judging_prompt2,
            batch_size=model_batch_size
        )
        df[f"{model_name}_predictions_error_correction"] = type2_responses

    print(f"Writing judged output to file: {judging_output_file}")
    df.to_csv(judging_output_file, index=False)

    return judging_output_file


def create_metadata_file(output_folder, dataset_filename, config):
    """
    Creates a metadata.json file with timestamp, dataset filename, and config.
    
    Args:
        output_folder: Path to the folder where metadata.json will be saved
        dataset_filename: Name of the dataset file
        config: Configuration dictionary
    """
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "dataset_filename": dataset_filename,
        "config": config
    }
    with open(os.path.join(output_folder, "metadata.json"), "w") as f:
        f.write(json.dumps(metadata, indent=4))


def main():
    # Parse command-line arguments
    args = parse_args()
    validate_config(args)
    config = load_config(args.config_file)
    query_models = config.get('query_models', [])
    judges = config.get('judges', [])

    # set to None so won't be undefined
    output_folder = None
    preds_file = None

    # QUERIES MODE
    if args.mode in ['queries', 'both']:
        # Initialize output folder
        dataset_file_name = Path(args.dataset_file).stem

        # Load dataset
        df = pd.read_csv(args.dataset_file, sep="\t")

        output_folder = create_output_folder(dataset_file_name, args.user)
        print(f"Output folder: {output_folder}")

        # Save a copy of the dataset in the output folder
        df.to_csv(os.path.join(output_folder, Path(args.dataset_file).name), index=False)
        
        # Write a metadata.json with config, timestamp and dataset filename
        create_metadata_file(output_folder, Path(args.dataset_file).name, config)
        
        # Run the query models (will return a single "predictions.csv")
        preds_file = run_queries(df.copy(), query_models, output_folder)
        print(f"Predictions file: {preds_file}")


    print("\n\n--------- JUDGING --------\n\n")

    # JUDGING MODE
    if args.mode in ['judging', 'both']:
        # If we didn't run queries just now, we need the existing output_folder the user passed
        if not output_folder:
            output_folder = args.output_folder
            print(f"Output folder set to: {output_folder}")

        # We'll usually know the predictions file location, but if not, we'll try to find it
        if not preds_file:
            preds_file = os.path.join(output_folder, "predictions.csv")
            print(f"Found preds_file: {preds_file}")
            if not os.path.exists(preds_file):
                print(f"No predictions file found in {preds_file}")
                sys.exit(1)

        run_judging(preds_file, judges)

    print("Run complete.")
    
if __name__ == '__main__':
    main()
