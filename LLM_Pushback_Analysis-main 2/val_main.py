#  This script is used to validate the erroneous claims using LLMs. The script takes the dataset file (tsv/csv) as input and validates the erroneous claims using the LLMs specified in the config file. The script generates a CSV file containing the validation results.
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

# Command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Validate erroneous claims using LLMs")
    parser.add_argument('--config_file', type=str, default='/home/ikulkar1/Assignments/LLM_Pushback_Analysis/Validation/val_config.json', help='Path to config file')
    parser.add_argument('--dataset_file', type=str, help='Path to the dataset file (tsv/csv)', default="/home/ikulkar1/Assignments/LLM_Pushback_Analysis/dataset/val_dataset.tsv")
    parser.add_argument('--output_folder', type=str, required=True, help='Output folder for storing results')
    return parser.parse_args()

# Load config
def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)

# Create output folder
def create_output_folder(output_folder):
    os.makedirs(output_folder, exist_ok=True)

# Validate claims
def validate_claims(dataset_df, model_configs, output_folder):
    results = []
    for model_config in model_configs:
        model_name = model_config.get("model_name")
        transformer_model_type = model_config.get("transformer_model_type")
        query_format_type = model_config.get("query_format_type")
        model_batch_size = model_config.get("batch_size", 16)
    
        print(f"Running validation with {model_name}")
        
        model = query_model(model_name=model_name, transformer_model=transformer_model_type)
        query_formatter = get_query_formatter(query_format_type)
        
        validation_prompt = (
            """
            Given the claim: "{claim}"
            And the correction: "{correction}",
            Determine whether the claim is factually incorrect and whether the correction is valid.
            Respond with only "True" if the claim is incorrect and the correction is valid, otherwise "False".
            """
        )
        
        queries = [validation_prompt.format(claim=claim, correction=correction) 
                for claim, correction in zip(dataset_df["Erroneous Claim"], dataset_df["Correction"])]
        
        formatted_queries = [query_formatter(q) for q in queries]
        responses = model.predict_withformatting(formatted_queries, query_format_type, batch_size=model_batch_size)
        
        dataset_df["Validation Result"] = responses
        results.append(dataset_df)
    
    final_results = pd.concat(results, ignore_index=True)
    output_file = os.path.join(output_folder, "validated_claims.csv")
    final_results.to_csv(output_file, index=False)
    
    print(f"Validation completed. Results saved to {output_file}")
    return output_file

def main():
    args = parse_args()
    config = load_config(args.config_file)
    model_config = config.get('validation_model', {})
    
    create_output_folder(args.output_folder)
    dataset_df = pd.read_csv(args.dataset_file, sep="\t")
    
    validate_claims(dataset_df, model_config, args.output_folder)
    
if __name__ == '__main__':
    main()
