# LLM Pushback Experiment Runner

This script is designed to run experiments using query models and judges. It processes datasets, generates queries, and evaluates responses based on specified configurations.

## Configuration File

The configuration file is a JSON file that specifies the models and judges to be used in the experiment. Below is an example structure of the `config.json` file:

```json
{
    "query_models": [
        {
            "model_name": "example_model_1",
            "transformer_model_type": "type_1",
            "query_format_type": "format_1"
        },
        {
            "model_name": "example_model_2",
            "transformer_model_type": "type_2",
            "query_format_type": "format_2"
        }
    ],
    "judges": [
        {
            "model_name": "judge_model_1",
            "transformer_model_type": "judge_type_1",
            "query_format_type": "judge_format_1"
        }
    ]
}
```

- **query_models**: A list of models to be used for generating queries. Each model requires:
  - `model_name`: The name of the model.
  - `transformer_model_type`: The type of transformer model (can be `None`).
  - `query_format_type`: The format type for the queries.

- **judges**: A list of models to be used for judging the responses. Each judge requires:
  - `model_name`: The name of the judge model.
  - `transformer_model_type`: The type of transformer model (can be `None`).
  - `query_format_type`: The format type for the judging queries.

## Command-Line Arguments

The script accepts several command-line arguments to control its behavior:

- `--config_file`: Path to the configuration file (default: `config.json`).
- `--dataset_file`: Path to the dataset file (must be in TSV or CSV format).
- `--mode`: Select the mode of operation. Options are:
  - `queries`: Run only the query models.
  - `judging`: Run only the judging models.
  - `both`: Run both queries and judging (default).
- `--user`: User name to be included in the output folder name.
- `--output_folder`: Path to the output folder. Required if mode is `judging`; otherwise, it will be created dynamically.

### Example Usage

To run the script with a specific configuration and dataset:

```bash
python app.py --config_file my_config.json --dataset_file my_dataset.csv --mode both --user my_name
```

This command will run both the query and judging models using the specified configuration and dataset, and it will create an output folder with the user's name included.

## Output

The script generates output files in a dynamically created folder (or specified output folder). These include:

- Copies of the dataset and configuration for traceability.
- CSV files with model predictions.
- Judging results appended to the prediction files.

## Notes

- Ensure that the dataset file exists and is accessible.
- The configuration file must be correctly formatted and include all necessary model and judge specifications.
- The output folder will be created if it does not exist, but ensure the parent directory is writable.
- Use `requirements.txt` to install the dependancies.
