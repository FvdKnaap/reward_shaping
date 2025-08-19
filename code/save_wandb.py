import wandb
import pandas as pd
from datetime import datetime, timedelta
from dateutil import parser
import os
import argparse # Import the argparse library

def main():
    # --- Argument Parsing ---
    # Create the parser
    arg_parser = argparse.ArgumentParser(
        description="Fetch and merge wandb run data filtered by a specific config parameter and date."
    )

    # Add arguments
    arg_parser.add_argument("--entity", type=str, default="",
                            help="Weights & Biases entity (username or team). Can often be omitted.")
    arg_parser.add_argument("--project", type=str, required=True,
                            help="Weights & Biases project name (e.g., 'sparse').")
    arg_parser.add_argument("--param_key", type=str, default="env_id",
                            help="The configuration key to filter by (e.g., 'env_id').")
    arg_parser.add_argument("--param_value", type=str, required=True,
                            help="The configuration value to match (e.g., 'mo-hopper-v5').")
    arg_parser.add_argument("--history_keys", nargs='+',
                            default=["global_step", "eval/spacing", "eval/hypervolume", "eval/eum", "eval/cardinality"],
                            help="A space-separated list of history keys (metrics) to fetch.")
    arg_parser.add_argument("--output_dir", type=str, default="pareto-front",
                            help="Base directory to save the output CSV file.")

    # Parse the arguments from the command line
    args = arg_parser.parse_args()
    # ---------------------

    # --- Main script logic ---
    # Construct the output path and create directories if they don't exist
    output_path = os.path.join(args.output_dir, args.project)
    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(output_path, f"{args.param_value}.csv")

    # Initialize wandb API and get runs
    api = wandb.Api()
    print(f"Fetching runs from: {args.entity}/{args.project}")
    runs = api.runs(f"{args.entity}/{args.project}")

    all_data = []
    print(f"Filtering runs for config '{args.param_key}: {args.param_value}' on {args.target_date}...")
    
    for run in runs:
        # We need to parse the string timestamp and remove timezone info for a direct comparison
        created_at = parser.parse(str(run.created_at)).replace(tzinfo=None)
        
        # Check if the config parameter matches
        config = {k: v for k, v in run.config.items() if not k.startswith("_")}
        if config.get(args.param_key) == args.param_value:
            print(f"  - Found matching run: {run.name} (ID: {run.id}) created at {created_at}")
            df = run.history(keys=args.history_keys)

            # Ensure the dataframe is not empty and has essential columns
            if not df.empty and "eval/hypervolume" in df.columns:
                df["run_id"] = run.id
                df["run_name"] = run.name
                all_data.append(df)

    if not all_data:
        print("No matching runs with the specified history keys found. Exiting.")
        return

    # Merge all collected dataframes and save to a CSV file
    merged_df = pd.concat(all_data, ignore_index=True)
    merged_df.to_csv(output_file, index=False)
    print(f"\n Saved merged file: {output_file} ({len(merged_df)} rows from {len(all_data)} runs)")


if __name__ == "__main__":
    main()