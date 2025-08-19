import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import t
import scienceplots  # Needed for the 'science' plot style
import argparse    # Import the argparse library

def plot_data(args):
    """
    Reads data from multiple CSV files and plots a separate line with a 95%
    confidence interval for each file on the same graph.

    Args:
        args (argparse.Namespace): An object containing the parsed command-line arguments.
    """
    # Construct the full file paths from the provided prefixes and environment
    file_paths = [os.path.join(prefix, f"{args.env}.csv") for prefix in args.prefixes]
    
    # Validate that the number of labels matches the number of paths if provided
    if args.labels and len(args.labels) != len(file_paths):
        print(f"Error: You provided {len(args.labels)} labels but {len(file_paths)} file paths. The counts must match.")
        return

    # Set up the plot style
    plt.style.use(['science', 'grid'])
    fig, ax = plt.subplots(figsize=(10, 6))

    # Loop through each file path to plot its data
    for i, file_path in enumerate(file_paths):
        try:
            df = pd.read_csv(file_path)
        except FileNotFoundError:
            print(f" Warning: The file '{file_path}' was not found and will be skipped.")
            continue

        # --- Calculations for the current file ---
        grouped = df.groupby(args.step_col)[args.value_col].agg(['mean', 'std', 'count'])
        
        # Add the initial zero-point row for a clean plot start
        new_row = pd.DataFrame({'mean': [0], 'std': [0], 'count': [grouped['count'].iloc[0] if not grouped.empty else 1]}, index=[0])
        grouped = pd.concat([new_row, grouped]).sort_index()

        # Calculate 95% CI using the t-distribution
        n = grouped['count']
        # Handle cases where n<=1, which would result in an error
        t_score = t.ppf(0.975, df=np.maximum(1, n - 1))
        
        sem = grouped['std'] / np.sqrt(n)
        ci = t_score * sem
        grouped['ci_lower'] = grouped['mean'] - ci
        grouped['ci_upper'] = grouped['mean'] + ci

        # --- Plotting for the current file ---
        # Determine the label for the current line
        if args.labels:
            label = args.labels[i]
        else:
            # If no labels are provided, infer from the prefix directory name
            label = os.path.basename(os.path.normpath(args.prefixes[i]))

        ax.plot(grouped.index, grouped['mean'], label=label)
        ax.fill_between(
            grouped.index,
            grouped['ci_lower'],
            grouped['ci_upper'],
            alpha=0.2
        )
        print(
            f"Results for {label}: "
            f"Final Mean: {grouped['mean'].iloc[-1]:.2f} ± {ci.iloc[-1]:.2f}, "
            f"Max Mean: {grouped['mean'].max():.2f}"
        )
       
    # --- Final Plot Configuration ---
    if not ax.lines:
        print("Error: No data was plotted. Check if your CSV files exist and contain data.")
        plt.close(fig)
        return
    
    # Set labels and legend with specified font sizes
    ax.set_xlabel('Timestep', fontsize=20)
    ax.set_ylabel(args.value_col.replace('_', ' ').title(), fontsize=20) # Nicer formatting for the label
    ax.legend(fontsize=16)
    fig.tight_layout()
    plt.grid(False)

    # Create output directory and save the plot
    os.makedirs(args.output_dir, exist_ok=True)
    output_filename = os.path.join(args.output_dir, f"{args.value_col.replace('/', '_')}_{args.env}_plot.png")
    plt.savefig(output_filename)
    print(f"\n Plot saved successfully as '{output_filename}'")
    
    plt.show()


if __name__ == "__main__":
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(
        description="Plot data with confidence intervals from multiple CSV files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Shows default values in help message
    )

    parser.add_argument("--prefixes", nargs='+', required=True,
                        help="Space-separated list of file path prefixes (e.g., 'pareto-front/dense/' 'pareto-front/sparse/').")
    parser.add_argument("--env", type=str, required=True,
                        help="Environment name to complete the file path (e.g., 'mo-swimmer-v5'). The script will look for '{prefix}{env}.csv'.")
    parser.add_argument("--labels", nargs='+',
                        help="Optional: Space-separated list of labels for the plot legend. Must match the number of prefixes.")
    parser.add_argument("--value_col", type=str, default="eval/hypervolume",
                        help="The column for the y-axis (the metric to plot).")
    parser.add_argument("--step_col", type=str, default="global_step",
                        help="The column for the x-axis (e.g., timesteps or epochs).")
    parser.add_argument("--output_dir", type=str, default="figures",
                        help="Directory to save the generated plot.")
    
    args = parser.parse_args()
    plot_data(args)