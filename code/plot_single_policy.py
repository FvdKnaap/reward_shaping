import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import scienceplots
import itertools
import os
from scipy.stats import t
import argparse # Import the argparse library

def load_data_from_runs(model_path: Path, args: argparse.Namespace) -> np.ndarray:
    """
    Loads and processes data from all .npz files for a specific model.

    Args:
        model_path (Path): Path to the model's directory (e.g., 'results/PPO').
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        A 2D numpy array where each row is the data from one run.
    """
    all_runs_data = []
    
    # Find all .npz files recursively within the model's directory
    npz_files = sorted(list(model_path.glob('**/*.npz')))
    
    if not npz_files:
        print(f"Warning: No .npz files found in {model_path}")
        return np.array([])

    print(f"Found {len(npz_files)} runs for model '{model_path.name}'")
    
    # Load data from each file
    for file_path in npz_files:
        try:
            with np.load(file_path) as data:
                if args.data_key in data:
                    # Assuming data is (timesteps, num_envs), average across envs
                    all_runs_data.append(data[args.data_key].mean(axis=1))
                else:
                    print(f"  - Warning: Key '{args.data_key}' not found in {file_path}. Available keys: {list(data.keys())}")
        except Exception as e:
            print(f"  - Error loading {file_path}: {e}")
            
    if not all_runs_data:
        return np.array([])

    # Special handling to group runs for a specific model if requested
    if args.group_by_model and model_path.name == args.group_by_model:
        print(f"Grouping runs for '{args.group_by_model}' with group size {args.group_size}...")
        chunk_size = args.group_size
        grouped_runs = [
            list(itertools.chain.from_iterable(all_runs_data[i : i + chunk_size]))
            for i in range(0, len(all_runs_data), chunk_size)
        ]
        all_runs_data = grouped_runs

    # Standardize run lengths by truncating to the shortest run
    min_length = min(len(run) for run in all_runs_data)
    standardized_data = [run[:min_length] for run in all_runs_data]
    
    return np.array(standardized_data)

def plot_learning_curves(args: argparse.Namespace):
    """
    Generates and saves the final plot based on user arguments.
    """
    plt.style.use(['science', 'grid'])
    fig, ax = plt.subplots(figsize=(10, 6))

    # Get all model subdirectories from the provided base directories
    all_model_dirs = []
    for base_dir in args.data_dirs:
        p = Path(base_dir)
        if not p.is_dir():
            print(f" Warning: Directory '{p}' not found, skipping.")
            continue
        all_model_dirs.extend([d for d in p.iterdir() if d.is_dir()])
    
    if not all_model_dirs:
        print("Error: No model directories found in the specified data_dirs. Exiting.")
        return

    # Plot data for each model
    for model_dir in all_model_dirs:
        data = load_data_from_runs(model_dir, args)
        
        # Prepend a column of zeros for a clean plot start at (0,0)
        data = np.hstack((np.zeros((data.shape[0], 1)), data))
        
        if data.size == 0:
            continue
        
        # Calculate mean, standard error, and confidence interval
        mean = np.mean(data, axis=0)
        n = data.shape[0]
        if n <= 1:
            print(f"  - Skipping CI for '{model_dir.name}' as it has only {n} run(s).")
            ci = np.zeros_like(mean)
        else:
            t_score = t.ppf(0.975, df=n - 1)
            sem = np.std(data, axis=0, ddof=1) / np.sqrt(n)
            ci = t_score * sem

        x_axis = np.arange(len(mean))
        
        # Plot the mean line and the confidence interval
        ax.plot(x_axis, mean, label=model_dir.name)
        ax.fill_between(x_axis, mean - ci, mean + ci, alpha=0.2)
        
        print(
            f"Results for {model_dir.name}: "
            f"Final Mean: {mean[-1]:.2f} ± {ci[-1]:.2f}, "
            f"Max Mean: {np.max(mean):.2f}"
        )

    # --- Final Plot Customization ---
    ax.set_xlabel(args.x_label, fontsize=20)
    ax.set_ylabel(args.y_label, fontsize=20)
    ax.legend(fontsize=16)
    fig.tight_layout()

    # Save and show the plot
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f'{args.title}.png')
    plt.grid(False)
    plt.savefig(output_path)
    print(f"\nPlot saved successfully as '{output_path}'")
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Plot learning curves with confidence intervals from .npz files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("--data_dirs", nargs='+', required=True, 
                        help="Space-separated list of base directories containing model folders.")
    parser.add_argument("--title", type=str, required=True, 
                        help="Title for the plot, also used as the output filename (without extension).")
    parser.add_argument("--data_key", type=str, default="results", 
                        help="The key for the data array within the .npz files.")
    parser.add_argument("--output_dir", type=str, default="figures", 
                        help="Directory to save the plot.")
    parser.add_argument("--x_label", type=str, default="Timesteps ($10^4$)", 
                        help="Label for the x-axis.")
    parser.add_argument("--y_label", type=str, default="Average Reward", 
                        help="Label for the y-axis.")
    parser.add_argument("--group_by_model", type=str, 
                        help="Optional: Name of a model whose runs should be grouped together.")
    parser.add_argument("--group_size", type=int, default=5, 
                        help="Number of runs to group together for the model specified by --group_by_model.")
    
    args = parser.parse_args()
    plot_learning_curves(args)