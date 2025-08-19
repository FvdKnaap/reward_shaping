# Credit Assignment in Mixed-Sparsity Multi-Objective Reinforcement Learning through Reward Shaping and Reflectional Equivariance

A research project for reward shaping in reinforcement learning.

## Table of Contents

- [Project Structure](#project-structure)
- [Installation](#installation)
- [File/Folder Explanations](#filefolder-explanations)
- [Usage](#usage)
- [Configuration](#configuration)


---

## Project Structure

```
code/
    plot_single_policy.py
    plot_wandb.py
    save_wandb.py
    requirements.txt
    setup.py
    configs/
    morl_baselines/
    src/

```

## Installation

1. **Clone the repository**  
   ```sh
   git clone <repo-url>
   cd code
   ```

2. **Create and activate a conda environment**  
   ```sh
   conda create -n shaping python=3.10
   conda activate shaping
   ```

3. **Install dependencies**  
   ```sh
   pip install -r requirements.txt
   ```
   Or, if you want to install as a package:
   ```sh
   pip install -e .
   ```

## File/Folder Explanations

- **plot_single_policy.py**  
  Script for plotting results of a single policy.

- **plot_wandb.py**  
  Script for plotting results using Weights & Biases runs.

- **save_wandb.py**  
  Script to fetch and merge Weights & Biases run data filtered by config parameters.

- **requirements.txt**  
  Python dependencies for the project.

- **setup.py**  
  Packaging and installation script. See [`setup.py`](setup.py).

- **configs/**  
  Contains configuration files (e.g., `config.yaml`) for experiments.
.

- **morl_baselines/**  
  Baseline algorithms and utilities for multi-objective RL. This comes from https://github.com/LucasAlegre/morl-baselines/tree/main?tab=readme-ov-file.

  @inproceedings{felten_toolkit_2023,
	author = {Felten, Florian and Alegre, Lucas N. and Now{\'e}, Ann and Bazzan, Ana L. C. and Talbi, El Ghazali and Danoy, Gr{\'e}goire and Silva, Bruno Castro da},
	title = {A Toolkit for Reliable Benchmarking and Research in Multi-Objective Reinforcement Learning},
	booktitle = {Proceedings of the 37th Conference on Neural Information Processing Systems ({NeurIPS} 2023)},
	year = {2023}
}

It has been adapted in this work.

## Usage

- **Plotting a single policy:**  
  ```sh
  python plot_single_policy.py [args]
  ```

- **Plotting with Weights & Biases:**  
  ```sh
  python plot_wandb.py [args]
  ```

- **Training/experiments:**  
  (Describe how to launch training or experiments, if applicable.)

- **src/**  
  Source code for the main package, including:
  - `pareto_front/`: Pareto front computation and utilities.
  - `reward_shaping/`: Reward shaping architectures and models.

## Configuration

- Main configuration file: `configs/config.yaml`
- (Describe configuration options as needed.)

