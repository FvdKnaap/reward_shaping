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

- **plot_single_policy.py**: Script for plotting results of a single policy.

- **plot_wandb.py**: Script for plotting results using Weights & Biases runs.

- **save_wandb.py**: Script to fetch and merge Weights & Biases run data filtered by config parameters.

- **requirements.txt**: Python dependencies for the project.

- **setup.py**: Packaging and installation script. See [`setup.py`](setup.py).

- **configs/**: Contains configuration files (e.g., `config.yaml`) for experiments.

- **morl_baselines/**  
  Baseline algorithms and utilities for multi-objective RL. This comes from https://github.com/LucasAlegre/morl-baselines/tree/main?tab=readme-ov-file.

  @inproceedings{felten_toolkit_2023,
	author = {Felten, Florian and Alegre, Lucas N. and Now{\'e}, Ann and Bazzan, Ana L. C. and Talbi, El Ghazali and Danoy, Gr{\'e}goire and Silva, Bruno Castro da},
	title = {A Toolkit for Reliable Benchmarking and Research in Multi-Objective Reinforcement Learning},
	booktitle = {Proceedings of the 37th Conference on Neural Information Processing Systems ({NeurIPS} 2023)},
	year = {2023}
  }

  It has been adapted in this work. Specifically, the capql file has been adapted to also create an equivariant policy.

- **src/**: This folder contains most of the files to run the models.
  - ***pareto_front/***:
    - main.py: file to run the gold standard dense or sparse rewards for CAPQL.
    - main_shaped.py: file to run the shaped rewards for CAPQL.
    - main_eq.py: file to run the shaped rewards suing equivariant policy.
  - ***reward_shaping/***:
    - architecture.py: file containing the neural networks for the reward shaper.
    - baseline.py: file to run the gold standard dense or sparse rewards for scalarised SAC.
    - env.py: wrapper to make the environments sparse.
    - reward_model.py: file to run the shaped rewards for scalarised SAC.

## Usage

- **Plotting a single policy:**  
  ```sh
  python plot_single_policy.py [args]
  ```

- **Plotting with Weights & Biases:**  
  ```sh
  python plot_wandb.py [args]
  ```


## Configuration

- Main configuration file: `configs/config.yaml`
- The shaped configs are as follows:

active_env: environment you want to use, which contains the following configs

env_hopper:
  env:
    name: name of the environment as defined by mo-gymnasium
    reward_type: use 'sparse', 'dense', or 'shaped' rewards
    sparsity_levels: sparsity levels to use for the sparse/shaped results, e.g., [1.0,0.0,0.0]
    reward_weights: reward weights used by scalarised SAC, e.g., [1.0, 0.0,1e-3]

  irl:
    initial_collection_episodes: number of episodes to train the reward shaper on at the start 
    expert_collection_episodes: number of expert episodes sued to update the reward shaper   
    num_refinement_cycles: number of times to update the reward shaper          
    refinement_timesteps: number of timestpes per cycle      
    nn_epochs: number of epochs to train the reward shaper on                     
    nn_lr: learning rate to use for the reward shaper
    ensemble_size: number of reward models to use
    reference: referecne point used for HV, e.g., [-100.0,-100.0,-100.0]
    use_dense: use dense rewards as features for the reward shaper
    use_residual: use residual architecture for the reward shaper
    use_enc: use symmetric state encoder for the equivariant policy
    lambda: loss parameter for the equivariance loss

