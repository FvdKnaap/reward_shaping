import mo_gymnasium as mo_gym
import numpy as np
import torch  

import sys
import os

# Check correct file
from src.reward_shaping.env import AsymmetricSparsityWrapper
from morl_baselines.multi_policy.capql.capql import CAPQL


import os
import time
import hydra
from omegaconf import DictConfig, OmegaConf
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from typing import List, Sequence

def main(cfg: DictConfig, seed: int, run_id: int):
    print(f"--- Starting Iterative Run {run_id} with seed {seed} ---")

    # Set random seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Convert config to a mutable dict
    config = OmegaConf.to_container(cfg, resolve=True)

    env = mo_gym.make(config['env']['name'])
    eval_env = mo_gym.make(config['env']['name'])

    if config['env']['reward_type'] == 'sparse':
        env = AsymmetricSparsityWrapper(env, sparsity_levels=config['env']['sparsity_levels'])
   
    algo = CAPQL(env=env, seed = seed, project_name=config['log_dir'], all_timesteps = config['irl']['refinement_timesteps'])

    algo.train(
        eval_env=eval_env,
        total_timesteps=config['irl']['refinement_timesteps'],
        ref_point=np.array(config['irl']['reference']),
        known_pareto_front=None,
    )

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
config_path = os.path.join(root_dir, "configs")

@hydra.main(config_path=config_path, config_name="config", version_base='1.3')
def run_parallel_iterative_training(cfg: DictConfig):
    """ Main function that orchestrates parallel iterative training runs. """
    cfg = cfg.shaping[cfg.shaping.active_env]
    print("--- Parallel Iterative Refinement Training ---")
    print(OmegaConf.to_yaml(cfg))
    print("------------------------------------------\n")

    base_seed = cfg.seed
    num_runs = cfg.num_parallel_runs
    seeds = [base_seed + i for i in range(num_runs)]
    
    print(f"Starting {num_runs} parallel experiments with seeds: {seeds}")
    
    with ProcessPoolExecutor(max_workers=num_runs) as executor:
        futures = []
        for i, seed in enumerate(seeds):
            future = executor.submit(main, cfg, seed, i + 1)
            futures.append(future)
        
        print("\nWaiting for all training runs to complete...")
        for future in futures:
            try:
                result = future.result()
                print(f"✓ {result}")
            except Exception as e:
                print(f"A run failed with an unexpected error: {e}")
    
    print("\n--- All Parallel Training Runs Have Finished ---")

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    run_parallel_iterative_training()
