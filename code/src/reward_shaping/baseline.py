import gymnasium as gym
import mo_gymnasium as mo_gym
import numpy as np
import pickle
import hydra
from omegaconf import DictConfig, OmegaConf
import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import torch

from scipy.spatial import KDTree
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from mo_gymnasium.wrappers import LinearReward
# Make sure this import path is correct for the project structure
from src.reward_shaping.env import AsymmetricSparsityWrapper

def train_single_agent(cfg: DictConfig, seed: int, run_id: int):
    """
    Train a single agent with a specific seed and run ID.
    This function will be called by each parallel process.
    """
    print(f"--- Starting Run {run_id} with seed {seed} ---")
    
    # Set the seed for this specific run
    cfg.seed = seed
    reward_type = cfg.env.reward_type
    
    # Set random seeds
    np.random.seed(seed)
    torch.manual_seed(seed)

    base_env = mo_gym.make(cfg.env.name)

    if reward_type == 'sparse':
        print(f"Using sparse rewards with sparsity levels: {cfg.env.sparsity_levels}")
        sparse_wrapped_env = AsymmetricSparsityWrapper(base_env, sparsity_levels=list(cfg.env.sparsity_levels))
        train_env = LinearReward(sparse_wrapped_env, weight=np.array(list(cfg.env.reward_weights)))

    elif reward_type == 'dense':
        print("Using dense (gold standard) rewards.")
        train_env = LinearReward(base_env, weight=np.array(list(cfg.env.reward_weights)))
        
    else:
        raise ValueError(f"Unknown reward_type: '{reward_type}'. Must be 'sparse', or 'dense'.")

    train_env.action_space.seed(seed)
    _ = train_env.reset(seed=seed)

    eval_env_base = mo_gym.make(cfg.env.name)
    eval_env = LinearReward(eval_env_base, weight=np.array(list(cfg.env.reward_weights)))
    eval_env = Monitor(eval_env)
    
    eval_env.action_space.seed(seed+123)
    _ = eval_env.reset(seed=seed+123)

    # Create a unique log path for each experiment type AND seed to keep results separate
    log_dir_for_run = os.path.join(os.getcwd(), config['log_dir'], cfg.env.name, reward_type, f"seed_{seed}")
    
    os.makedirs(log_dir_for_run, exist_ok=True)
    eval_callback = EvalCallback(
        eval_env, 
        best_model_save_path=os.path.join(log_dir_for_run, "best_model"),
        log_path=log_dir_for_run, 
        eval_freq=cfg.rl_agent.eval_freq,
        deterministic=True, 
        render=False
    )

    if cfg.rl_agent.device == 'cuda':
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    else:
        device = torch.device('cpu')

    if cfg.rl_agent.rl_algorithm.upper() == "SAC":
        agent = SAC("MlpPolicy", train_env, verbose=1, device=device, seed=seed, learning_starts=cfg.rl_agent.learning_starts)
    else:
        raise ValueError(f"Unsupported RL algorithm: {cfg.rl_agent.rl_algorithm}")

    print(f"\n--- Starting training for '{reward_type}' setting (Run {run_id}, Seed {seed}) for {cfg.rl_agent.total_timesteps} timesteps ---")
    agent.learn(total_timesteps=cfg.rl_agent.total_timesteps, callback=eval_callback)
    
    print(f"--- Training Complete for Run {run_id} ---")
    print(f"Best model and logs saved in: {log_dir_for_run}")
    
    return f"Run {run_id} (seed {seed}) completed successfully. Results saved in: {log_dir_for_run}"


# This may have to be adapted depending on where your config file is
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
config_path = os.path.join(root_dir, "configs")

@hydra.main(config_path=config_path, config_name="config", version_base='1.3')
def evaluate_agent_parallel(cfg: DictConfig):
    cfg = cfg.shaping[cfg.shaping.active_env]
    print("--- Parallel Evaluation Configuration ---")
    print(OmegaConf.to_yaml(cfg))
    print("---------------------------------\n")

    # Define the seeds for parallel runs
    base_seed = cfg.seed
    seeds = [base_seed + i for i in range(cfg.num_parallel_runs)] 
    
    print(f"Running {cfg.num_parallel_runs} parallel experiments with seeds: {seeds}")
    
    # Using ProcessPoolExecutor
    print("Starting parallel training using ProcessPoolExecutor...")
    with ProcessPoolExecutor(max_workers=cfg.num_parallel_runs) as executor:
        # Submit all jobs
        futures = []
        for i, seed in enumerate(seeds):
            
            cfg_copy = OmegaConf.create(OmegaConf.to_yaml(cfg))
            future = executor.submit(train_single_agent, cfg_copy, seed, i+1)
            futures.append(future)
        
        # Collect results as they complete
        print("Waiting for all training runs to complete...")
        for i, future in enumerate(futures):
            try:
                result = future.result()  
                print(f"{result}")
            except Exception as e:
                print(f"Run {i+1} failed with error: {e}")
    
    print("\n All Parallel Training Runs Complete")


if __name__ == "__main__":
    # Set start method for multiprocessing (important for some systems)
    mp.set_start_method('spawn', force=True)
    evaluate_agent_parallel()