import gymnasium as gym
import mo_gymnasium as mo_gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
import time
import hydra
from omegaconf import DictConfig, OmegaConf
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from typing import List, Sequence

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from mo_gymnasium.wrappers import LinearReward
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
from morl_baselines.multi_policy.capql.capql_equivariance import CAPQL
import random

from src.reward_shaping.env import AsymmetricSparsityWrapper
from src.reward_shaping.reward_model import IRLRewardShaper, IRLShapingWrapper, set_all_seeds

def run_single_iterative_run(cfg: DictConfig, seed: int, run_id: int):

    print(f"--- Starting Iterative Run {run_id} with seed {seed} ---")

    set_all_seeds(seed)
    config = OmegaConf.to_container(cfg, resolve=True)
    device_name = config['rl_agent']['device']
    device = torch.device(device_name if device_name == 'cuda' and torch.cuda.is_available() else 'cpu')
    config['rl_agent']['device'] = str(device)
    log_dir = os.path.join(os.getcwd(), config['log_dir'], cfg.env.name, 'shaped', f"seed_{seed}")
    os.makedirs(log_dir, exist_ok=True)

    try:

        # Create the train and validation environments
        collection_env = AsymmetricSparsityWrapper(
            mo_gym.make(config['env']['name']),
            sparsity_levels=config['env']['sparsity_levels']
        )

        obs_dim = collection_env.observation_space.shape[0]
        act_dim = collection_env.action_space.shape[0]

        irl_shaper = IRLRewardShaper(config, obs_dim, act_dim)
        shaping_wrapper = IRLShapingWrapper(collection_env, irl_shaper)
        
        train_env = shaping_wrapper

        train_env.action_space.seed(seed)
        _ = train_env.reset(seed=seed)
        
        eval_env = mo_gym.make(config['env']['name'])

        eval_env.action_space.seed(seed+123)
        _ = eval_env.reset(seed=seed+123)

        agent = CAPQL(env=train_env, seed = seed, project_name=config['log_dir'], all_timesteps = int(config['irl']['num_refinement_cycles'] * config['irl']['refinement_timesteps']), lambda_loss = config['irl']['lambda'], use_enc= config['irl']['use_enc'])

        # Collec the random trjaecotries
        print(f"\n[Run {run_id}] Phase 0: Initial Random Data Collection")
        for _ in range(config['irl']['initial_collection_episodes']):

            obs, _ = collection_env.reset()
            done = False
            ep_obs, ep_true_dense, ep_action, cum_sparse_rew = [], [], [], 0.0

            while not done:
                action = collection_env.action_space.sample()
                next_obs, reward_vec, term, trunc, info = collection_env.step(action)
                done = term or trunc
                ep_obs.append(obs); ep_action.append(action); ep_true_dense.append(info['true_dense_rewards'])
                obs = next_obs

                # add datapoint when sparse reward is released, this is already a sum
                if reward_vec[irl_shaper.sparse_channel_idx] != 0:
                    cum_sparse_rew += reward_vec[irl_shaper.sparse_channel_idx]
                    irl_shaper.add_episode_data(ep_obs, ep_action, ep_true_dense, cum_sparse_rew)
                    ep_obs, ep_true_dense, ep_action, cum_sparse_rew = [], [], [], 0.0

        num_cycles = config['irl']['num_refinement_cycles']
        for cycle in range(num_cycles):
            print(f"\n[Run {run_id}] Cycle {cycle + 1}/{num_cycles}")

            # train reward model on data
 
            irl_shaper.train_reward_model_nn(
                epochs=config['irl']['nn_epochs'],
                run_id=run_id,
                val_split=config['irl'].get('val_split', 0.2),
                patience=config['irl'].get('early_stop_patience', 20)
            )

            # train Rl algo
            print(f"[Run {run_id}] Training SAC Agent...")
        
            agent.train(
                eval_env=eval_env,
                total_timesteps=config['irl']['refinement_timesteps'],
                ref_point=np.array(config['irl']['reference']),
                known_pareto_front=None,
            )
            
            print(f"[Run {run_id}] Collecting Expert Data...")
            
            # Collect expert trajectories like before but now using agnet policy
            for _ in range(config['irl']['expert_collection_episodes']):
                obs, _ = collection_env.reset()
                done = False
                ep_obs, ep_true_dense, ep_action, cum_sparse_rew = [], [], [], 0.0
                while not done:
                    action = agent.predict(obs)
                    next_obs, reward_vec, term, trunc, info = collection_env.step(action)
                    done = term or trunc

                    ep_obs.append(obs); ep_action.append(action); ep_true_dense.append(info['true_dense_rewards'])
                    obs = next_obs

                    if reward_vec[irl_shaper.sparse_channel_idx] != 0:
                        cum_sparse_rew += reward_vec[irl_shaper.sparse_channel_idx]
                        irl_shaper.add_episode_data(ep_obs, ep_action, ep_true_dense, cum_sparse_rew)
                        ep_obs, ep_true_dense, ep_action, cum_sparse_rew = [], [], [], 0.0

        success_msg = f"Run {run_id} (seed {seed}) completed successfully. Results in: {log_dir}"
        print(f"\n--- {success_msg} ---")
        return success_msg

    except Exception as e:
        error_msg = f"Run {run_id} (seed {seed}) failed: {str(e)}"
        print(f"\n--- {error_msg} ---")
        import traceback
        traceback.print_exc()
        return error_msg

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
config_path = os.path.join(root_dir, "configs")

@hydra.main(config_path=config_path, config_name="config", version_base='1.3')
def run_parallel_iterative_training(cfg: DictConfig):
    
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
            future = executor.submit(run_single_iterative_run, cfg, seed, i + 1)
            futures.append(future)
        
        print("\n Waiting for all training runs to complete...")
        for future in futures:
            try:
                result = future.result()
                print(f"✓ {result}")
            except Exception as e:
                print(f"A run failed with an unexpected error: {e}")
    print("\n All Parallel Training Runs Have Finished")

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    run_parallel_iterative_training()