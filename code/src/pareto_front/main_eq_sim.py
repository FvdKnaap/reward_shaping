import gymnasium as gym
import mo_gymnasium as mo_gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
import json
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

import wandb
from src.reward_shaping.env import AsymmetricSparsityWrapper, Walker2dRealityGapWrapper
from src.reward_shaping.reward_model import IRLRewardShaper, IRLShapingWrapper, set_all_seeds

def make_sim_env(config: dict, *, training: bool):
    rg = config['sim_gap']
    env = mo_gym.make(config['env']['name'])

    return Walker2dRealityGapWrapper(
        env = env,
        action_delay_steps=rg['action_delay_steps'],
        randomize_actuator_gain=training,
        actuator_gain_low=rg['actuator_gain_low'],
        actuator_gain_high= rg['actuator_gain_high'],
        observation_noise_std= rg.get('observation_noise_std', 0.0),
        record_clean_observation= rg.get('record_clean_observation', False)
    )

def run_single_iterative_run(cfg: DictConfig, seed: int, run_id: int):

    print(f"--- Starting Iterative Run {run_id} with seed {seed} ---")

    set_all_seeds(seed)
    config = OmegaConf.to_container(cfg, resolve=True)
    device_name = config['rl_agent']['device']
    device = torch.device(device_name if device_name == 'cuda' and torch.cuda.is_available() else 'cpu')
    config['rl_agent']['device'] = str(device)
    log_dir = os.path.join(os.getcwd(), config['log_dir'], cfg.env.name, 'shaped', f"seed_{seed}")
    os.makedirs(log_dir, exist_ok=True)

    # Create a local checkpoints directory
    checkpoint_dir = os.path.join(log_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    try:

        #base_collection_env = make_sim_env(config= config, training= True)

        # Create the train and validation environments
        #collection_env = AsymmetricSparsityWrapper(
        #    base_collection_env,
        #    sparsity_levels=config['env']['sparsity_levels']
        #)

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
        
        eval_env = mo_gym.make(config['env']['name']) #make_sim_env(config=config, training=True)

        eval_env.action_space.seed(seed+123)
        _ = eval_env.reset(seed=seed+123)

        agent = CAPQL(
            env=train_env,
            seed=seed,
            project_name=config['log_dir'],
            all_timesteps=int(
                config['irl']['num_refinement_cycles']
                * config['irl']['refinement_timesteps']
            ),
            lambda_loss=config['irl']['lambda'],
        )

        # CAPQL's train() closes W&B when global_step == all_timesteps.  In an
        # iterative run that happens INSIDE the final cycle, before this file
        # gets a chance to upload the final cycle artifact.  Keep the run open
        # here and close it explicitly after all cycle artifacts are logged.
        _capql_close_wandb = agent.close_wandb
        agent.close_wandb = lambda: None

        if wandb.run is None:
            raise RuntimeError("CAPQL did not initialize an active W&B run.")

        # Persist the PRISM/shaper settings in the W&B run itself so future
        # evaluation does not have to infer sparsity from launch order.
        release_probs = [1.0 - float(s) for s in config['env']['sparsity_levels']]
        wandb.config.update(
            {
                "prism_env_name": config['env']['name'],
                "prism_seed": int(seed),
                "prism_run_id": int(run_id),
                "prism_sparsity_levels": list(config['env']['sparsity_levels']),
                "prism_reward_release_probs": release_probs,
                "prism_num_refinement_cycles": int(config['irl']['num_refinement_cycles']),
                "prism_refinement_timesteps": int(config['irl']['refinement_timesteps']),
                "prism_ensemble_size": int(config['irl']['ensemble_size']),
                "prism_use_dense": bool(config['irl']['use_dense']),
                "prism_use_residual": bool(config['irl']['use_residual']),
                "prism_nn_lr": float(config['irl']['nn_lr']),
                "prism_lambda": float(config['irl']['lambda']),
                "prism_full_config": config,
            },
            allow_val_change=True,
        )

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

            # SAVE RESYMNET LOCALLY
            resymnet_save_path = os.path.join(checkpoint_dir, f"cycle_{cycle+1}_resymnet")
            irl_shaper.save_reward_model(save_dir=resymnet_save_path)

            # train Rl algo
            print(f"[Run {run_id}] Training SAC Agent...")
        
            agent.train(
                eval_env=eval_env,
                total_timesteps=config['irl']['refinement_timesteps'],
                ref_point=np.array(config['irl']['reference']),
                known_pareto_front=None,
            )

            # SAVE CAPQL AGENT LOCALLY
            capql_save_path = os.path.join(checkpoint_dir, f"cycle_{cycle+1}_capql")
            agent.save(save_dir=capql_save_path, filename="capql_policy")

            # --- C. UPLOAD TO W&B ARTIFACTS ---
            if wandb.run is None:
                raise RuntimeError(
                    f"W&B run is closed before cycle {cycle+1} artifact upload."
                )

            # Save explicit cycle metadata alongside the model files.
            cycle_metadata = {
                "wandb_run_id": wandb.run.id,
                "seed": int(seed),
                "local_run_id": int(run_id),
                "cycle": int(cycle + 1),
                "env_name": config['env']['name'],
                "sparsity_levels": list(config['env']['sparsity_levels']),
                "reward_release_probs": [
                    1.0 - float(s) for s in config['env']['sparsity_levels']
                ],
                "sparse_channel_idx": int(irl_shaper.sparse_channel_idx),
                "dense_channel_indices": [
                    int(i) for i in irl_shaper.dense_channel_indices
                ],
                "ensemble_size": int(irl_shaper.ensemble_size),
                "feature_dim": int(irl_shaper.feature_dim),
                "refinement_timesteps": int(config['irl']['refinement_timesteps']),
                "global_step": int(agent.global_step),
            }
            metadata_path = os.path.join(
                checkpoint_dir, f"cycle_{cycle+1}_metadata.json"
            )
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(cycle_metadata, f, indent=2)

            # W&B run ID makes artifact names unique across separate launches;
            # seed/run_id alone are reused in your current experiment pattern.
            artifact = wandb.Artifact(
                name=f"prism-{wandb.run.id}-cycle-{cycle+1}-models",
                type="model",
                description=(
                    f"ReSymNet and CAPQL models for seed {seed}, "
                    f"cycle {cycle+1}"
                ),
                metadata=cycle_metadata,
            )
            artifact.add_dir(resymnet_save_path, name="resymnet")
            artifact.add_dir(capql_save_path, name="capql")
            artifact.add_file(metadata_path, name="cycle_metadata.json")

            logged_artifact = wandb.log_artifact(artifact)
            # Make persistence explicit before continuing/finishing the run.
            logged_artifact.wait()
            print(
                f"[Run {run_id}] Uploaded cycle {cycle+1} artifact: "
                f"{artifact.name}:{logged_artifact.version}"
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

        # CAPQL was deliberately prevented from closing W&B inside its final
        # train() call. All model artifacts are now safely uploaded, so close.
        if wandb.run is not None:
            _capql_close_wandb()

        success_msg = f"Run {run_id} (seed {seed}) completed successfully. Results in: {log_dir}"
        print(f"\n--- {success_msg} ---")
        return success_msg

    except Exception as e:
        if wandb.run is not None:
            wandb.finish()
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