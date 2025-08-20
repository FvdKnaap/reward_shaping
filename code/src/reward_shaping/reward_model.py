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
from src.reward_shaping.env import AsymmetricSparsityWrapper
import random
from src.reward_shaping.architecture import RewardNet, RewardNet_No_Res, init_weights

def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class TrajectoryDataset(Dataset):
    def __init__(self, trajectory_data, device):
        self.trajectory_data = trajectory_data
        self.device = device

    def __len__(self):
        return len(self.trajectory_data)

    def __getitem__(self, idx):
        ep_features, target_g = self.trajectory_data[idx]
        return (
            torch.tensor(np.array(ep_features), dtype=torch.float32),
            torch.tensor(target_g, dtype=torch.float32)
        )

def collate_trajectories(batch, normalizer=None):
    features = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    
    # Normalise features
    features_flat = [step for ep in features for step in ep]
    if normalizer is not None:
        features_flat = normalizer(np.array(features_flat))
    else:
        features_flat = np.array(features_flat)

    ep_lengths = [len(ep) for ep in features]
    idx = 0
    features_normed = []
    for L in ep_lengths:
        features_normed.append(torch.tensor(features_flat[idx:idx+L], dtype=torch.float32))
        idx += L
    
    # Pads to longest trajectory using 0
    features_padded = pad_sequence(features_normed, batch_first=True, padding_value=0.0)
    lengths = [f.shape[0] for f in features_normed]
    max_len = features_padded.size(1)
    masks = torch.zeros((len(features), max_len))

    # Save mask, 1 indicates a real step
    for i, l in enumerate(lengths):
        masks[i, :l] = 1

    targets_tensor = torch.stack(targets)

    return features_padded, targets_tensor, masks



class IRLRewardShaper:
    """ A reward shaper that uses kind of Inverse Reinforcement Learning (IRL) to shape rewards based on expert demonstrations.
    uses a nueral network to predict the shaped reward based on the features extracted from the state, action, and dense rewards.
    """

    def __init__(self, config: dict, obs_dim: int, action_dim: int):

        self.config = config
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        device = config['rl_agent']['device']

        if device == 'cuda' and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        
        # Identify dense and sparse channels based on sparsity levels
        self.dense_channel_indices = [i for i, s in enumerate(config['env']['sparsity_levels']) if s == 0.0]
        self.sparse_channel_idx = [i for i, s in enumerate(config['env']['sparsity_levels']) if s > 0.0][0]

        if config['irl']['use_dense']:
            self.feature_dim = obs_dim + action_dim + len(self.dense_channel_indices)
        else:
            self.feature_dim = obs_dim + action_dim

        self.num_objectives = len(self.dense_channel_indices) + 1
        self.trajectory_data = []
        self.feature_scaler = StandardScaler()
        self.scaler_fitted = False
        
        self.ensemble_size = self.config['irl'].get('ensemble_size', 1)

        if config['irl']['use_residual']:
            self.reward_nets = [RewardNet(self.feature_dim).to(self.device) for _ in range(self.ensemble_size)]
        else:
            self.reward_nets = [RewardNet_No_Res(self.feature_dim).to(self.device) for _ in range(self.ensemble_size)]
        self.nn_optimizers = [optim.AdamW(net.parameters(), lr=self.config['irl']['nn_lr']) for net in self.reward_nets]


    def _get_features(self, obs: np.ndarray, action: np.ndarray, true_dense_rewards: np.ndarray) -> np.ndarray:
        """ Extracts features from the observation, action, and true dense rewards.
        The features include the observation, action, and dense rewards.
        """
        dense_features = true_dense_rewards[self.dense_channel_indices]
        return np.concatenate([obs, action, dense_features])

    def fit_feature_scaler(self, data):
        """ Fits the feature scaler on the provided trajectory data.
        The data should be a list of tuples where each tuple contains per-step features and cumulative sparse reward.
        """
        all_features = []
        for per_step_features, _ in data:
            all_features.extend(per_step_features)
        all_features = np.array(all_features)
        self.feature_scaler.fit(all_features)
        self.scaler_fitted = True
    
    def reset(self):
        self.trajectory_data = []

    def add_episode_data(self, episode_obs: List, episode_action: List, episode_true_dense_rewards: List, cumulative_sparse_reward: float):
        """ Adds an episode's data to the trajectory data.
        This includes the observations, actions, true dense rewards, and cumulative sparse reward.
        """
        per_step_features = [self._get_features(o, a, dr) for o, a, dr in zip(episode_obs, episode_action, episode_true_dense_rewards)]
        if per_step_features:
            self.trajectory_data.append((per_step_features, cumulative_sparse_reward))

    def train_reward_model_nn(self, epochs=50, run_id=0, val_split=0.3, patience=30):
        """ Trains a neural network ensemble to predict the shaped reward based on the trajectory data.
        The training uses early stopping based on validation loss.
        """
        if not self.trajectory_data:
            print(f"[Run {run_id}] No trajectory data to train reward model. Skipping.")
            return
        
        # Reset the reward nets and optimizers to retrain on entire dataset
        self.reward_nets = [
            RewardNet(self.feature_dim).to(self.device)
           for _ in range(self.ensemble_size)
        ]
        
        self.nn_optimizers = [optim.AdamW(net.parameters(), lr=self.config['irl']['nn_lr']) for net in self.reward_nets]
        
        # Split into train/val
        total_len = len(self.trajectory_data)
        val_len = int(total_len * val_split)
        train_len = total_len - val_len
        trajectory_dataset = TrajectoryDataset(self.trajectory_data, self.device)
        train_set, val_set = random_split(trajectory_dataset, [train_len, val_len], generator=torch.Generator().manual_seed(self.config['seed']))
        # Fit the feature scaler on the training set
        self.fit_feature_scaler(train_set)

        # Create data loaders
        train_loader = DataLoader(
            train_set,
            batch_size=32,
            shuffle=True,
            collate_fn=lambda batch: collate_trajectories(batch, normalizer=self.feature_scaler.transform),
            worker_init_fn=lambda worker_id: np.random.seed(self.config['seed'] + worker_id),
            generator=torch.Generator().manual_seed(self.config['seed'])
        )
        val_loader = DataLoader(
            val_set,
            batch_size=32,
            shuffle=False,
            collate_fn=lambda batch: collate_trajectories(batch, normalizer=self.feature_scaler.transform),
            worker_init_fn=lambda worker_id: np.random.seed(self.config['seed'] + worker_id),
            generator=torch.Generator().manual_seed(self.config['seed'])
        )

        print(f"\n[Run {run_id}] Training NN ensemble ({self.ensemble_size} models) with early stopping")
        # Train each model in the ensemble
        for model_idx in range(self.ensemble_size):

            net = self.reward_nets[model_idx]
            optimizer = self.nn_optimizers[model_idx]
            net.train()

            # Initialize the learning rate scheduler
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
            best_val_loss = float('inf')
            epochs_no_improve = 0

            best_model_state = None

            # Training loop
            for epoch in range(epochs):
                total_loss = 0

                # Iterate over the training data
                for ep_features_batch, target_g_batch, masks in train_loader:

                    features_tensor = ep_features_batch.to(self.device)
                    target_g_tensor = target_g_batch.to(self.device)
                    mask_tensor = masks.to(self.device)

                    # Forward pass
                    per_step_preds = net(features_tensor).squeeze(-1)
                    masked_preds = per_step_preds * mask_tensor
                    predicted_g = masked_preds.sum(dim=1)

                    loss = nn.functional.mse_loss(predicted_g, target_g_tensor)

                    optimizer.zero_grad()
                    loss.backward()

                    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                    optimizer.step()
                    total_loss += loss.item()

                # Step the scheduler   
                scheduler.step()
                avg_loss = total_loss / len(train_loader)

                # Validation
                net.eval()
                with torch.no_grad():
                    val_loss = 0
                    for ep_features_batch, target_g_batch, masks in val_loader:

                        features_tensor = ep_features_batch.to(self.device)
                        target_g_tensor = target_g_batch.to(self.device)
                        mask_tensor = masks.to(self.device)

                        per_step_preds = net(features_tensor).squeeze(-1)
                        masked_preds = per_step_preds * mask_tensor
                        predicted_g = masked_preds.sum(dim=1)

                        loss = nn.functional.mse_loss(predicted_g, target_g_tensor)
                        val_loss += loss.item()

                    avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0

                print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

                # Early stopping
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    epochs_no_improve = 0
                    best_model_state = net.state_dict()
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= patience:
                        print(f"Early stopping at epoch {epoch+1} for model {model_idx+1}")
                        break

                net.train()

            if best_model_state:
                print(f"Loading best model for model {model_idx+1} with validation loss: {best_val_loss:.4f}")
                net.load_state_dict(best_model_state)
            net.eval()

        print(f"[Run {run_id}] Neural net ensemble training complete.")

    def get_shaped_reward(self, obs: np.ndarray, action: np.ndarray, true_dense_rewards: np.ndarray) -> float:
        """ Computes the shaped reward based on the observation, action, and true dense rewards.
        If the neural network is used, it predicts the shaped reward based on the features extracted from the state, action, and dense rewards.
        """
        features = self._get_features(obs, action, true_dense_rewards)
        
        predictions = []
        if self.scaler_fitted:
            normed_features = self.feature_scaler.transform([features])[0]
        else:
            normed_features = features

        features_tensor = torch.tensor(normed_features, dtype=torch.float32, device=self.device)

        for net in self.reward_nets:
            net.eval()
            with torch.no_grad():
                shaped_reward = net(features_tensor.unsqueeze(0)).item()
                predictions.append(shaped_reward)

        return np.mean(predictions)
        

class IRLShapingWrapper(gym.Wrapper):
    """ A wrapper that applies the IRL reward shaping to the environment.
    It uses the IRLRewardShaper to compute the shaped reward based on the last observation, action, and true dense rewards.
    The shaped reward is then combined with the true dense rewards to form the final reward vector.
    """
    def __init__(self, env: gym.Env, shaper: IRLRewardShaper):

        super().__init__(env)
        self.shaper = shaper
        self.num_objectives = shaper.num_objectives
        self.sparse_channel_idx = shaper.sparse_channel_idx
        self.last_obs = None
        self.last_true_dense_rewards = None

    def step(self, action: np.ndarray):
        """
        Applies the IRL reward shaping to the environment step.
        """

        next_obs, reward_vec, terminated, truncated, info = self.env.step(action)
        true_dense_rewards = info['true_dense_rewards']

        self.last_true_dense_rewards = true_dense_rewards

        shaped_reward = self.shaper.get_shaped_reward(self.last_obs, action, self.last_true_dense_rewards)

        final_reward_vec = np.copy(true_dense_rewards)
        final_reward_vec[self.sparse_channel_idx] = shaped_reward
        self.last_obs = next_obs
        
        return next_obs, final_reward_vec, terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self.last_obs = obs
        self.last_true_dense_rewards = np.zeros(self.shaper.num_objectives)
        return obs, info

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
        collection_env = AsymmetricSparsityWrapper(
            mo_gym.make(config['env']['name']),
            sparsity_levels=config['env']['sparsity_levels']
        )

        obs_dim = collection_env.observation_space.shape[0]
        act_dim = collection_env.action_space.shape[0]

        irl_shaper = IRLRewardShaper(config, obs_dim, act_dim)
        shaping_wrapper = IRLShapingWrapper(collection_env, irl_shaper)

        train_env = LinearReward(shaping_wrapper, weight=np.array(config['env']['reward_weights']))
        train_env = Monitor(train_env, log_dir)


        _ = train_env.reset(seed=seed)
        
        eval_env = Monitor(LinearReward(mo_gym.make(config['env']['name']), weight=np.array(config['env']['reward_weights'])))

        _ = eval_env.reset(seed=seed+123)

        agent = SAC(
            "MlpPolicy", train_env, verbose=1, gamma=config['rl_agent']['gamma'],
            learning_starts=config['rl_agent']['learning_starts'], device=config['rl_agent']['device'], seed=seed
        )

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
                if reward_vec[irl_shaper.sparse_channel_idx] != 0:
                    cum_sparse_rew += reward_vec[irl_shaper.sparse_channel_idx]
                    irl_shaper.add_episode_data(ep_obs, ep_action, ep_true_dense, cum_sparse_rew)
                    ep_obs, ep_true_dense, ep_action, cum_sparse_rew = [], [], [], 0.0

        num_cycles = config['irl']['num_refinement_cycles']
        for cycle in range(num_cycles):
            print(f"\n[Run {run_id}] Cycle {cycle + 1}/{num_cycles}")

           
            irl_shaper.train_reward_model_nn(
                epochs=config['irl']['nn_epochs'],
                run_id=run_id,
                val_split=config['irl'].get('val_split', 0.2),
                patience=config['irl'].get('early_stop_patience', 20)
            )

            print(f"[Run {run_id}] Training SAC Agent...")
            eval_callback = EvalCallback(eval_env, best_model_save_path=f"{log_dir}/cycle_{cycle+1}",
                                         log_path=f"{log_dir}/cycle_{cycle+1}", eval_freq=config['rl_agent']['eval_freq'],
                                         deterministic=True, render=False)

            agent.learn(total_timesteps=config['irl']['refinement_timesteps'],
                        callback=eval_callback, progress_bar=False, reset_num_timesteps=False)
            
            print(f"[Run {run_id}] Collecting Expert Data...")
        
            for _ in range(config['irl']['expert_collection_episodes']):
                obs, _ = collection_env.reset()
                done = False
                ep_obs, ep_true_dense, ep_action, cum_sparse_rew = [], [], [], 0.0
                while not done:
                    action, _ = agent.predict(obs, deterministic=True)
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
        print("\nWaiting for all training runs to complete...")
        for future in futures:
            try:
                result = future.result()
                print(f"{result}")
            except Exception as e:
                print(f"A run failed with an unexpected error: {e}")
    print("\nAll Parallel Training Runs Have Finished")

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    run_parallel_iterative_training()