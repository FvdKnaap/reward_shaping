"""Walker2d reward-reconstruction feature ablation.

This is a separate experiment entry point. It leaves the normal PRISM pipeline
unchanged while reusing the same environment wrapper, CAPQL implementation,
reward-model architecture, trajectory-return training objective, scaler,
ensemble training, and early stopping.

Expected experiment-specific reward model module:
    src/experiments/reward_model.py

That copied IRLRewardShaper must support:
    config['irl']['removed_observation_indices']
    evaluate_reconstruction(held_out_trajectories)

The evaluator is called with a list of tuples:
    (observations, actions, true_dense_rewards)

Only the full-observation shaper is used to shape CAPQL. The ablated shaper is
trained in parallel on exactly the same trajectories but never affects policy
training. Final test trajectories are newly collected and are never passed to
add_episode_data for either shaper.
"""

from __future__ import annotations

import copy
import csv
import os
import random
from pathlib import Path
from typing import Callable

import hydra
import mo_gymnasium as mo_gym
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from morl_baselines.multi_policy.capql.capql_equivariance import CAPQL
from src.experiments.reward_model import (
    IRLRewardShaper,
    IRLShapingWrapper,
    set_all_seeds,
)
from src.reward_shaping.env import AsymmetricSparsityWrapper


def _capture_rng_state() -> dict:
    """Capture RNG state so the auxiliary ablation does not alter CAPQL RNG."""

    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _collect_episode(
    env,
    sparse_channel_idx: int,
    action_fn: Callable[[np.ndarray], np.ndarray],
    reset_seed: int | None = None,
) -> list[dict]:
    """Collect episode trajectories, splitting into segments on non-zero sparse rewards."""

    obs, _ = env.reset(seed=reset_seed)
    terminated = False
    truncated = False

    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    true_dense_rewards: list[np.ndarray] = []
    cumulative_sparse_reward = 0.0

    segments: list[dict] = []

    while not (terminated or truncated):
        action = np.asarray(action_fn(obs), dtype=np.float32)

        next_obs, reward_vec, terminated, truncated, info = env.step(action)

        observations.append(np.asarray(obs, dtype=np.float32).copy())
        actions.append(action.copy())
        true_dense_rewards.append(
            np.asarray(info["true_dense_rewards"], dtype=np.float32).copy()
        )
        obs = next_obs

        # Check for sparse reward release mid-episode
        sparse_reward = float(reward_vec[sparse_channel_idx])
        if sparse_reward != 0:
            cumulative_sparse_reward += sparse_reward
            segments.append(
                {
                    "observations": np.asarray(observations, dtype=np.float32),
                    "actions": np.asarray(actions, dtype=np.float32),
                    "true_dense_rewards": np.asarray(true_dense_rewards, dtype=np.float32),
                    "cumulative_sparse_reward": float(cumulative_sparse_reward),
                }
            )
            observations, actions, true_dense_rewards = [], [], []
            cumulative_sparse_reward = 0.0

    return segments


def _add_episode_to_shapers(segments: list[dict], shapers: list[IRLRewardShaper]) -> None:
    """Give every trajectory segment in the episode to each shaper."""

    for segment in segments:
        for shaper in shapers:
            shaper.add_episode_data(
                episode_obs=segment["observations"],
                episode_action=segment["actions"],
                episode_true_dense_rewards=segment["true_dense_rewards"],
                cumulative_sparse_reward=segment["cumulative_sparse_reward"],
            )


def _make_shaper_config(
    base_config: dict,
    removed_observation_indices: list[int],
) -> dict:
    config = copy.deepcopy(base_config)
    config["irl"]["use_dense"] = True
    config["irl"]["removed_indices"] = list(removed_observation_indices)
    return config


def _train_auxiliary_shaper_without_changing_main_rng(
    shaper: IRLRewardShaper,
    config: dict,
    seed: int,
    cycle: int,
) -> None:
    """Train the ablated model while preserving the main pipeline RNG state."""

    main_rng_state = _capture_rng_state()
    try:
        set_all_seeds(seed + 100_000 + cycle)
        shaper.train_reward_model_nn(
            epochs=int(config["irl"]["nn_epochs"]),
            run_id=f"no_forward_velocity_cycle_{cycle + 1}",
            val_split=float(config["irl"].get("val_split", 0.2)),
            patience=int(config["irl"].get("early_stop_patience", 20)),
        )
    finally:
        _restore_rng_state(main_rng_state)


def _collect_held_out_episodes(
    env,
    agent: CAPQL,
    sparse_channel_idx: int,
    num_episodes: int,
    seed: int,
) -> list[dict]:
    """Collect fresh final-policy trajectory segments not used by either reward model."""

    held_out: list[dict] = []
    set_all_seeds(seed + 200_000)

    for episode_idx in range(num_episodes):
        segments = _collect_episode(
            env=env,
            sparse_channel_idx=sparse_channel_idx,
            action_fn=agent.predict,
            reset_seed=seed + 300_000 + episode_idx,
        )
        held_out.extend(segments)

    return held_out


def _evaluation_payload(episodes: list[dict]) -> list[tuple[np.ndarray, ...]]:
    """Format expected by IRLRewardShaper.evaluate_reconstruction()."""

    return [
        (
            episode["observations"],
            episode["actions"],
            episode["true_dense_rewards"],
        )
        for episode in episodes
    ]


def _write_summary(rows: list[dict], output_path: Path) -> None:
    if not rows:
        raise ValueError("No result rows were produced.")

    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_single_nn(cfg: DictConfig, seed: int) -> Path:
    """Run one seed of the full-versus-no-forward-velocity study."""

    set_all_seeds(seed)
    config = OmegaConf.to_container(cfg, resolve=True)
    config["seed"] = seed

    device_name = str(config["rl_agent"]["device"])

    if device_name == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    config["rl_agent"]["device"] = str(device)

    if config["env"]["name"] != "mo-walker2d-v5":
        raise ValueError(
            "This first feature-ablation runner is restricted to mo-walker2d-v5."
        )

    sparsity_levels = list(config["env"]["sparsity_levels"])
    sparse_indices = [i for i, sparsity in enumerate(sparsity_levels) if sparsity > 0.0]
    if len(sparse_indices) != 1:
        raise ValueError("Expected exactly one sparse reward objective.")

    collection_env = AsymmetricSparsityWrapper(
        mo_gym.make(config["env"]["name"]),
        sparsity_levels=sparsity_levels,
    )
    eval_env = mo_gym.make(config["env"]["name"])

    eval_env.action_space.seed(seed + 123)
    eval_env.reset(seed=seed + 123)

    obs_dim = int(collection_env.observation_space.shape[0])
    action_dim = int(collection_env.action_space.shape[0])

    full_config = _make_shaper_config(config, removed_observation_indices=[])

    full_shaper = IRLRewardShaper(full_config, obs_dim, action_dim)
    shaped_train_env = IRLShapingWrapper(collection_env, full_shaper)

    # Constructing the auxiliary model must not perturb the main run's RNG.
    main_rng_state = _capture_rng_state()
    no_velocity_config = _make_shaper_config(
        config,
        removed_observation_indices=config["irl"].get("removed_indices", []),
    )
    no_velocity_shaper = IRLRewardShaper(
        no_velocity_config,
        obs_dim,
        action_dim,
    )

    _restore_rng_state(main_rng_state)

    shapers = [full_shaper, no_velocity_shaper]
    sparse_channel_idx = full_shaper.sparse_channel_idx

    initial_episodes = int(config["irl"]["initial_collection_episodes"])
    print(f"Collecting {initial_episodes} initial random trajectories...")
    for _ in range(initial_episodes):
        segments = _collect_episode(
            env=collection_env,
            sparse_channel_idx=sparse_channel_idx,
            action_fn=lambda _obs: collection_env.action_space.sample(),
        )

        _add_episode_to_shapers(segments, shapers)

    full_shaper.train_reward_model_nn(
        epochs=int(config["irl"]["nn_epochs"]),
        run_id=f"full_cycle_{1}",
        val_split=float(config["irl"].get("val_split", 0.2)),
        patience=int(config["irl"].get("early_stop_patience", 20)),
    )

    _train_auxiliary_shaper_without_changing_main_rng(
        shaper=no_velocity_shaper,
        config=no_velocity_config,
        seed=seed,
        cycle=1,
    )

    held_out_count = int(config["irl"].get("reconstruction_test_episodes", 200))
    print(
        f"\nCollecting {held_out_count} fresh held-out trajectories. "
        "These are not added to either reward model."
    )
    held_out_episodes: list[dict] = []
    for episode_idx in range(held_out_count):
        segments = _collect_episode(
            env=collection_env,
            sparse_channel_idx=sparse_channel_idx,
            action_fn=lambda _obs: collection_env.action_space.sample(),
            reset_seed=seed + 300_000 + episode_idx,
        )

        held_out_episodes.extend(segments)

    payload = held_out_episodes

    full_metrics = full_shaper.evaluate_reconstruction(payload)
    no_velocity_metrics = no_velocity_shaper.evaluate_reconstruction(payload)

    if not isinstance(full_metrics, dict) or not isinstance(no_velocity_metrics, dict):
        raise TypeError("evaluate_reconstruction() must return a metrics dictionary.")

    result_rows = [
        {
            "seed": seed,
            "condition": "full_observation",
            "removed_observation_indices": "[]",
            "feature_dim": int(full_shaper.feature_dim),
            "training_trajectories": len(full_shaper.trajectory_data),
            "held_out_trajectories": held_out_count,
            **full_metrics,
        },
        {
            "seed": seed,
            "condition": "without_forward_velocity",
            "removed_observation_indices": f"[{config['irl'].get('removed_indices', [])}]",
            "feature_dim": int(no_velocity_shaper.feature_dim),
            "training_trajectories": len(no_velocity_shaper.trajectory_data),
            "held_out_trajectories": held_out_count,
            **no_velocity_metrics,
        },
    ]

    root_dir = Path(__file__).resolve().parents[2]
    output_dir = root_dir / "outputs" / "reconstruction_ablation" / f"seed_{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "summary.csv"
    _write_summary(result_rows, summary_path)

    resolved_config_path = output_dir / "resolved_config.yaml"
    resolved_config_path.write_text(
        OmegaConf.to_yaml(OmegaConf.create(config)),
        encoding="utf-8",
    )

    collection_env.close()
    eval_env.close()

    print(f"\nSaved reconstruction results to {summary_path}")
    return summary_path


def run_single_ablation(cfg: DictConfig, seed: int) -> Path:
    """Run one seed of the full-versus-no-forward-velocity study."""

    set_all_seeds(seed)
    config = OmegaConf.to_container(cfg, resolve=True)
    config["seed"] = seed

    device_name = str(config["rl_agent"]["device"])
    if device_name == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    config["rl_agent"]["device"] = str(device)

    if config["env"]["name"] != "mo-walker2d-v5":
        raise ValueError(
            "This first feature-ablation runner is restricted to mo-walker2d-v5."
        )

    sparsity_levels = list(config["env"]["sparsity_levels"])
    sparse_indices = [i for i, sparsity in enumerate(sparsity_levels) if sparsity > 0.0]
    if len(sparse_indices) != 1:
        raise ValueError("Expected exactly one sparse reward objective.")

    collection_env = AsymmetricSparsityWrapper(
        mo_gym.make(config["env"]["name"]),
        sparsity_levels=sparsity_levels,
    )
    eval_env = mo_gym.make(config["env"]["name"])

    collection_env.action_space.seed(seed)
    collection_env.reset(seed=seed)
    eval_env.action_space.seed(seed + 123)
    eval_env.reset(seed=seed + 123)

    obs_dim = int(collection_env.observation_space.shape[0])
    action_dim = int(collection_env.action_space.shape[0])

    full_config = _make_shaper_config(config, removed_observation_indices=[])
    full_shaper = IRLRewardShaper(full_config, obs_dim, action_dim)

    shaped_train_env = IRLShapingWrapper(collection_env, full_shaper)

    num_cycles = int(config["irl"]["num_refinement_cycles"])
    refinement_timesteps = int(config["irl"]["refinement_timesteps"])

    agent = CAPQL(
        env=shaped_train_env,
        seed=seed,
        project_name=str(config["log_dir"]),
        all_timesteps=num_cycles * refinement_timesteps,
        lambda_loss=config["irl"]["lambda"],
        device=str(device),
        log=bool(config["irl"].get("reconstruction_agent_log", False)),
    )

    # Constructing the auxiliary model must not perturb the main run's RNG.
    main_rng_state = _capture_rng_state()
    no_velocity_config = _make_shaper_config(
        config,
        removed_observation_indices=config["irl"].get("removed_indices", []),
    )
    no_velocity_shaper = IRLRewardShaper(
        no_velocity_config,
        obs_dim,
        action_dim,
    )
    _restore_rng_state(main_rng_state)

    shapers = [full_shaper, no_velocity_shaper]
    sparse_channel_idx = full_shaper.sparse_channel_idx

    initial_episodes = int(config["irl"]["initial_collection_episodes"])
    print(f"Collecting {initial_episodes} initial random trajectories...")
    for _ in range(initial_episodes):
        segments = _collect_episode(
            env=collection_env,
            sparse_channel_idx=sparse_channel_idx,
            action_fn=lambda _obs: collection_env.action_space.sample(),
        )
        _add_episode_to_shapers(segments, shapers)

    for cycle in range(num_cycles):
        print(f"\nCycle {cycle + 1}/{num_cycles}: training full reward model")
        full_shaper.train_reward_model_nn(
            epochs=int(config["irl"]["nn_epochs"]),
            run_id=f"full_cycle_{cycle + 1}",
            val_split=float(config["irl"].get("val_split", 0.2)),
            patience=int(config["irl"].get("early_stop_patience", 20)),
        )

        print(f"Cycle {cycle + 1}/{num_cycles}: training ablated reward model")
        _train_auxiliary_shaper_without_changing_main_rng(
            shaper=no_velocity_shaper,
            config=no_velocity_config,
            seed=seed,
            cycle=cycle,
        )

        print(f"Cycle {cycle + 1}/{num_cycles}: training CAPQL")
        agent.train(
            eval_env=eval_env,
            total_timesteps=refinement_timesteps,
            ref_point=np.asarray(config["irl"]["reference"], dtype=np.float32),
            known_pareto_front=None,
            eval_freq=int(config["rl_agent"].get("eval_freq", 10_000)),
        )

        expert_episodes = int(config["irl"]["expert_collection_episodes"])
        print(
            f"Cycle {cycle + 1}/{num_cycles}: collecting "
            f"{expert_episodes} common expert trajectories"
        )
        for _ in range(expert_episodes):
            segments = _collect_episode(
                env=collection_env,
                sparse_channel_idx=sparse_channel_idx,
                action_fn=agent.predict,
            )
            _add_episode_to_shapers(segments, shapers)

    held_out_count = int(config["irl"].get("reconstruction_test_episodes", 1000))
    print(
        f"\nCollecting {held_out_count} fresh held-out trajectories. "
        "These are not added to either reward model."
    )
    held_out_episodes = _collect_held_out_episodes(
        env=collection_env,
        agent=agent,
        sparse_channel_idx=sparse_channel_idx,
        num_episodes=held_out_count,
        seed=seed,
    )
    payload = held_out_episodes

    full_metrics = full_shaper.evaluate_reconstruction(payload)
    no_velocity_metrics = no_velocity_shaper.evaluate_reconstruction(payload)

    if not isinstance(full_metrics, dict) or not isinstance(no_velocity_metrics, dict):
        raise TypeError("evaluate_reconstruction() must return a metrics dictionary.")

    result_rows = [
        {
            "seed": seed,
            "condition": "full_observation",
            "removed_observation_indices": "[]",
            "feature_dim": int(full_shaper.feature_dim),
            "training_trajectories": len(full_shaper.trajectory_data),
            "held_out_trajectories": held_out_count,
            **full_metrics,
        },
        {
            "seed": seed,
            "condition": "without_forward_velocity",
            "removed_observation_indices": f"[{config['irl'].get('removed_indices', [])}]",
            "feature_dim": int(no_velocity_shaper.feature_dim),
            "training_trajectories": len(no_velocity_shaper.trajectory_data),
            "held_out_trajectories": held_out_count,
            **no_velocity_metrics,
        },
    ]

    root_dir = Path(__file__).resolve().parents[2]

    str1 = ''.join(str(e) for e in sparsity_levels)
    output_dir = root_dir / "outputs" / "reconstruction_ablation" / f"seed_{seed}" /f"sparse_{str1}"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "summary.csv"
    _write_summary(result_rows, summary_path)

    resolved_config_path = output_dir / "resolved_config.yaml"
    resolved_config_path.write_text(
        OmegaConf.to_yaml(OmegaConf.create(config)),
        encoding="utf-8",
    )

    collection_env.close()
    eval_env.close()

    print(f"\nSaved reconstruction results to {summary_path}")
    return summary_path


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CONFIG_PATH = os.path.join(ROOT_DIR, "configs")


@hydra.main(config_path=CONFIG_PATH, config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    selected_cfg = cfg.shaping[cfg.shaping.active_env]
    seed = int(selected_cfg.seed)
    run_single_ablation(selected_cfg, seed=seed)
    #run_single_nn(selected_cfg, seed=seed)


if __name__ == "__main__":
    main()