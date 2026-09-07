#!/usr/bin/env python3
"""
Evaluate a chronological block of PRISM W&B runs.

Modes
-----
1) List runs by creation time:
   python evaluate_wandb_project.py --project ENTITY/PROJECT --list-runs

2) Evaluate runs [10, 20), final cycle only:
   python evaluate_wandb_project.py \
       --project ENTITY/PROJECT \
       --config /path/to/config.yaml \
       --run-start 10 \
       --run-end 20 \
       --cycle final \
       --num-episodes 100 \
       --test-seed 10000 \
       --label p_release_0.2

3) Override sparsity levels if they were not logged:
   --sparsity-levels "0.0,0.2"

Notes
-----
- --run-end is exclusive.
- Runs are sorted locally by run.created_at.
- Per selected run, model artifacts are discovered from run.logged_artifacts().
- Artifact names are expected to contain "cycle-N" and to contain:
      resymnet/
      capql/
- Fresh held-out trajectories are generated with new seeds.
- Your existing IRLRewardShaper.evaluate_reconstruction() is used.
"""

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import wandb
from omegaconf import OmegaConf
from hydra import compose, initialize_config_dir
import mo_gymnasium as mo_gym

from morl_baselines.multi_policy.capql.capql_equivariance import CAPQL
from src.reward_shaping.env import AsymmetricSparsityWrapper
from src.reward_shaping.reward_model import IRLRewardShaper, set_all_seeds


CYCLE_RE = re.compile(r"(?:^|[-_])cycle[-_]?(\d+)(?:[-_]|$)", re.IGNORECASE)


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--project", required=True, help="ENTITY/PROJECT")
    p.add_argument("--list-runs", action="store_true")

    p.add_argument("--config", default=None)
    p.add_argument("--config-section", default=None)

    p.add_argument("--run-start", type=int, default=None)
    p.add_argument("--run-end", type=int, default=None)

    p.add_argument(
        "--cycle",
        default="final",
        help="'final', 'all', or integer cycle number",
    )

    p.add_argument("--num-episodes", type=int, default=100)
    p.add_argument("--calibration-episodes", type=int, default=100, help="Episodes used only to reconstruct the missing StandardScaler.")
    p.add_argument("--test-seed", type=int, default=10000)
    p.add_argument("--calibration-seed", type=int, default=50000, help="Base seed for scaler calibration rollouts.")

    p.add_argument("--label", default=None)

    p.add_argument(
        "--sparsity-levels",
        default=None,
        help='Optional override such as "0.0,0.2"',
    )

    p.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default=None,
    )

    p.add_argument(
        "--output-dir",
        default="./wandb_reconstruction_eval",
    )

    p.add_argument(
        "--include-failed-runs",
        action="store_true",
    )

    return p.parse_args()


# ---------------------------------------------------------------------------
# W&B run discovery
# ---------------------------------------------------------------------------

def get_runs_sorted(api, project):
    runs = list(api.runs(project))
    runs.sort(key=lambda r: str(getattr(r, "created_at", "") or ""))
    return runs


def nested_get(d, keys, default=None):
    cur = d
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def guess_sparsity_from_run_config(run):
    cfg = getattr(run, "config", {}) or {}

    for keys in (
        ["env", "sparsity_levels"],
        ["sparsity_levels"],
        ["irl", "sparsity_levels"],
        ["config", "env", "sparsity_levels"],
    ):
        value = nested_get(cfg, keys)
        if value is not None:
            return value

    return None


def print_runs(runs):
    print()
    print("=" * 120)
    print("W&B RUNS SORTED BY CREATION TIME")
    print("=" * 120)

    header = (
        f"{'IDX':>4}  "
        f"{'CREATED':<26}  "
        f"{'STATE':<12}  "
        f"{'RUN NAME':<34}  "
        f"{'RUN ID':<12}  "
        f"SPARSITY(if logged)"
    )

    print(header)
    print("-" * len(header))

    previous_date = None

    for idx, run in enumerate(runs):
        created = str(getattr(run, "created_at", "") or "")
        state = str(getattr(run, "state", "") or "")
        name = str(getattr(run, "name", "") or "")
        run_id = str(getattr(run, "id", "") or "")
        sparsity = guess_sparsity_from_run_config(run)

        date_part = created[:10]
        if previous_date is not None and date_part != previous_date:
            print("-" * len(header))
        previous_date = date_part

        print(
            f"{idx:4d}  "
            f"{created[:26]:<26}  "
            f"{state[:12]:<12}  "
            f"{name[:34]:<34}  "
            f"{run_id[:12]:<12}  "
            f"{sparsity}"
        )

    print("=" * 120)
    print(f"Total runs: {len(runs)}")
    print()


# ---------------------------------------------------------------------------
# Artifact selection
# ---------------------------------------------------------------------------

def extract_cycle_number(name: str) -> Optional[int]:
    match = CYCLE_RE.search(name.split(":")[0])
    return int(match.group(1)) if match else None


def artifact_version_number(artifact) -> int:
    """Convert W&B artifact.version such as 'v19' -> 19."""
    version = str(getattr(artifact, "version", "") or "")
    match = re.fullmatch(r"v(\d+)", version, flags=re.IGNORECASE)
    return int(match.group(1)) if match else -1


def get_model_artifacts(run):
    """
    Get model artifacts actually logged by THIS W&B run.

    Do not resolve a global ':latest' alias here. Your artifact base names can
    be reused across separate launches, so global latest may belong to another
    run / sparsity condition.
    """
    found = []

    for artifact in run.logged_artifacts():
        if getattr(artifact, "type", None) != "model":
            continue

        cycle = extract_cycle_number(artifact.name)
        if cycle is None:
            continue

        version_num = artifact_version_number(artifact)
        found.append((cycle, version_num, artifact))

    found.sort(key=lambda x: (x[0], x[1]))
    return found


def select_artifacts(run, cycle_mode):
    artifacts = get_model_artifacts(run)

    if not artifacts:
        return []

    mode = str(cycle_mode).lower()

    if mode == "final":
        # Highest refinement cycle logged by this run.
        final_cycle = max(c for c, _, _ in artifacts)

        # If the same cycle was logged more than once, take highest vN.
        candidates = [
            (c, v, a)
            for c, v, a in artifacts
            if c == final_cycle
        ]

        c, _, a = max(candidates, key=lambda x: x[1])
        return [(c, a)]

    if mode == "all":
        # Keep only the highest artifact version for each cycle.
        best_by_cycle = {}

        for c, v, a in artifacts:
            if c not in best_by_cycle or v > best_by_cycle[c][0]:
                best_by_cycle[c] = (v, a)

        return [
            (c, best_by_cycle[c][1])
            for c in sorted(best_by_cycle)
        ]

    wanted = int(mode)

    candidates = [
        (c, v, a)
        for c, v, a in artifacts
        if c == wanted
    ]

    if not candidates:
        return []

    c, _, a = max(candidates, key=lambda x: x[1])
    return [(c, a)]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(path, section=None, sparsity_override=None, device=None):
    """
    Load the experiment config exactly like the Hydra training entry point.

    Your config structure is:

        configs/config.yaml
            defaults:
              - shaping: main.yaml
              - _self_

        configs/shaping/main.yaml
            active_env: env_walker2d
            env_hopper: ...
            env_cheetah: ...
            env_walker2d: ...
            ...

    The training script receives the COMPOSED config and then does:

        cfg = cfg.shaping[cfg.shaping.active_env]

    Therefore simply calling OmegaConf.load("configs/config.yaml") is NOT
    sufficient: that only reads the defaults list and does not compose
    shaping/main.yaml.

    This function uses Hydra compose when --config points at config.yaml.
    It also supports passing shaping/main.yaml directly as a fallback.
    """
    path = Path(path).resolve()

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    # ------------------------------------------------------------------
    # Case 1: top-level Hydra config, e.g. configs/config.yaml
    # ------------------------------------------------------------------
    if path.name in {"config.yaml", "config.yml"}:
        config_dir = str(path.parent)
        config_name = path.stem

        with initialize_config_dir(
            version_base="1.3",
            config_dir=config_dir,
        ):
            cfg = compose(config_name=config_name)

        # This mirrors your training code:
        #     cfg = cfg.shaping[cfg.shaping.active_env]
        if "shaping" not in cfg:
            raise KeyError(
                "Hydra composition succeeded, but the composed config has no "
                "'shaping' node. Check defaults in config.yaml."
            )

        shaping_cfg = cfg.shaping

        if section is None:
            if "active_env" not in shaping_cfg:
                raise KeyError(
                    "Composed cfg.shaping has no active_env. "
                    "Pass --config-section explicitly."
                )
            section = str(shaping_cfg.active_env)

        if section not in shaping_cfg:
            raise KeyError(
                f"Environment section '{section}' not found under cfg.shaping. "
                f"Available keys: {list(shaping_cfg.keys())}"
            )

        cfg = shaping_cfg[section]

    # ------------------------------------------------------------------
    # Case 2: user points directly at configs/shaping/main.yaml
    # ------------------------------------------------------------------
    else:
        cfg = OmegaConf.load(path)

        # main.yaml itself has:
        #   active_env: env_walker2d
        #   env_hopper: ...
        #   env_walker2d: ...
        if "active_env" in cfg:
            if section is None:
                section = str(cfg.active_env)

            if section not in cfg:
                raise KeyError(
                    f"Environment section '{section}' not found in {path}. "
                    f"Available keys: {list(cfg.keys())}"
                )

            cfg = cfg[section]

        # Also tolerate an already-composed config.
        elif "shaping" in cfg:
            shaping_cfg = cfg.shaping

            if section is None:
                section = str(shaping_cfg.active_env)

            cfg = shaping_cfg[section]

    config = OmegaConf.to_container(cfg, resolve=True)

    # Manual sparsity override, useful for old W&B batches where the
    # sparsity level was not logged.
    if sparsity_override is not None:
        values = [
            float(x.strip())
            for x in sparsity_override.split(",")
            if x.strip()
        ]

        if not values:
            raise ValueError("--sparsity-levels contained no numeric values.")

        config["env"]["sparsity_levels"] = values

    if device is not None:
        config["rl_agent"]["device"] = device

    if (
        config["rl_agent"].get("device", "cpu") == "cuda"
        and not torch.cuda.is_available()
    ):
        print("CUDA unavailable; falling back to CPU.")
        config["rl_agent"]["device"] = "cpu"

    print(f"Loaded environment section: {section}")
    print(f"Environment: {config['env']['name']}")
    print(f"Sparsity levels: {config['env']['sparsity_levels']}")

    return config


# ---------------------------------------------------------------------------
# Download artifact
# ---------------------------------------------------------------------------

def safe_name(text):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text))


def download_artifact(artifact, output_root, run_index, cycle):
    target = (
        output_root
        / "artifacts"
        / f"run_index_{run_index:04d}"
        / f"cycle_{cycle}"
        / safe_name(artifact.name)
    )
    target.mkdir(parents=True, exist_ok=True)

    artifact_dir = Path(artifact.download(root=str(target)))

    resymnet_dir = artifact_dir / "resymnet"
    capql_dir = artifact_dir / "capql"

    if not resymnet_dir.exists():
        raise FileNotFoundError(f"Missing {resymnet_dir}")

    if not capql_dir.exists():
        raise FileNotFoundError(f"Missing {capql_dir}")

    return resymnet_dir, capql_dir


# ---------------------------------------------------------------------------
# ReSymNet loading
# ---------------------------------------------------------------------------

def load_resymnet(irl_shaper, resymnet_dir):
    """Restore ReSymNet weights and report whether a fitted scaler was restored."""
    if not hasattr(irl_shaper, "load_reward_model"):
        raise AttributeError(
            "IRLRewardShaper has no load_reward_model() method. Add the inverse of "
            "save_reward_model() for the network weights before evaluating."
        )

    fn = irl_shaper.load_reward_model
    attempts = [
        lambda: fn(load_dir=str(resymnet_dir)),
        lambda: fn(save_dir=str(resymnet_dir)),
        lambda: fn(str(resymnet_dir)),
    ]

    last = None
    for call in attempts:
        try:
            call()
            scaler_ok = bool(getattr(irl_shaper, "scaler_fitted", False))
            if scaler_ok:
                print("  ReSymNet restored; scaler restored from checkpoint.")
            else:
                print("  ReSymNet restored; scaler missing, will reconstruct from calibration rollouts.")
            return scaler_ok
        except TypeError as exc:
            last = exc

    raise TypeError(f"Could not call load_reward_model(); last error: {last}")


# ---------------------------------------------------------------------------
# CAPQL loading
# ---------------------------------------------------------------------------

def find_capql_checkpoint(capql_dir):
    preferred = [
        "capql_policy.tar",
        "capql_policy.pt",
        "capql_policy.pth",
        "capql_policy.zip",
    ]

    for name in preferred:
        p = capql_dir / name
        if p.exists():
            return p

    candidates = [
        p for p in capql_dir.rglob("*")
        if p.is_file()
        and p.suffix.lower() in {".tar", ".pt", ".pth", ".zip"}
    ]

    if not candidates:
        raise FileNotFoundError(
            f"No CAPQL checkpoint found in {capql_dir}"
        )

    candidates.sort()
    return candidates[0]


def load_capql(agent, checkpoint):
    attempts = [
        lambda: agent.load(str(checkpoint), load_replay_buffer=False),
        lambda: agent.load(str(checkpoint)),
        lambda: agent.load(
            save_dir=str(checkpoint.parent),
            filename=checkpoint.stem,
        ),
    ]

    errors = []

    for call in attempts:
        try:
            call()
            return
        except (TypeError, FileNotFoundError, AttributeError) as exc:
            errors.append(repr(exc))

    raise RuntimeError(
        "Could not restore CAPQL.\n" + "\n".join(errors)
    )


# ---------------------------------------------------------------------------
# Build env + models
# ---------------------------------------------------------------------------

def make_env(config):
    return AsymmetricSparsityWrapper(
        mo_gym.make(config["env"]["name"]),
        sparsity_levels=config["env"]["sparsity_levels"],
    )


def build_models(config, resymnet_dir, capql_dir):
    env = make_env(config)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    shaper = IRLRewardShaper(
        config,
        obs_dim,
        act_dim,
    )
    scaler_restored = load_resymnet(shaper, resymnet_dir)

    for net in shaper.reward_nets:
        net.eval()

    all_timesteps = int(
        config["irl"]["num_refinement_cycles"]
        * config["irl"]["refinement_timesteps"]
    )

    agent = CAPQL(
        env=env,
        seed=int(config["seed"]),
        project_name=config["log_dir"],
        all_timesteps=all_timesteps,
        lambda_loss=config["irl"]["lambda"],
    )

    checkpoint = find_capql_checkpoint(capql_dir)
    load_capql(agent, checkpoint)

    return env, shaper, agent, scaler_restored


# ---------------------------------------------------------------------------
# Fresh held-out trajectories
# ---------------------------------------------------------------------------

def predict_action(agent, obs):
    action = agent.predict(obs)
    if isinstance(action, tuple):
        action = action[0]
    return np.asarray(action)


def generate_held_out(env, agent, num_episodes, seed_start):
    episodes = []

    for ep_idx in range(num_episodes):
        ep_seed = seed_start + ep_idx
        obs, _ = env.reset(seed=ep_seed)

        observations = []
        actions = []
        true_dense_rewards = []

        terminated = False
        truncated = False

        while not (terminated or truncated):
            action = predict_action(agent, obs)

            next_obs, _, terminated, truncated, info = env.step(action)

            if "true_dense_rewards" not in info:
                raise KeyError(
                    "Expected info['true_dense_rewards']."
                )

            observations.append(
                np.asarray(obs, dtype=np.float32).copy()
            )
            actions.append(
                np.asarray(action, dtype=np.float32).copy()
            )
            true_dense_rewards.append(
                np.asarray(
                    info["true_dense_rewards"],
                    dtype=np.float32,
                ).copy()
            )

            obs = next_obs

        episodes.append(
            {
                "observations": observations,
                "actions": actions,
                "true_dense_rewards": true_dense_rewards,
                "seed": ep_seed,
            }
        )

    return episodes


def fit_scaler_from_calibration_episodes(shaper, episodes):
    """Fit StandardScaler on calibration trajectories only, never on the test set."""
    features = []
    for ep in episodes:
        for obs, action, dense_reward in zip(
            ep["observations"], ep["actions"], ep["true_dense_rewards"]
        ):
            features.append(shaper._get_features(obs, action, dense_reward))

    if not features:
        raise RuntimeError("Calibration set produced no features.")

    features = np.asarray(features, dtype=np.float64)
    shaper.feature_scaler.fit(features)
    shaper.scaler_fitted = True
    print(f"  Fitted replacement scaler on {len(episodes)} calibration episodes ({len(features)} steps).")


# ---------------------------------------------------------------------------
# Extra per-episode metrics
# ---------------------------------------------------------------------------

def evaluate_per_episode(shaper, episodes):
    rows = []

    for i, ep in enumerate(episodes):
        preds = []
        targets = []

        for obs, action, dense_reward in zip(
            ep["observations"],
            ep["actions"],
            ep["true_dense_rewards"],
        ):
            pred = shaper.get_shaped_reward(
                obs,
                action,
                dense_reward,
            )

            target = dense_reward[shaper.sparse_channel_idx]

            preds.append(float(pred))
            targets.append(float(target))

        preds = np.asarray(preds, dtype=np.float64)
        targets = np.asarray(targets, dtype=np.float64)
        errors = preds - targets

        rows.append(
            {
                "episode_index": i,
                "seed": int(ep["seed"]),
                "length": int(len(targets)),
                "per_step_mse": float(np.mean(errors ** 2)),
                "per_step_rmse": float(np.sqrt(np.mean(errors ** 2))),
                "per_step_mae": float(np.mean(np.abs(errors))),
                "predicted_return": float(preds.sum()),
                "target_return": float(targets.sum()),
                "return_error": float(preds.sum() - targets.sum()),
            }
        )

    return rows


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def save_individual(result, root):
    d = root / "individual"
    d.mkdir(parents=True, exist_ok=True)

    path = d / (
        f"runidx_{result['wandb_run_index']:04d}"
        f"_cycle_{result['cycle']}"
        f"_{safe_name(result['wandb_run_id'])}.json"
    )

    with path.open("w") as f:
        json.dump(result, f, indent=2)

    return path


def flatten_result(result):
    row = {
        "condition": result["condition"],
        "wandb_run_index": result["wandb_run_index"],
        "wandb_run_id": result["wandb_run_id"],
        "wandb_run_name": result["wandb_run_name"],
        "wandb_created_at": result["wandb_created_at"],
        "cycle": result["cycle"],
        "artifact_name": result["artifact_name"],
        "environment": result["environment"],
        "sparsity_levels": json.dumps(result["sparsity_levels"]),
        "test_seed_start": result["test_seed_start"],
        "num_test_episodes": result["num_test_episodes"],
        "calibration_seed_start": result["calibration_seed_start"],
        "num_calibration_episodes": result["num_calibration_episodes"],
        "scaler_restored_from_checkpoint": result["scaler_restored_from_checkpoint"],
    }

    row.update(result["aggregate_metrics"])
    return row


def save_csv(results, root):
    if not results:
        return None

    rows = [flatten_result(r) for r in results]
    path = root / "all_model_metrics.csv"

    fieldnames = []
    seen = set()

    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return path


def make_summary(results, label):
    summary = {
        "condition": label,
        "n_models": len(results),
        "metrics": {},
    }

    if not results:
        return summary

    common_keys = set(results[0]["aggregate_metrics"].keys())
    for r in results[1:]:
        common_keys &= set(r["aggregate_metrics"].keys())

    for key in sorted(common_keys):
        vals = [
            r["aggregate_metrics"][key]
            for r in results
        ]

        if not all(
            isinstance(v, (int, float, np.integer, np.floating))
            and not isinstance(v, bool)
            for v in vals
        ):
            continue

        vals = np.asarray(vals, dtype=float)
        vals = vals[np.isfinite(vals)]

        if len(vals) == 0:
            continue

        std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        sem = std / np.sqrt(len(vals)) if len(vals) > 1 else 0.0

        summary["metrics"][key] = {
            "mean": float(np.mean(vals)),
            "std": std,
            "sem": float(sem),
            "n": int(len(vals)),
        }

    return summary


# ---------------------------------------------------------------------------
# Evaluate one artifact
# ---------------------------------------------------------------------------

def evaluate_one(
    config,
    artifact,
    cycle,
    run,
    run_index,
    root,
    num_episodes,
    seed_start,
    calibration_episodes,
    calibration_seed_start,
    label,
):
    print()
    print("=" * 80)
    print(
        f"Run index {run_index} | run={run.id} | "
        f"cycle={cycle} | artifact={artifact.name} | version={artifact.version}"
    )
    print("=" * 80)

    resymnet_dir, capql_dir = download_artifact(
        artifact,
        root,
        run_index,
        cycle,
    )

    env = None

    try:
        env, shaper, agent, scaler_restored = build_models(
            config,
            resymnet_dir,
            capql_dir,
        )

        if not scaler_restored:
            calibration_set = generate_held_out(
                env, agent, calibration_episodes, calibration_seed_start
            )
            fit_scaler_from_calibration_episodes(shaper, calibration_set)

        episodes = generate_held_out(
            env,
            agent,
            num_episodes,
            seed_start,
        )

        # Existing evaluator, on the disjoint held-out test set only.
        aggregate = shaper.evaluate_reconstruction(episodes)

        per_episode = evaluate_per_episode(
            shaper,
            episodes,
        )

        result = {
            "condition": label,
            "wandb_run_index": run_index,
            "wandb_run_id": run.id,
            "wandb_run_name": run.name,
            "wandb_created_at": run.created_at,
            "wandb_run_state": run.state,
            "cycle": cycle,
            "artifact_name": artifact.name,
            "artifact_version": artifact.version,
            "artifact_qualified_name": f"{artifact.name}:{artifact.version}",
            "environment": config["env"]["name"],
            "sparsity_levels": config["env"]["sparsity_levels"],
            "test_seed_start": seed_start,
            "num_test_episodes": num_episodes,
            "calibration_seed_start": calibration_seed_start,
            "num_calibration_episodes": calibration_episodes,
            "scaler_restored_from_checkpoint": scaler_restored,
            "aggregate_metrics": aggregate,
            "per_episode_metrics": per_episode,
        }

        path = save_individual(result, root)

        for key, value in aggregate.items():
            if isinstance(value, float):
                print(f"{key:28s}: {value:.6f}")
            else:
                print(f"{key:28s}: {value}")

        print(f"Saved: {path}")

        return result

    finally:
        if env is not None:
            env.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    api = wandb.Api()
    print(f"Reading project: {args.project}")
    runs = get_runs_sorted(api, args.project)

    if args.list_runs:
        print_runs(runs)
        return

    if args.config is None:
        raise ValueError("--config is required for evaluation.")

    if args.run_start is None or args.run_end is None:
        raise ValueError(
            "Pass both --run-start and --run-end."
        )

    if not (0 <= args.run_start < args.run_end <= len(runs)):
        raise ValueError(
            f"Invalid range [{args.run_start}, {args.run_end}) "
            f"for project with {len(runs)} runs."
        )

    selected = list(
        enumerate(
            runs[args.run_start:args.run_end],
            start=args.run_start,
        )
    )

    print_runs(runs)

    config = load_config(
        args.config,
        section=args.config_section,
        sparsity_override=args.sparsity_levels,
        device=args.device,
    )

    root = Path(args.output_dir)
    if args.label:
        root = root / safe_name(args.label)
    root.mkdir(parents=True, exist_ok=True)

    print(
        f"Selected runs [{args.run_start}, {args.run_end}) "
        f"({len(selected)} runs)"
    )
    print(f"Cycle mode: {args.cycle}")
    print(f"Condition label: {args.label}")
    print(f"Environment: {config['env']['name']}")
    print(
        "Sparsity levels used during evaluation: "
        f"{config['env']['sparsity_levels']}"
    )

    set_all_seeds(args.test_seed)

    all_results = []
    failures = []
    bad_states = {"failed", "crashed", "killed"}

    for selected_offset, (run_index, run) in enumerate(selected):
        state = str(getattr(run, "state", "") or "").lower()

        if (
            state in bad_states
            and not args.include_failed_runs
        ):
            print(
                f"Skipping run index {run_index} "
                f"because state={run.state}"
            )
            continue

        try:
            artifacts = select_artifacts(
                run,
                args.cycle,
            )

            if not artifacts:
                print(
                    f"No PRISM model artifacts found for "
                    f"run index {run_index} ({run.id})."
                )
                continue

            for artifact_offset, (cycle, artifact) in enumerate(artifacts):
                # Different W&B runs get disjoint held-out seed blocks.
                seed_start = (
                    args.test_seed
                    + selected_offset * 100000
                    + artifact_offset * args.num_episodes
                )
                calibration_seed_start = (
                    args.calibration_seed
                    + selected_offset * 100000
                    + artifact_offset * args.calibration_episodes
                )

                result = evaluate_one(
                    config=config,
                    artifact=artifact,
                    cycle=cycle,
                    run=run,
                    run_index=run_index,
                    root=root,
                    num_episodes=args.num_episodes,
                    seed_start=seed_start,
                    calibration_episodes=args.calibration_episodes,
                    calibration_seed_start=calibration_seed_start,
                    label=args.label,
                )

                all_results.append(result)

        except Exception as exc:
            print(
                f"FAILED run index {run_index} ({run.id}): "
                f"{type(exc).__name__}: {exc}"
            )

            failures.append(
                {
                    "wandb_run_index": run_index,
                    "wandb_run_id": run.id,
                    "wandb_run_name": run.name,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )

    csv_path = save_csv(all_results, root)

    summary = make_summary(
        all_results,
        args.label,
    )

    summary.update(
        {
            "project": args.project,
            "run_start": args.run_start,
            "run_end_exclusive": args.run_end,
            "cycle_mode": args.cycle,
            "environment": config["env"]["name"],
            "sparsity_levels": config["env"]["sparsity_levels"],
            "num_test_episodes_per_model": args.num_episodes,
            "num_calibration_episodes_per_model": args.calibration_episodes,
            "calibration_seed_base": args.calibration_seed,
            "num_failures": len(failures),
            "failures": failures,
        }
    )

    summary_path = root / "summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    failures_path = root / "failures.json"
    with failures_path.open("w") as f:
        json.dump(failures, f, indent=2)

    print()
    print("=" * 80)
    print("DONE")
    print("=" * 80)
    print(f"Successful model evaluations: {len(all_results)}")
    print(f"Failures: {len(failures)}")

    if csv_path is not None:
        print(f"CSV: {csv_path.resolve()}")

    print(f"Summary: {summary_path.resolve()}")
    print(f"Failures: {failures_path.resolve()}")

    if summary["metrics"]:
        print("\nAcross-model summary:")
        for key, stats in summary["metrics"].items():
            print(
                f"{key:28s}: "
                f"{stats['mean']:.6f} ± {stats['std']:.6f} "
                f"(SD, n={stats['n']})"
            )


if __name__ == "__main__":
    main()
