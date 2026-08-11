from __future__ import annotations
import gymnasium as gym
import numpy as np
from typing import Sequence


from collections import deque
from collections.abc import Mapping, Sequence
from typing import Any

import mo_gymnasium as mo_gym



class AsymmetricSparsityWrapper(gym.Wrapper):
    """ A wrapper that implements asymmetric sparsity in the reward structure.
    It allows for different sparsity levels for each reward objective.
    The sparsity levels are defined as a sequence of floats, where each float represents the sparsity level for a corresponding reward objective.
    A sparsity level of 0.0 means the reward is dense and always released, while a sparsity level of 1.0 means the reward is never released.
    The wrapper accumulates rewards for each objective and releases them based on the defined sparsity levels.
    """

    def __init__(self, env: gym.Env, sparsity_levels: Sequence[float]):
        super().__init__(env)

        self.num_objectives = self.unwrapped.reward_space.shape[0]
        if len(sparsity_levels) != self.num_objectives:
            raise ValueError("Sparsity levels length must match the number of reward objectives.")

        self.sparsity_levels = np.array(sparsity_levels, dtype=np.float32)
        self.reward_release_probs = 1.0 - self.sparsity_levels
        self.reward_accumulator = np.zeros(self.num_objectives, dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self.reward_accumulator.fill(0)
        return obs, info

    def step(self, action: np.ndarray):
        obs, true_dense_reward_vec, terminated, truncated, info = self.env.step(action)
        info['true_dense_rewards'] = true_dense_reward_vec

        self.reward_accumulator += true_dense_reward_vec
        reward_to_return = np.zeros_like(self.reward_accumulator)
        release_decisions = self.np_random.random(size=self.num_objectives) < self.reward_release_probs
    
       
        # Release rewards based on sparsity levels and release decisions
        for i in range(self.num_objectives):
            is_dense_channel = self.sparsity_levels[i] == 0.0
            if is_dense_channel or release_decisions[i] or terminated or truncated:
                reward_to_return[i] = self.reward_accumulator[i]
                self.reward_accumulator[i] = 0.0
        
        return obs, reward_to_return, terminated, truncated, info



"""Literature-inspired Walker2d reality-gap wrapper.

Drop this file into:
    code/src/reward_shaping/reality_gap_env.py

The wrapper keeps the original MO-Walker2d reward calculation intact while
adding three deployment imperfections:

1. Fixed action-command delay.
2. Episode-level actuator-strength randomisation by scaling MuJoCo's
   actuator gear parameters.
3. Optional timestep-level Gaussian observation noise.

Because actuator strength is changed inside the MuJoCo model, the original
control-cost objective is still computed from the delayed command passed to
``env.step`` rather than from an externally scaled action.
"""


class Walker2dRealityGapWrapper(gym.Wrapper):
    """Add simple actuator and sensing imperfections to MO-Walker2d.

    Parameters
    ----------
    env:
        A ``mo-walker2d-v5`` environment.
    action_delay_steps:
        Number of RL steps by which motor commands are delayed. Walker2d's
        default control step is 8 ms, so two steps correspond to about 16 ms.
    randomize_actuator_gain:
        When ``True``, sample one global actuator gain at every reset and keep
        it fixed for the full episode.
    actuator_gain_low, actuator_gain_high:
        Uniform sampling interval used during training.
    fixed_actuator_gain:
        Gain used when randomisation is disabled, typically 1.0 for periodic
        evaluation.
    observation_noise_std:
        Gaussian-noise multiplier. A scalar applies the same multiplier to
        all dimensions; a vector may specify one multiplier per dimension.
        Set to 0.0 to disable observation noise.
    observation_noise_scale:
        Per-dimension physical scale multiplied by ``observation_noise_std``.
        If omitted, a vector of ones is used. For a later calibrated study,
        this can be set to clean-rollout observation standard deviations.
    record_clean_observation:
        Store the uncorrupted observation in ``info`` for diagnostics. This
        does not expose it to the policy.
    """

    def __init__(
        self,
        env: gym.Env,
        *,
        action_delay_steps: int = 2,
        randomize_actuator_gain: bool = True,
        actuator_gain_low: float = 0.8,
        actuator_gain_high: float = 1.2,
        fixed_actuator_gain: float = 1.0,
        observation_noise_std: float | Sequence[float] = 0.0,
        observation_noise_scale: float | Sequence[float] | None = None,
        record_clean_observation: bool = False,
    ) -> None:
        super().__init__(env)

        if action_delay_steps < 0:
            raise ValueError("action_delay_steps must be non-negative")
        if actuator_gain_low <= 0.0:
            raise ValueError("actuator_gain_low must be positive")
        if actuator_gain_high < actuator_gain_low:
            raise ValueError(
                "actuator_gain_high must be greater than or equal to "
                "actuator_gain_low"
            )
        if fixed_actuator_gain <= 0.0:
            raise ValueError("fixed_actuator_gain must be positive")
        if not isinstance(self.action_space, gym.spaces.Box):
            raise TypeError("Walker2dRealityGapWrapper requires a Box action space")
        if not isinstance(self.observation_space, gym.spaces.Box):
            raise TypeError(
                "Walker2dRealityGapWrapper requires a Box observation space"
            )

        model = getattr(self.unwrapped, "model", None)
        if model is None or not hasattr(model, "actuator_gear"):
            raise TypeError(
                "The wrapped environment must expose a MuJoCo model with "
                "actuator_gear"
            )

        self.action_delay_steps = int(action_delay_steps)
        self.randomize_actuator_gain = bool(randomize_actuator_gain)
        self.actuator_gain_low = float(actuator_gain_low)
        self.actuator_gain_high = float(actuator_gain_high)
        self.fixed_actuator_gain = float(fixed_actuator_gain)
        self.record_clean_observation = bool(record_clean_observation)

        observation_shape = self.observation_space.shape
        if observation_shape is None:
            raise ValueError("Observation space must have a fixed shape")

        self._observation_noise_std = self._as_observation_vector(
            observation_noise_std,
            observation_shape,
            name="observation_noise_std",
        )
        if np.any(self._observation_noise_std < 0.0):
            raise ValueError("observation_noise_std must be non-negative")

        if observation_noise_scale is None:
            self._observation_noise_scale = np.ones(
                observation_shape,
                dtype=np.float64,
            )
        else:
            self._observation_noise_scale = self._as_observation_vector(
                observation_noise_scale,
                observation_shape,
                name="observation_noise_scale",
            )
            if np.any(self._observation_noise_scale < 0.0):
                raise ValueError("observation_noise_scale must be non-negative")

        self._base_actuator_gear = np.asarray(
            self.unwrapped.model.actuator_gear,
            dtype=np.float64,
        ).copy()
        self.current_actuator_gain = self.fixed_actuator_gain
        self._action_queue: deque[np.ndarray] = deque()

    @staticmethod
    def _as_observation_vector(
        value: float | Sequence[float],
        shape: tuple[int, ...],
        *,
        name: str,
    ) -> np.ndarray:
        array = np.asarray(value, dtype=np.float64)
        if array.ndim == 0:
            return np.full(shape, float(array), dtype=np.float64)
        try:
            return np.broadcast_to(array, shape).astype(np.float64, copy=True)
        except ValueError as exc:
            raise ValueError(
                f"{name} with shape {array.shape} cannot be broadcast to {shape}"
            ) from exc

    def _set_actuator_gain(self, gain: float) -> None:
        self.current_actuator_gain = float(gain)
        np.copyto(
            self.unwrapped.model.actuator_gear,
            self._base_actuator_gear * self.current_actuator_gain,
        )

    def _reset_action_queue(self) -> None:
        zero_action = np.zeros(
            self.action_space.shape,
            dtype=self.action_space.dtype,
        )
        self._action_queue = deque(
            zero_action.copy() for _ in range(self.action_delay_steps)
        )

    def _delay_action(self, action: np.ndarray) -> np.ndarray:
        command = np.asarray(action, dtype=self.action_space.dtype)
        if command.shape != self.action_space.shape:
            raise ValueError(
                f"Expected action shape {self.action_space.shape}, "
                f"received {command.shape}"
            )

        command = np.clip(command, self.action_space.low, self.action_space.high)
        if self.action_delay_steps == 0:
            return command.copy()

        self._action_queue.append(command.copy())
        return self._action_queue.popleft()

    def _corrupt_observation(self, observation: np.ndarray) -> np.ndarray:
        observation_array = np.asarray(observation)
        if not np.any(self._observation_noise_std):
            return observation_array.copy()

        noise_scale = self._observation_noise_std * self._observation_noise_scale
        noise = self.np_random.normal(
            loc=0.0,
            scale=noise_scale,
            size=observation_array.shape,
        )
        return (observation_array + noise).astype(
            observation_array.dtype,
            copy=False,
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        clean_observation, info = self.env.reset(seed=seed, options=options)

        # Always restore the nominal model before applying a new episode gain;
        # otherwise multiplicative scaling would accumulate across resets.
        np.copyto(self.unwrapped.model.actuator_gear, self._base_actuator_gear)

        if self.randomize_actuator_gain:
            gain = self.np_random.uniform(
                self.actuator_gain_low,
                self.actuator_gain_high,
            )
        else:
            gain = self.fixed_actuator_gain
        self._set_actuator_gain(float(gain))
        self._reset_action_queue()

        noisy_observation = self._corrupt_observation(clean_observation)
        info = dict(info)
        info.update(
            {
                "reality_gap/action_delay_steps": self.action_delay_steps,
                "reality_gap/actuator_gain": self.current_actuator_gain,
                "reality_gap/observation_noise_enabled": bool(
                    np.any(self._observation_noise_std)
                ),
            }
        )
        if self.record_clean_observation:
            info["reality_gap/clean_observation"] = np.asarray(
                clean_observation
            ).copy()

        return noisy_observation, info

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, bool, bool, dict[str, Any]]:
        commanded_action = np.asarray(action, dtype=self.action_space.dtype)
        delayed_action = self._delay_action(commanded_action)

        # The delayed command is passed unchanged to Walker2d. MuJoCo's
        # actuator gear was scaled at reset, so the physical torque changes
        # while MO-Walker2d continues to calculate control cost from this
        # delayed command.
        clean_observation, reward, terminated, truncated, info = self.env.step(
            delayed_action
        )
        noisy_observation = self._corrupt_observation(clean_observation)

        info = dict(info)
        info.update(
            {
                "reality_gap/commanded_action": commanded_action,
                "reality_gap/delayed_action": delayed_action.copy(),
                "reality_gap/action_delay_steps": self.action_delay_steps,
                "reality_gap/actuator_gain": self.current_actuator_gain,
            }
        )
        if self.record_clean_observation:
            info["reality_gap/clean_observation"] = np.asarray(
                clean_observation
            ).copy()

        return noisy_observation, reward, terminated, truncated, info

    def close(self) -> None:
        # Leave the MuJoCo model in its nominal state for safe reuse/debugging.
        np.copyto(self.unwrapped.model.actuator_gear, self._base_actuator_gear)
        self.env.close()


def make_walker2d_reality_gap_env(
    *,
    training: bool,
    env_name: str = "mo-walker2d-v5",
    action_delay_steps: int = 2,
    actuator_gain_low: float = 0.8,
    actuator_gain_high: float = 1.2,
    evaluation_actuator_gain: float = 1.0,
    observation_noise_std: float | Sequence[float] = 0.0,
    observation_noise_scale: float | Sequence[float] | None = None,
    record_clean_observation: bool = False,
    env_kwargs: Mapping[str, Any] | None = None,
) -> Walker2dRealityGapWrapper:
    """Create the wrapped environment used by PRISM, baseline, or oracle.

    During training, one global actuator gain is sampled per episode from
    ``[actuator_gain_low, actuator_gain_high]``. During evaluation, the gain is
    fixed to ``evaluation_actuator_gain`` so periodic learning curves remain
    comparable and low-variance.
    """

    if env_name != "mo-walker2d-v5":
        raise ValueError(
            "This first implementation is intentionally restricted to "
            "mo-walker2d-v5"
        )

    base_env = mo_gym.make(env_name, **dict(env_kwargs or {}))
    
    return Walker2dRealityGapWrapper(
        base_env,
        action_delay_steps=action_delay_steps,
        randomize_actuator_gain=training,
        actuator_gain_low=actuator_gain_low,
        actuator_gain_high=actuator_gain_high,
        fixed_actuator_gain=evaluation_actuator_gain,
        observation_noise_std=observation_noise_std,
        observation_noise_scale=observation_noise_scale,
        record_clean_observation=record_clean_observation,
    )


def apply_walker2d_reality_gap(
    env: gym.Env,
    config: Mapping[str, Any] | None,
    *,
    training: bool,
) -> gym.Env:
    """Apply the wrapper to an already-created environment from Hydra config."""

    if not config or not bool(config.get("enabled", False)):
        return env

    gain_config = config.get("actuator_gain", {})
    noise_config = config.get("observation_noise", {})

    return Walker2dRealityGapWrapper(
        env,
        action_delay_steps=int(config.get("action_delay_steps", 2)),
        randomize_actuator_gain=(
            training and bool(gain_config.get("randomize_train", True))
        ),
        actuator_gain_low=float(gain_config.get("low", 0.8)),
        actuator_gain_high=float(gain_config.get("high", 1.2)),
        fixed_actuator_gain=float(gain_config.get("eval", 1.0)),
        observation_noise_std=noise_config.get("std", 0.0),
        observation_noise_scale=noise_config.get("scale", None),
        record_clean_observation=bool(
            config.get("record_clean_observation", False)
        ),
    )
