import gymnasium as gym
import numpy as np
from typing import Sequence

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


