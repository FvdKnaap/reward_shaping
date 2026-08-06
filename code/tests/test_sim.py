import mo_gymnasium as mo_gym
import numpy as np

from src.reward_shaping.env import Walker2dRealityGapWrapper

env = Walker2dRealityGapWrapper(
    mo_gym.make("mo-walker2d-v5"),
    action_delay_steps=1,
    randomize_actuator_gain=True,
    actuator_gain_high=1.2,
    actuator_gain_low=0.8,
    observation_noise_std=0.0
)

obs, info = env.reset(seed=0)

gain = info['reality_gap/actuator_gain']
print(gain)

assert 0.8 <= gain <= 1.2

action = np.ones(env.action_space.shape,dtype=np.float32)

for step in range(4):
    obs, reward, terminated, truncated, info = env.step(action)

    print(step, info['reality_gap/actuator_gain'], info['reality_gap/commanded_action'], info['reality_gap/delayed_action'])