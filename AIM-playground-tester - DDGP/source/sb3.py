import stable_baselines3 as sb3
import gymnasium as gym
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.env_util import make_vec_env

# Create the Ant-v3 environment
# env = make_vec_env('Ant-v3', n_envs=8)
env = make_vec_env('MountainCarContinuous-v0')

# Create the A2C agent
model = PPO('MlpPolicy', env, verbose=1)

# Train the agent
model.learn(total_timesteps=100000)

# Save the trained agent
model.save("a2c_ant")

# Load the trained agent
loaded_model = A2C.load("a2c_ant")

# Evaluate the trained agent
mean_reward, _ = model.evaluate(env, n_eval_episodes=10)

print("Mean reward:", mean_reward)
