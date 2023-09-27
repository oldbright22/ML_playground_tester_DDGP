# %%
import gymnasium as gym
import time
import numpy as np

# Set the environment ID
# env_id = 'InvertedDoublePendulum-v4'
env_id = 'HalfCheetah-v4'

# Create the environment
env = gym.make(env_id)

# Reset the environment
obs = env.reset()


# %%
episode_rewards = []
# Run a few episodes
for episode in range(200):
    done = truncated = False
    total_reward = 0.

    obs = env.reset()

    while not done:
        # time.sleep(0.01)
        # Choose a random action
        action = env.action_space.sample()

        # Perform the action in the environment
        obs, reward, done, truncated, info = env.step(action)

        # Update the total reward
        total_reward += float(reward)

        if done or truncated:
            break

    # Print the total reward for the episode
    print("Episode:", episode + 1, "Total Reward:", total_reward)
    episode_rewards.append(total_reward)

episode_rewards = np.array(episode_rewards)
print("Average reward:", np.mean(episode_rewards))
print("Standard deviation:", np.std(episode_rewards))

# %%
# Close the environment
env.close()
