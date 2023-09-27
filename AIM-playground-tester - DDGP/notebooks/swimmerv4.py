# %%
import gymnasium as gym
import time

# Set the environment ID
env_id = 'Swimmer-v4'

# Create the environment
env = gym.make(env_id, render_mode='human')

# Reset the environment
obs = env.reset()


# %%
# Run a few episodes
for episode in range(5):
    done = truncated = False
    total_reward = 0.

    obs = env.reset()

    while not done:
        time.sleep(0.05)
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

# %%
# Close the environment
env.close()
