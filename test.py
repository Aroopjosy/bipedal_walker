import gymnasium as gym
from stable_baselines3 import PPO
import time

# Load the trained model (ensure the path matches where your model was saved)
model_path = "bipedal_walker_ppo"  # or provide a full path if needed
model = PPO.load(model_path)

# Create the environment with rendering enabled
env_id = "BipedalWalker-v3"
env = gym.make(env_id, render_mode="human")  # gymnasium uses the render_mode argument

# Number of episodes to test
num_episodes = 10

for episode in range(1, num_episodes + 1):
    # Reset the environment (gymnasium reset returns (observation, info))
    obs, info = env.reset()
    done = False
    episode_reward = 0

    while not done:
        # Use the trained model to predict the next action
        action, _ = model.predict(obs, deterministic=True)
        
        # Step through the environment
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        episode_reward += reward

        # Slow down the loop for a better visual experience (optional)
        time.sleep(0.02)

    print(f"Episode {episode} reward: {episode_reward}")

env.close()
