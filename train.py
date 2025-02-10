import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, ProgressBarCallback
from stable_baselines3.common.env_util import make_vec_env
import os

# Create a vectorized environment for the continuous action space task
env_id = "BipedalWalker-v3"
env = make_vec_env(env_id, n_envs=1)

# Initialize the PPO agent with an MLP policy (suitable for continuous actions)
model = PPO("MlpPolicy", env, verbose=1, device="cpu")

# (Optional) Define callbacks for saving checkpoints and evaluating performance:
# Save a checkpoint every 1000 steps.
checkpoint_callback = CheckpointCallback(
    save_freq=1000, 
    save_path='./ppo_continuous_checkpoints/',
    name_prefix='ppo_model'
)

# Create a separate evaluation environment.
eval_env = gym.make(env_id)
# Evaluate the model every 500 steps and save the best model.
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path='./ppo_continuous_best_model/',
    log_path='./ppo_continuous_eval_logs/',
    eval_freq=500,
    deterministic=True,
    render=False
)

# Add progress bar callback
progress_bar = ProgressBarCallback()

# Train the PPO agent for a total of 1,000,000 timesteps
model.learn(total_timesteps=1000000, callback=[checkpoint_callback, eval_callback, progress_bar])

# Save the final trained model
model.save("bipedal_walker_ppo")

# Evaluate the trained model over 10 episodes
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward} +/- {std_reward}")
