import json
import os
import torch
import gymnasium as gym
from gymnasium.wrappers import RecordVideo

from .policy import Policy


def save_json(data: dict, path: str):
    with open(path, "w") as f:
        json.dump(data, f)


def evaluate_policy(env: gym.Env, policy: Policy, batch_size: int = 3):
    total_reward = 0
    for _ in range(batch_size):
        obs, _ = env.reset()
        policy.eval()
        episode_reward = 0

        with torch.no_grad():
            done = truncated = False
            while not (done or truncated):
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
                action, _ = policy(obs_tensor)
                obs, reward, done, truncated, _ = env.step(action.item())
                episode_reward += float(reward)

        total_reward += episode_reward

    return total_reward / batch_size


def record_episode(
    env_id: str,
    policy: torch.nn.Module,
    save_dir: str,
    prefix: str = "episode",
    seed: int = None,
    max_steps: int = 1000,
):
    """
    Record an episode using the given policy and environment.

    Args:
        env_id: The environment ID (e.g., 'LunarLander-v2')
        policy: The policy network to use for action selection
        save_dir: Directory to save the video
        prefix: Prefix for the video filename
        seed: Random seed for reproducibility
        greedy_sampling: Whether to use greedy action selection
        max_steps: Maximum number of steps per episode
    """
    eval_env = gym.make(env_id, render_mode="rgb_array")
    eval_env = RecordVideo(
        eval_env,
        save_dir,
        episode_trigger=lambda x: True,
        name_prefix=prefix,
    )

    obs, _ = eval_env.reset(seed=seed)
    policy.eval()
    with torch.no_grad():
        rewards = []
        step_count = 0

        while step_count < max_steps:
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
            action, _ = policy(obs_tensor)
            action = action.item()

            obs, reward, done, truncated, _ = eval_env.step(action)
            rewards.append(float(reward))
            step_count += 1

            if done or truncated:
                break

        total_reward = sum(rewards)
        print(f"\nEvaluation episode total reward: {total_reward:.2f}")
        print(f"Episode length: {len(rewards)} steps")

    eval_env.close()
    return total_reward, rewards
