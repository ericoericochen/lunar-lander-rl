import json
import os
import torch
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import matplotlib.pyplot as plt

from .policy import Policy, action_pt_to_env


def save_json(data: dict, path: str):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def create_n_envs(env_id: str, env_kwargs: dict, n_envs: int):
    def make_env():
        return gym.make(env_id, **env_kwargs)

    return gym.vector.AsyncVectorEnv([make_env for _ in range(n_envs)])


def evaluate_policy(env: gym.Env, policy: Policy, batch_size: int = 3):
    total_reward = 0
    for _ in range(batch_size):
        obs, _ = env.reset()
        policy.eval()
        episode_reward = 0

        with torch.no_grad():
            done = truncated = False
            while not (done or truncated):
                obs_tensor = torch.as_tensor(obs)
                action, _ = policy(obs_tensor)
                obs, reward, done, truncated, _ = env.step(
                    action_pt_to_env(action, env)
                )
                episode_reward += float(reward)

        total_reward += episode_reward

    return total_reward / batch_size


def record_episode(
    env_id: str,
    policy: torch.nn.Module,
    save_dir: str,
    env_kwargs: dict = {},
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
    eval_env = gym.make(env_id, render_mode="rgb_array", **env_kwargs)
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
            # obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            obs_tensor = torch.as_tensor(obs)
            action, _ = policy(obs_tensor)

            obs, reward, done, truncated, _ = eval_env.step(
                action_pt_to_env(action, eval_env)
            )
            rewards.append(float(reward))
            step_count += 1

            if done or truncated:
                break

        total_reward = sum(rewards)
        print(f"\nEvaluation episode total reward: {total_reward:.2f}")
        print(f"Episode length: {len(rewards)} steps")

    eval_env.close()
    return total_reward, rewards


def plot_training_rewards(
    rewards: list, log_every: int, num_epochs: int, save_dir: str, env: gym.Env
):
    plt.figure(figsize=(10, 6))
    epochs = range(log_every, num_epochs + 1, log_every)

    plt.plot(epochs, rewards, "b-", label="Rewards")

    if env.spec.reward_threshold is not None:
        plt.axhline(
            y=env.spec.reward_threshold,
            color="g",
            linestyle="--",
            label=f"Solved ({env.spec.reward_threshold})",
        )

    plt.xlabel("Epoch")
    plt.ylabel("Episode Reward")
    plt.title(f"Training Progress - {env.spec.id}")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "training_rewards.png"))
    plt.close()
