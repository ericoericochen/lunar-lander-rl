import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Callable

from ..utils import save_json, record_episode, evaluate_policy
from ..policy import Policy


def get_episode_batch(env: gym.Env, policy: Policy, batch_size: int, gamma: float):
    actions, rewards, log_probs, dones, returns = [], [], [], [], []

    obs, _ = env.reset()
    for i in range(batch_size):
        obs = torch.as_tensor(obs, dtype=torch.float32)
        action, log_prob = policy(obs)
        obs, reward, done, _, __ = env.step(action.item())

        actions.append(action)
        log_probs.append(log_prob)
        rewards.append(float(reward))
        dones.append(int(done))

        if done:
            # break
            obs, _ = env.reset()

    # calculate g_t = r_t + 1 + gamma * r_t+1 for each timestep
    R = 0
    for reward, done in zip(rewards[::-1], dones[::-1]):
        R = reward + gamma * R * (1 - done)
        returns.insert(0, R)

    return {
        "actions": torch.stack(actions),
        "rewards": torch.tensor(rewards),
        "dones": torch.tensor(dones, dtype=torch.bool),
        "log_probs": torch.stack(log_probs),
        "returns": torch.tensor(returns),
    }


def train_reinforce(
    env: gym.Env,
    policy: Policy,
    gamma: float,
    lr: float,
    batch_size: int,
    n_epochs: int,
    save_dir: str,
    with_baseline: bool = False,
    eval_every: int = 100,
    log_every: int = 50,
    seed: int = None,
):
    torch.manual_seed(seed)
    os.makedirs(save_dir, exist_ok=True)
    print(f"[INFO] REINFORCE: env={env.spec.id} policy={policy.config}")
    print(f"[INFO] Saving to {save_dir}")
    print(
        f"[INFO] Training with gamma={gamma}, lr={lr}, batch_size={batch_size}, n_epochs={n_epochs}, with_baseline={with_baseline}"
    )
    save_json(
        {
            "gamma": gamma,
            "lr": lr,
            "batch_size": batch_size,
            "n_epochs": n_epochs,
            "with_baseline": with_baseline,
            "env_id": env.spec.id,
            "policy": policy.config,
        },
        os.path.join(save_dir, "config.json"),
    )

    record_episode(
        env_id=env.spec.id,
        policy=policy,
        save_dir=os.path.join(save_dir, "videos"),
        prefix="no_train",
    )

    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    train_rewards = []

    for i in tqdm(range(n_epochs)):
        episode_batch = get_episode_batch(
            env=env, policy=policy, gamma=gamma, batch_size=batch_size
        )

        returns, log_probs = episode_batch["returns"], episode_batch["log_probs"]

        # policy gradient with baseline = (Q(s, a) - b(s)) * ∇log π(a | s)
        # we calculate -(Q(s, a) - b(s)) * log π(a | s), then do gradient descent which moves policy parameters
        # in direction increase expected returns. θ = θ + α * (Q(s, a) - b(s)) * ∇log π(a | s)
        if with_baseline:
            returns = (returns - returns.mean()) / (
                returns.std() + 1e-8
            )  # this is equivalent to multiplying by a scalar - doesn't change direction of gradient and also reduces varaiance

        policy_loss = -(returns * log_probs).mean()
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        if ((i + 1) % log_every) == 0:
            avg_reward = evaluate_policy(env, policy, batch_size=4)
            train_rewards.append(avg_reward)
            print(f"[Epoch {i + 1}] Reward={avg_reward:.2f}")

        if ((i + 1) % eval_every) == 0:
            record_episode(
                env_id=env.spec.id,
                policy=policy,
                save_dir=os.path.join(save_dir, "videos"),
                prefix=f"eval-{i}",
            )

    plot_training_rewards(train_rewards, log_every, n_epochs, save_dir, env)
    for i in range(4):
        record_episode(
            env_id=env.spec.id,
            policy=policy,
            save_dir=os.path.join(save_dir, "videos"),
            prefix=f"final-{i}",
        )


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
