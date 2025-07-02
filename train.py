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
import time
import numpy as np

from dataclasses import dataclass
from src.utils import save_json, record_episode, evaluate_policy, plot_training_rewards
from src.policy import (
    Policy,
    action_pt_to_env,
    create_discrete_policy,
    ContinuousPolicy,
    DiscretePolicy,
)


def get_episode_batch(
    env: gym.Env, obs: np.ndarray, policy: Policy, batch_size: int, gamma: float
):
    states = []
    next_states = []
    actions, rewards, log_probs, dones, returns = [], [], [], [], []

    obs, _ = env.reset()
    for i in range(batch_size):
        obs = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        # states.append(obs)
        action, log_prob = policy(obs)
        obs, reward, done, _, __ = env.step(action_pt_to_env(action, env))

        # next_states.append(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))
        # actions.append(action)
        # log_probs.append(log_prob)
        # rewards.append(float(reward))
        # dones.append(int(done))

        if done:
            # break
            obs, _ = env.reset()

    # calculate g_t = r_t + 1 + gamma * r_t+1 for each timestep
    R = 0
    for reward, done in zip(rewards[::-1], dones[::-1]):
        R = reward + gamma * R * (1 - done)
        returns.insert(0, R)

    return {
        # "states": torch.cat(states, dim=0),
        # "next_states": torch.cat(next_states, dim=0),
        # "actions": torch.stack(actions),
        # "rewards": torch.tensor(rewards),
        # "dones": torch.tensor(dones, dtype=torch.bool),
        # "log_probs": torch.stack(log_probs),
        # "returns": torch.tensor(returns),
    }


@dataclass
class Episode:
    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    log_probs: torch.Tensor
    timesteps: int
    n_envs: int


@torch.no_grad()
def rollout_episode(
    env: gym.vector.AsyncVectorEnv, obs: np.ndarray, policy: Policy, timesteps: int
) -> tuple[Episode, np.ndarray]:

    states = torch.zeros(env.num_envs, timesteps, obs.shape[-1])
    if isinstance(policy, DiscretePolicy):
        actions = torch.zeros(env.num_envs, timesteps, dtype=torch.int32)
    elif isinstance(policy, ContinuousPolicy):
        actions = torch.zeros(env.num_envs, timesteps, policy.n_acts)
    rewards = torch.zeros(env.num_envs, timesteps)
    dones = torch.zeros(env.num_envs, timesteps, dtype=torch.bool)
    log_probs = torch.zeros(env.num_envs, timesteps)

    for t in range(timesteps):
        obs = torch.as_tensor(obs)
        action, log_prob = policy(obs)
        states[:, t] = obs
        actions[:, t] = action
        log_probs[:, t] = log_prob
        obs, reward, terminated, truncated, _ = env.step(action_pt_to_env(action, env))

        rewards[:, t] = torch.as_tensor(reward)
        dones[:, t] = torch.as_tensor(terminated | truncated)

    return (
        Episode(
            states=states,
            actions=actions,
            rewards=rewards,
            dones=dones,
            log_probs=log_probs,
            timesteps=timesteps,
            n_envs=env.num_envs,
        ),
        obs,
    )


def get_returns(episode: Episode, gamma: float):
    returns = torch.zeros_like(episode.rewards)
    R = torch.zeros(episode.n_envs)
    for t in range(episode.timesteps - 1, -1, -1):
        R = episode.rewards[:, t] + gamma * R * (1 - episode.dones[:, t].float())
        returns[:, t] = R

    return returns


def create_n_envs(env_id: str, env_kwargs: dict, n_envs: int):
    def make_env():
        return gym.make(env_id, **env_kwargs)

    return gym.vector.AsyncVectorEnv([make_env for _ in range(n_envs)])


def train_reinforce(
    env_id: str,
    env_kwargs: dict,
    policy: Policy,
    gamma: float,
    lr: float,
    timesteps: int,
    n_epochs: int,
    save_dir: str,
    eval_every: int = 100,
    log_every: int = 50,
    seed: int = None,
    n_envs: int = 1,
):
    env = create_n_envs(env_id, env_kwargs, n_envs)
    print("env: ", env)
    obs, _ = env.reset(seed=seed)
    eval_env = gym.make(env_id, **env_kwargs)

    # torch.manual_seed(seed)
    os.makedirs(save_dir, exist_ok=True)

    print(f"[INFO] REINFORCE: env={env_id} policy={policy.config}")
    print(f"[INFO] Saving to {save_dir}")
    print(
        f"[INFO] Training with gamma={gamma}, lr={lr}, timesteps={timesteps}, n_epochs={n_epochs}"
    )
    save_json(
        {
            "gamma": gamma,
            "lr": lr,
            "timesteps": timesteps,
            "n_epochs": n_epochs,
            "env_id": env_id,
            "policy": policy.config,
        },
        os.path.join(save_dir, "config.json"),
    )

    # record_episode(
    #     env_id=env_id,
    #     policy=policy,
    #     save_dir=os.path.join(save_dir, "videos"),
    #     prefix="no_train",
    # )

    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    train_rewards = []
    pbar = tqdm(range(n_epochs))

    for i in pbar:
        start_time = time.time()
        episode, obs = rollout_episode(
            env,
            obs=obs,
            policy=policy,
            timesteps=timesteps,
        )
        returns = get_returns(episode, gamma)
        log_probs = policy.get_log_probs(episode.states, episode.actions)

        pbar.set_postfix(returns=returns.mean().item())

        # normalize returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        policy_loss = -(returns * log_probs).view(-1).mean()

        # print("policy_loss: ", policy_loss.shape)
        # print("log_probs: ", log_probs)
        # print("episode.actions: ", episode.actions)
        # print("returns: ", returns)
        # print("episode: ", episode)
        # episode_batch = get_episode_batch(
        #     env=env, policy=policy, gamma=gamma, batch_size=timesteps
        # )
        end_time = time.time()

        # print("episode_batch: ", episode_batch)
        # print("time: ", end_time - start_time, "s")

        # raise RuntimeError

        # returns, log_probs = episode_batch["returns"], episode_batch["log_probs"]
        # _, log_probs = policy(episode_batch["states"])

        # policy gradient with baseline = (Q(s, a) - b(s)) * ∇log π(a | s)
        # we calculate -(Q(s, a) - b(s)) * log π(a | s), then do gradient descent which moves policy parameters
        # in direction increase expected returns. θ = θ + α * (Q(s, a) - b(s)) * ∇log π(a | s)

        # returns = (returns - returns.mean()) / (
        #     returns.std() + 1e-8
        # )  # this is equivalent to multiplying by a scalar - doesn't change direction of gradient and also reduces varaiance

        # policy_loss = -(returns * log_probs).mean()
        # print("policy_loss: ", policy_loss)
        # raise RuntimeError
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        if ((i + 1) % log_every) == 0:
            avg_reward = evaluate_policy(eval_env, policy, batch_size=4)
            train_rewards.append(avg_reward)
            print(f"[Epoch {i + 1}] Reward={avg_reward:.2f}")

        if ((i + 1) % eval_every) == 0:
            record_episode(
                env_id=env_id,
                policy=policy,
                save_dir=os.path.join(save_dir, "videos"),
                prefix=f"eval-{i}",
            )

    plot_training_rewards(train_rewards, log_every, n_epochs, save_dir, env)
    for i in range(4):
        record_episode(
            env_id=env_id,
            policy=policy,
            save_dir=os.path.join(save_dir, "videos"),
            prefix=f"final-{i}",
        )

    torch.save(policy.state_dict(), os.path.join(save_dir, "policy.pth"))


if __name__ == "__main__":
    env = gym.make("LunarLander-v3")
    torch.manual_seed(42)
    policy = create_discrete_policy(env, n_hidden=16)
    train_reinforce(
        env_id="LunarLander-v3",
        env_kwargs={},
        n_envs=8,
        policy=policy,
        gamma=0.99,
        lr=1e-2,
        timesteps=512,
        n_epochs=2000,
        save_dir="./training_runs/mook/mook",
        eval_every=100,
        log_every=25,
        seed=42,
    )
