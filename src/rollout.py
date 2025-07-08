import torch
import gymnasium as gym
import numpy as np

from dataclasses import dataclass
from src.policy import (
    Policy,
    action_pt_to_env,
    ContinuousPolicy,
    DiscretePolicy,
)
from src.critic import Critic


@dataclass
class Episode:
    states: torch.Tensor
    all_states: torch.Tensor
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

    states = torch.zeros(env.num_envs, timesteps + 1, obs.shape[-1])
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

    states[:, -1] = torch.as_tensor(obs)

    return (
        Episode(
            states=states[:, :-1],
            all_states=states,
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


@torch.no_grad()
def get_gae_advantages(episode: Episode, critic: Critic, gamma: float, lmbda: float):
    advantages = torch.zeros_like(episode.rewards)
    values = critic(episode.all_states)
    deltas = (
        episode.rewards
        + gamma * values[:, 1:] * (1 - episode.dones.float())
        - values[:, :-1]
    )
    # values = critic(episode.states)

    # V = torch.zeros(episode.n_envs)
    A = torch.zeros(episode.n_envs)
    for t in range(episode.timesteps - 1, -1, -1):
        # delta = (
        #     episode.rewards[:, t]
        #     + gamma * V * (1 - episode.dones[:, t].float())
        #     - values[:, t]
        # )
        A = deltas[:, t] + gamma * lmbda * A * (1 - episode.dones[:, t].float())
        # V = values[:, t]
        advantages[:, t] = A

    return advantages, values[:, :-1]
