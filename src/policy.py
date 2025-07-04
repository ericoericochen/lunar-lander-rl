import torch
import torch.nn as nn
import gymnasium as gym
from typing import Union
from gymnasium.spaces import Discrete, Box, MultiDiscrete
import numpy as np


def get_obs_and_act_dims(env: gym.Env):
    obs_dim = env.observation_space.shape[0]
    if hasattr(env.action_space, "n"):
        n_acts = int(env.action_space.n)
    else:
        n_acts = env.action_space.shape[0]
    return obs_dim, n_acts


def action_pt_to_env(
    action: torch.Tensor,
    env: gym.vector.AsyncVectorEnv,
):

    if isinstance(env.action_space, Discrete):
        return action.detach().numpy()
    elif isinstance(env.action_space, MultiDiscrete):
        return action.detach().numpy()
    elif isinstance(env.action_space, Box):
        # Clip continuous actions to environment's bounds
        action_np = action.detach().numpy()
        return np.clip(action_np, env.action_space.low, env.action_space.high)


class DiscretePolicy(nn.Module):
    def __init__(self, obs_dim: int, n_hidden: int, n_acts: int):
        super().__init__()
        self.obs_dim = obs_dim
        self.n_hidden = n_hidden
        self.n_acts = n_acts

        self.mlp = nn.Sequential(
            nn.Linear(obs_dim, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_acts),
        )

    @property
    def config(self):
        return {
            "type": "DiscretePolicy",
            "args": {
                "obs_dim": self.obs_dim,
                "n_hidden": self.n_hidden,
                "n_acts": self.n_acts,
            },
        }

    def get_log_probs(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        logits = self.mlp(obs)
        dist = torch.distributions.Categorical(logits=logits)
        log_prob = dist.log_prob(actions)
        return log_prob

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.mlp(obs)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob


class ContinuousPolicy(nn.Module):
    def __init__(self, obs_dim: int, n_hidden: int, n_acts: int):
        super().__init__()
        self.obs_dim = obs_dim
        self.n_hidden = n_hidden
        self.n_acts = n_acts

        self.mlp = nn.Sequential(
            nn.Linear(obs_dim, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_acts * 2),
            nn.Tanh(),
        )

        def orthogonal_init(module, gain):
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain)
                nn.init.constant_(module.bias, 0)

        # Apply orthogonal initialization to each layer
        self.mlp[0].apply(lambda m: orthogonal_init(m, np.sqrt(2)))  # first hidden
        self.mlp[2].apply(lambda m: orthogonal_init(m, np.sqrt(2)))  # second hidden
        self.mlp[4].apply(lambda m: orthogonal_init(m, 1.0))  # output

    @property
    def config(self):
        return {
            "type": "ContinuousPolicy",
            "args": {
                "obs_dim": self.obs_dim,
                "n_hidden": self.n_hidden,
                "n_acts": self.n_acts,
            },
        }

    def get_log_probs(self, obs: torch.Tensor, actions: torch.Tensor):
        mean, log_std = self.mlp(obs).split(self.n_acts, dim=-1)
        std = log_std.exp()
        cov = torch.diag_embed(std**2)
        dist = torch.distributions.MultivariateNormal(loc=mean, covariance_matrix=cov)
        log_prob = dist.log_prob(actions)

        return log_prob, dist

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.mlp(obs).split(self.n_acts, dim=-1)
        log_std = torch.clamp(log_std, min=-20, max=2)
        std = log_std.exp()
        cov = torch.diag_embed(std**2)
        dist = torch.distributions.MultivariateNormal(loc=mean, covariance_matrix=cov)
        action = dist.rsample()
        log_prob = dist.log_prob(action)

        return action, log_prob


def create_discrete_policy(env: gym.Env, n_hidden: int):
    obs_dim, n_acts = get_obs_and_act_dims(env)
    return DiscretePolicy(
        obs_dim=obs_dim,
        n_hidden=n_hidden,
        n_acts=n_acts,
    )


def create_continuous_policy(env: gym.Env, n_hidden: int):
    obs_dim, n_acts = get_obs_and_act_dims(env)
    return ContinuousPolicy(
        obs_dim=obs_dim,
        n_hidden=n_hidden,
        n_acts=n_acts,
    )


Policy = Union[DiscretePolicy, ContinuousPolicy]
