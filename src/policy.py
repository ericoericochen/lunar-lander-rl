import torch
import torch.nn as nn
import gymnasium as gym
from typing import Union


class DiscretePolicy(nn.Module):
    def __init__(self, obs_dim: int, n_hidden: int, n_acts: int):
        super().__init__()
        self.obs_dim = obs_dim
        self.n_hidden = n_hidden
        self.n_acts = n_acts

        self.mlp = nn.Sequential(
            nn.Linear(obs_dim, n_hidden),
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

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.mlp(obs)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob


class ContinuousPolicy(nn.Module):
    pass


def create_discrete_policy(env: gym.Env, n_hidden: int):
    obs_dim = env.observation_space.shape[0]
    n_acts = int(env.action_space.n)

    return DiscretePolicy(
        obs_dim=obs_dim,
        n_hidden=n_hidden,
        n_acts=n_acts,
    )


Policy = Union[DiscretePolicy, ContinuousPolicy]
