import torch
import torch.nn as nn
import gymnasium as gym
import numpy as np


def create_critic(env: gym.Env, n_hidden: int):
    obs_dim = env.observation_space.shape[0]
    return Critic(obs_dim, n_hidden)


class Critic(nn.Module):
    def __init__(self, obs_dim: int, n_hidden: int):
        super().__init__()
        self.obs_dim = obs_dim
        self.n_hidden = n_hidden
        self.mlp = nn.Sequential(
            nn.Linear(obs_dim, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, 1),
        )

        def orthogonal_init(module, gain):
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain)
                nn.init.constant_(module.bias, 0)

        self.mlp[0].apply(lambda m: orthogonal_init(m, np.sqrt(2)))  # first hidden
        self.mlp[2].apply(lambda m: orthogonal_init(m, np.sqrt(2)))  # second hidden
        self.mlp[4].apply(lambda m: orthogonal_init(m, 1.0))  # output

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.mlp(obs).view(*obs.shape[:-1])
