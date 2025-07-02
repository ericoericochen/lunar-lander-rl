import torch
import torch.nn as nn
import gymnasium as gym


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
            nn.Linear(n_hidden, 1),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.mlp(obs).view(*obs.shape[:-1])
