import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from tqdm import tqdm


N_HIDDEN = 32
N_EPOCHS = 10000
BATCH_SIZE = 128
GAMMA = 0.99


def record_episode(
    env: gym.Env, actor: "Actor", save_dir: str, prefix: str, seed: int = None
):
    env = RecordVideo(
        env,
        save_dir,
        episode_trigger=lambda x: True,
        name_prefix=prefix,
    )
    obs, _ = env.reset(seed=seed)

    rewards = []
    min_action, max_action = min_max_action_bound(env)

    while True:
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action, _ = actor(obs_tensor)
            action = action.clamp(min=min_action, max=max_action).squeeze(0)
            action = action.detach().numpy()

        obs, reward, done, truncated, _ = env.step(action)
        rewards.append(float(reward))
        if done or truncated:
            break

    total_reward = sum(rewards)
    print(f"\nEvaluation episode total reward: {total_reward:.2f}")
    env.close()
    return total_reward


def get_obs_dim_and_act_dim(env: gym.Env):
    obs_dim = env.observation_space.shape[0]
    if hasattr(env.action_space, "n"):
        n_acts = env.action_space.n
    else:
        n_acts = env.action_space.shape[0]
    return obs_dim, n_acts


def min_max_action_bound(env: gym.Env):
    min_action, max_action = env.action_space.low, env.action_space.high
    return torch.as_tensor(min_action), torch.as_tensor(max_action)


def sample_action(
    actor: "Actor",
    obs: torch.Tensor,
    min_action: torch.Tensor,
    max_action: torch.Tensor,
):
    pass


# continuous action space actor
class Actor(nn.Module):
    def __init__(self, obs_dim: int, n_hidden: int, n_acts: int):
        super().__init__()
        self.n_acts = n_acts
        self.mlp = nn.Sequential(
            nn.Linear(obs_dim, n_hidden), nn.ReLU(), nn.Linear(n_hidden, n_acts * 2)
        )

    def forward(self, obs: torch.Tensor):
        params = self.mlp(obs).view(-1, 2, self.n_acts)
        mean, log_std = torch.unbind(params, dim=1)
        std = log_std.exp()
        dist = torch.distributions.Normal(loc=mean, scale=std)
        action = dist.rsample()
        log_prob = dist.log_prob(action)

        print("log_prob", log_prob)
        raise RuntimeError
        return action, log_prob


class Critic(nn.Module):
    def __init__(self, obs_dim: int, n_acts: int, n_hidden: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(obs_dim + n_acts, n_hidden), nn.ReLU(), nn.Linear(n_hidden, 1)
        )

    def forward(self, obs: torch.Tensor, actions: torch.Tensor):
        x = torch.cat((obs, actions), dim=1)
        return self.mlp(x)


def get_episode_batch(env: gym.Env, actor: Actor, batch_size: int):
    obs, _ = env.reset()

    observations = []
    actions = []
    rewards = []
    log_probs = []

    min_action, max_action = min_max_action_bound(env)

    for i in range(500):
        obs = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        observations.append(obs)

        # sample action
        action, log_prob = actor(obs)
        actions.append(action)
        log_probs.append(log_prob)
        action = action.clamp(min=min_action, max=max_action).squeeze(0)
        action = action.detach().numpy()

        # get reward
        obs, reward, done, _, __ = env.step(action)
        rewards.append(float(reward))

        if done or i == 499:
            obs = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            observations.append(obs)
            break

    return {
        "obs": torch.cat(observations, dim=0),
        "actions": torch.cat(actions, dim=0),
        "rewards": torch.tensor(rewards),
        "log_probs": torch.cat(log_probs, dim=0),
    }


def train_actor_critic(env: gym.Env, actor: Actor, critic: Critic):
    # actor_optimizer = torch.optim.Adam(actor.parameters(), lr=1e-3)
    # critic_optimizer = torch.optim.Adam(critic.parameters(), lr=1e-2)
    optimizer = torch.optim.Adam([*actor.parameters(), *critic.parameters()], lr=1e-2)

    for i in tqdm(range(N_EPOCHS), desc="Training"):
        episode_batch = get_episode_batch(env, actor, BATCH_SIZE)
        obs = episode_batch["obs"]
        actions = episode_batch["actions"]
        rewards = episode_batch["rewards"]
        log_probs = episode_batch["log_probs"]

        # Print total reward every 50 epochs
        if (i + 1) % 50 == 0:
            total_reward = rewards.sum().item()
            print(f"\nEpoch {i + 1}: Total reward = {total_reward:.2f}")

        # update actor
        curr_obs = obs[:-1, ...]
        next_obs = obs[1:, ...]
        assert curr_obs.shape[0] == actions.shape[0]
        # q = critic(curr_obs, actions.detach())
        q = critic(curr_obs, actions)

        policy_loss = -(q.detach() * log_probs).mean()
        # actor_optimizer.zero_grad()
        # policy_loss.backward()
        # actor_optimizer.step()

        # update critic
        done = torch.zeros_like(rewards)
        done[-1] = 1

        action_space = actions.shape[1]
        next_actions = torch.cat((actions[1:], torch.ones((1, action_space))), dim=0)

        critic_target = rewards + GAMMA * critic(next_obs, next_actions).squeeze(1) * (
            1 - done
        )
        # q = critic(curr_obs, actions.detach()).squeeze(1)
        critic_loss = F.mse_loss(critic_target.detach(), q.view(-1))

        loss = policy_loss + critic_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # critic_optimizer.zero_grad()
        # critic_loss.backward()
        # critic_optimizer.step()


if __name__ == "__main__":
    env = gym.make("BipedalWalker-v3", render_mode="rgb_array")
    obs_dim, n_acts = get_obs_dim_and_act_dim(env)
    actor = Actor(obs_dim, N_HIDDEN, n_acts)
    critic = Critic(obs_dim, n_acts, N_HIDDEN)

    train_actor_critic(env, actor, critic)

    # Record the trained agent
    print("\nTraining completed! Recording final performance...")
    final_reward = record_episode(env, actor, "videos", "trained-agent")
    print(f"Final agent total reward: {final_reward:.2f}")
