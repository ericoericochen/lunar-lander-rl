import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import matplotlib.pyplot as plt
from tqdm import tqdm


def create_mlp(obs_dim: int, n_hidden: int, n_acts: int):
    mlp = nn.Sequential(
        nn.Linear(obs_dim, n_hidden),
        nn.ReLU(),
        nn.Linear(n_hidden, n_acts),
    )
    mlp.n_hidden = n_hidden
    return mlp


def create_mlp_from_env(env: gym.Env, n_hidden: int):
    obs_dim = env.observation_space.shape[0]
    n_acts = env.action_space.n
    return create_mlp(obs_dim, n_hidden, n_acts)


def save_config(save_dir: str, config: dict):
    config_path = os.path.join(save_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)


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


class RLAgent:
    def __init__(self, env: gym.Env, policy: nn.Module):
        super().__init__()
        self.env = env
        self.policy = policy

    def sample_action(self, obs: torch.Tensor, greedy: bool = False):
        logits = self.policy(obs)
        dist = torch.distributions.Categorical(logits=logits)
        if greedy:
            action = logits.argmax(dim=-1)
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob[0]

    def generate_episode_batch(self, gamma: float):
        gt = []
        actions = []
        rewards = []
        log_probs = []
        t = 0

        # generate an episode by sampling action from policy net
        obs, _ = self.env.reset()
        while True:
            obs = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            action, log_prob = self.sample_action(obs)
            obs, reward, done, truncated, __ = self.env.step(action)

            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(float(reward))

            if done or t == 500:
                # calculate g_t = r_t + 1 + gamma * r_t+1 for each timestep
                R = 0
                for r in reversed(rewards):
                    R = r + gamma * R
                    gt.insert(0, R)

                break

            t += 1

        return {
            "actions": torch.tensor(actions),
            "log_probs": torch.stack(log_probs),
            "rewards": torch.tensor(rewards),
            "gt": torch.tensor(gt),
        }

    def train(
        self,
        num_epochs: int,
        lr: float,
        batch_size: int,
        gamma: float,
        log_every: int,
        save_dir: str,
        seed: int = None,
    ):
        print("[INFO] Training RL Agent")

        os.makedirs(save_dir, exist_ok=True)
        config = {
            "num_epochs": num_epochs,
            "gamma": gamma,
            "learning_rate": lr,
            "batch_size": batch_size,
            "n_hidden": self.policy.n_hidden,
            "seed": seed,
            "env_id": self.env.spec.id,
        }
        save_config(save_dir, config)
        self.record_episode(os.path.join(save_dir, "videos"), "no_train", seed)

        train_rewards = []
        optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        for epoch in tqdm(range(num_epochs)):
            epoch_rewards = 0
            policy_loss = torch.tensor(0.0)

            for b in range(batch_size):
                episode_batch = self.generate_episode_batch(gamma)
                gt, log_probs, rewards = (
                    episode_batch["gt"],
                    episode_batch["log_probs"],
                    episode_batch["rewards"],
                )
                # vanilla policy gradient with baseline
                baseline = rewards.mean()
                policy_loss += -((gt - baseline) * log_probs).mean()
                epoch_rewards += rewards.sum().item()

            policy_loss /= batch_size
            epoch_rewards /= batch_size

            if ((epoch + 1) % log_every) == 0:
                train_rewards.append(epoch_rewards)
                print(f"[Epoch {epoch + 1}] episode_reward={epoch_rewards}")

            # backprop
            optimizer.zero_grad()
            policy_loss.backward()
            optimizer.step()

        self.record_episode(os.path.join(save_dir, "videos"), "final", seed)
        torch.save(self.policy.state_dict(), os.path.join(save_dir, "policy.pth"))
        plot_training_rewards(train_rewards, log_every, num_epochs, save_dir, self.env)

    def record_episode(
        self,
        save_dir: str,
        prefix: str,
        seed: int = None,
        greedy_sampling: bool = False,
    ):
        eval_env = gym.make(self.env.spec.id, render_mode="rgb_array")
        eval_env = RecordVideo(
            eval_env,
            save_dir,
            episode_trigger=lambda x: True,
            name_prefix=prefix,
        )
        obs, _ = eval_env.reset(seed=seed)

        self.policy.eval()
        with torch.no_grad():
            rewards = []
            while True:
                obs = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
                action, _ = self.sample_action(obs, greedy_sampling)
                obs, reward, done, truncated, _ = eval_env.step(action)
                rewards.append(float(reward))
                if done or truncated:
                    break

            print(f"\nEvaluation episode total reward: {sum(rewards):.2f}")
        eval_env.close()
