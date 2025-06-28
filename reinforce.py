import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import matplotlib.pyplot as plt
from tqdm import tqdm
from gymnasium.wrappers import RecordVideo
import random

NUM_EPOCHS = 10000
GAMMA = 0.999
# GAMMA = 1
LR = 8e-3
T = 500
LOG_EVERY = 100
SEED = 0

env = gym.make("LunarLander-v3")
# env = gym.make("CartPole-v1")
env.reset(seed=SEED)
torch.manual_seed(SEED)


class PolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        # self.mlp = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 2))
        self.mlp = nn.Sequential(nn.Linear(8, 32), nn.ReLU(), nn.Linear(32, 4))

    def forward(self, x: torch.Tensor):
        return self.mlp(x)

    def sample(self, x: torch.Tensor, env: gym.Env, epsilon: float = 0.1):
        p = random.random()
        logits = self(x)
        probs = F.softmax(logits, dim=-1)
        log_probs = torch.log(probs)
        dist = torch.distributions.Categorical(probs=probs)

        if p < epsilon:
            action = env.action_space.sample()
            log_probs = dist.log_prob(torch.tensor([action]))
            return {"action": torch.tensor([action]), "log_probs": log_probs}

        action = dist.sample()
        log_probs = log_probs[0, action]
        return {"action": action, "log_probs": log_probs}


policy = PolicyNet()
optimizer = torch.optim.Adam(policy.parameters(), lr=LR)


def sample_policy(
    policy: PolicyNet, observation: torch.Tensor
) -> tuple[int, torch.Tensor]:
    """Samples a discrete action A ~ π(a | s) and returns the action and its log probability"""
    logits = policy(observation)
    probs = F.softmax(logits, dim=-1)
    dist = torch.distributions.Categorical(probs=probs)
    action = dist.sample()
    log_prob = dist.log_prob(action)
    return action[0].item(), log_prob[0]


def sample_episode(env: gym.Env, policy: PolicyNet, max_t: int = T):
    observation, _ = env.reset()
    rewards = []
    log_probs = []

    for t in range(max_t):
        observation = torch.from_numpy(observation).to(torch.float32).unsqueeze(0)

        action, log_prob = sample_policy(policy, observation)
        observation, reward, done, _, __ = env.step(action)
        rewards.append(float(reward))
        log_probs.append(log_prob)

        if done:
            break

    return {
        "rewards": rewards,
        "log_probs": torch.stack(log_probs),
    }


def calculate_qs(rewards: list[float], gamma: float):
    # return [sum(rewards)] * len(rewards)
    R = 0
    qs = []

    for reward in reversed(rewards):
        R = reward + gamma * R
        qs.insert(0, R)
    return qs


total_rewards = []
for epoch in tqdm(range(NUM_EPOCHS)):
    policy_loss = torch.tensor(0.0)
    episode = sample_episode(env, policy, T)
    rewards, log_probs = episode["rewards"], episode["log_probs"]
    qs = torch.tensor(calculate_qs(rewards, GAMMA))
    policy_loss += -(qs * log_probs).mean()
    # policy_loss += -((qs - sum(rewards) / len(rewards)) * log_probs).mean()

    optimizer.zero_grad()
    policy_loss.backward()
    optimizer.step()

    if ((epoch + 1) % LOG_EVERY) == 0:
        total_rewards.append(sum(rewards))
        print(f"[Epoch {epoch}] reward = {sum(rewards)}")


def plot_training_curves(metrics: list, title: str, ylabel: str, save_path: str):
    """Plot and save training curves.

    Args:
        metrics: List of values to plot
        title: Title of the plot
        ylabel: Label for y-axis
        save_path: Where to save the plot
    """
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(metrics)), metrics)
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()


plot_training_curves(
    total_rewards, "Total Reward vs Epochs", "Total Reward", "rewards_mook.png"
)


eval_env = gym.make("LunarLander-v3", render_mode="rgb_array")
# eval_env = gym.make("CartPole-v1", render_mode="rgb_array")
eval_env = RecordVideo(
    eval_env,
    "videosmook",
    episode_trigger=lambda x: True,  # Record every episode
    name_prefix="lunar_lander",
)
eval_env.reset(seed=42)

# Evaluate for one episode
policy.eval()  # Set to evaluation mode
with torch.no_grad():
    observation, _ = eval_env.reset()
    done = False
    total_reward = 0

    while not done:
        observation = torch.from_numpy(observation).unsqueeze(0)
        action, _ = sample_policy(policy, observation)
        observation, reward, terminated, truncated, _ = eval_env.step(action)
        total_reward += reward
        done = terminated or truncated

    print(f"\nEvaluation episode total reward: {total_reward:.2f}")

eval_env.close()

torch.save(policy.state_dict(), "policy.pth")
