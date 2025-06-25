import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import matplotlib.pyplot as plt
from tqdm import tqdm
from gymnasium.wrappers import RecordVideo

torch.manual_seed(42)

# Set up device
# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
device = torch.device("cpu")
print(f"Using device: {device}")

# Initialise the environment
# env = gym.make("LunarLander-v3", render_mode="human")
env = gym.make("LunarLander-v3")
env.reset(seed=42)

NUM_EPOCHS = 10000
GAMMA = 0.95
LR = 3e-4
T = 1000


class PolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 4)
        )

    def forward(self, x: torch.Tensor):
        return self.mlp(x)

    def sample(self, x: torch.Tensor):
        logits = self(x)
        probs = F.softmax(logits, dim=-1)
        log_probs = torch.log(probs)
        dist = torch.distributions.Categorical(probs=probs)
        action = dist.sample()

        return {"action": action, "log_probs": log_probs}


def sample_episode(policy: PolicyNet, env, max_samples: int = 250):
    observation, info = env.reset()

    actions = []
    rewards = []
    log_probs_list = []

    for i in range(max_samples):
        observation = torch.from_numpy(observation).unsqueeze(0).to(device)  # (1, 8)
        sample = policy.sample(observation)
        action, log_probs = sample["action"], sample["log_probs"]
        action = action[0].item()
        observation, reward, terminated, truncated, info = env.step(action)

        actions.append(action)
        rewards.append(reward)
        log_probs_list.append(log_probs)

        if terminated:
            break

    return {
        "actions": torch.tensor(actions, device=device),
        "rewards": torch.tensor(rewards, device=device, dtype=torch.float32),
        "log_probs": torch.cat(log_probs_list, dim=0),
    }


policy = PolicyNet().to(device)
optimizer = torch.optim.Adam(policy.parameters(), lr=LR)
losses = []
total_rewards = []
pbar = tqdm(range(NUM_EPOCHS))

for epoch in pbar:
    episode = sample_episode(policy, env, max_samples=T)
    actions, rewards, log_probs = (
        episode["actions"],
        episode["rewards"],
        episode["log_probs"],
    )
    T = actions.shape[0]

    action_log_probs = log_probs[torch.arange(T, device=device), actions]
    total_reward = (rewards * GAMMA ** torch.arange(T, device=device)).sum()
    total_rewards.append(total_reward.item())

    R = torch.tensor(0.0, device=device)
    loss = torch.tensor(0.0, device=device)

    for t in range(T - 1, -1, -1):
        R = rewards[t] + GAMMA * R
        action_log_prob = action_log_probs[t]
        loss += -((R - rewards.mean()) * action_log_prob)

    loss /= T
    losses.append(loss.item())
    pbar.set_postfix(loss=loss.item(), total_reward=total_reward.item())

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
    optimizer.step()


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


# Plot the training curves
plot_training_curves(
    total_rewards, "Total Reward vs Epochs", "Total Reward", "rewards.png"
)
plot_training_curves(losses, "Loss vs Epochs", "Loss", "losses.png")

# Record an evaluation video
eval_env = gym.make("LunarLander-v3", render_mode="rgb_array")
eval_env = RecordVideo(
    eval_env,
    "videos",
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
        observation = torch.from_numpy(observation).unsqueeze(0).to(device)
        action = policy.sample(observation)["action"][0].item()
        observation, reward, terminated, truncated, _ = eval_env.step(action)
        total_reward += reward
        done = terminated or truncated

    print(f"\nEvaluation episode total reward: {total_reward:.2f}")

eval_env.close()
