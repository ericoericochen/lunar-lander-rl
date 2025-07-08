import os
import torch
import gymnasium as gym
from tqdm import tqdm

from ..utils import (
    save_json,
    record_episode,
    evaluate_policy,
    plot_training_rewards,
    create_n_envs,
)
from ..policy import (
    Policy,
)
from ..rollout import rollout_episode, get_returns


def train_reinforce(
    env_id: str,
    env_kwargs: dict,
    policy: Policy,
    save_dir: str,
    gamma: float = 0.99,
    lr: float = 3e-4,
    timesteps: int = 256,
    n_epochs: int = 1000,
    eval_every: int = 100,
    log_every: int = 50,
    seed: int = 42,
    n_envs: int = 1,
):
    env = create_n_envs(env_id, env_kwargs, n_envs)
    obs, _ = env.reset(seed=seed)
    eval_env = gym.make(env_id, **env_kwargs)

    os.makedirs(save_dir, exist_ok=True)

    print(f"[INFO] REINFORCE: env={env_id} policy={policy.config}")
    print(f"[INFO] Saving to {save_dir}")
    print(
        f"[INFO] Training with gamma={gamma}, lr={lr}, timesteps={timesteps}, n_epochs={n_epochs}"
    )
    save_json(
        {
            "type": "reinforce",
            "gamma": gamma,
            "lr": lr,
            "timesteps": timesteps,
            "n_epochs": n_epochs,
            "env_id": env_id,
            "n_envs": n_envs,
            "policy": policy.config,
        },
        os.path.join(save_dir, "config.json"),
    )

    record_episode(
        env_id=env_id,
        env_kwargs=env_kwargs,
        policy=policy,
        save_dir=os.path.join(save_dir, "videos"),
        prefix="no_train",
    )

    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    train_rewards = []
    pbar = tqdm(range(n_epochs))

    for i in pbar:
        episode, obs = rollout_episode(
            env,
            obs=obs,
            policy=policy,
            timesteps=timesteps,
        )
        returns = get_returns(episode, gamma)
        log_probs, dist = policy.get_log_probs(episode.states, episode.actions)

        pbar.set_postfix(returns=returns.mean().item())

        # normalize returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # policy gradient with baseline = (Q(s, a) - b(s)) * ∇log π(a | s)
        # we calculate -(Q(s, a) - b(s)) * log π(a | s), then do gradient descent which moves policy parameters
        # in direction increase expected returns. θ = θ + α * (Q(s, a) - b(s)) * ∇log π(a | s)
        policy_loss = -(returns * log_probs).view(-1).mean()

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
                env_kwargs=env_kwargs,
                policy=policy,
                save_dir=os.path.join(save_dir, "videos"),
                prefix=f"eval-{i}",
            )

    plot_training_rewards(train_rewards, log_every, n_epochs, save_dir, eval_env)
    for i in range(4):
        record_episode(
            env_id=env_id,
            env_kwargs=env_kwargs,
            policy=policy,
            save_dir=os.path.join(save_dir, "videos"),
            prefix=f"final-{i}",
        )

    torch.save(policy.state_dict(), os.path.join(save_dir, "policy.pth"))
