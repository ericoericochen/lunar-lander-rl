import os
import torch
import gymnasium as gym
from tqdm import tqdm

from src.utils import (
    record_episode,
    save_json,
    evaluate_policy,
    plot_training_rewards,
    create_n_envs,
)
from src.policy import Policy
from src.rollout import rollout_episode, get_returns


def train_grpo(
    env_id: str,
    env_kwargs: dict,
    policy: Policy,
    lr: float,
    n_epochs: int,
    save_dir: str,
    log_every: int,
    eval_every: int,
    batch_size: int,
    n_envs: int = 1,
    timesteps: int = 512,
    gamma: float = 0.99,
    lmbda: float = 0.95,
    beta: float = 0.1,
    seed: int = None,
    eps: float = 0.2,
):
    env = create_n_envs(env_id, env_kwargs, n_envs)
    obs, _ = env.reset(seed=seed)
    eval_env = gym.make(env_id, **env_kwargs)

    os.makedirs(save_dir, exist_ok=True)
    print(f"[INFO] GRPO: env={env_id} policy={policy.config}")
    print(f"[INFO] Saving to {save_dir}")
    print(
        f"[INFO] Training with gamma={gamma}, lr={lr}, eps={eps}, timesteps={timesteps}, n_epochs={n_epochs}"
    )
    save_json(
        {
            "type": "grpo",
            "gamma": gamma,
            "lmbda": lmbda,
            "eps": eps,
            "beta": beta,
            "lr": lr,
            "timesteps": timesteps,
            "n_epochs": n_epochs,
            "env_id": env_id,
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
        pbar.set_postfix(returns=returns.mean().item())

        returns = returns.view(-1)  # (N * T)
        # normalize returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        old_log_probs = episode.log_probs.view(-1)
        states = episode.states.reshape(n_envs * timesteps, -1)
        actions = episode.actions.reshape(n_envs * timesteps, -1)

        for t in range(0, n_envs * timesteps, batch_size):
            batch_returns = returns[t : t + batch_size]
            batch_states = states[t : t + batch_size]
            batch_actions = actions[t : t + batch_size]
            batch_old_log_probs = old_log_probs[t : t + batch_size]

            log_probs, dist = policy.get_log_probs(batch_states, batch_actions)
            ratio = torch.exp(log_probs - batch_old_log_probs)

            t1 = batch_returns * ratio
            t2 = batch_returns * torch.clamp(ratio, 1 - eps, 1 + eps)
            grpo_loss = (
                -torch.min(t1, t2).mean()
                + beta * (log_probs - batch_old_log_probs).mean()
            )

            optimizer.zero_grad()
            grpo_loss.backward()
            optimizer.step()

        if ((i + 1) % log_every) == 0:
            avg_reward = evaluate_policy(eval_env, policy, batch_size=4)
            train_rewards.append(avg_reward)
            print(f"[Epoch {i + 1}] Reward={avg_reward:.2f} Loss={grpo_loss.item()}")

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
