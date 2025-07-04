import os
import torch
import gymnasium as gym
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

from src.utils import (
    record_episode,
    save_json,
    evaluate_policy,
    plot_training_rewards,
    create_n_envs,
)
from src.policy import Policy
from src.critic import Critic
from src.rollout import rollout_episode, get_gae_advantages


def train_ppo(
    env_id: str,
    env_kwargs: dict,
    actor: Policy,
    critic: Critic,
    actor_lr: float,
    critic_lr: float,
    n_epochs: int,
    save_dir: str,
    log_every: int,
    eval_every: int,
    batch_size: int,
    n_envs: int = 1,
    n_critic_updates: int = 1,
    timesteps: int = 512,
    gamma: float = 0.99,
    lmbda: float = 0.95,
    entropy_coef: float = 0.01,
    seed: int = None,
    eps: float = 0.2,
):
    env = create_n_envs(env_id, env_kwargs, n_envs)
    obs, _ = env.reset(seed=seed)
    eval_env = gym.make(env_id, **env_kwargs)

    os.makedirs(save_dir, exist_ok=True)
    print(f"[INFO] PPO: env={env_id} actor={actor.config}")
    print(f"[INFO] Saving to {save_dir}")
    print(
        f"[INFO] Training with gamma={gamma}, actor_lr={actor_lr}, critic_lr={critic_lr}, eps={eps}, timesteps={timesteps}, n_epochs={n_epochs}"
    )
    save_json(
        {
            "type": "ppo",
            "gamma": gamma,
            "lmbda": lmbda,
            "eps": eps,
            "entropy_coef": entropy_coef,
            "actor_lr": actor_lr,
            "critic_lr": critic_lr,
            "timesteps": timesteps,
            "n_epochs": n_epochs,
            "env_id": env_id,
            "actor": actor.config,
        },
        os.path.join(save_dir, "config.json"),
    )

    record_episode(
        env_id=env_id,
        env_kwargs=env_kwargs,
        policy=actor,
        save_dir=os.path.join(save_dir, "videos"),
        prefix="no_train",
    )

    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=actor_lr)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=critic_lr)
    train_rewards = []

    pbar = tqdm(range(n_epochs))
    for i in pbar:
        episode, obs = rollout_episode(
            env,
            obs=obs,
            policy=actor,
            timesteps=timesteps,
        )

        advantages, values = get_gae_advantages(
            episode, critic=critic, gamma=gamma, lmbda=lmbda
        )
        v_targets = advantages + values

        pbar.set_postfix(advantages=advantages.mean().item())

        # normalize advantages
        # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        advantages = advantages.view(-1)  # (N * T)

        # train actor
        old_log_probs = episode.log_probs.view(-1)
        states = episode.states.reshape(n_envs * timesteps, -1)
        actions = episode.actions.reshape(n_envs * timesteps, -1)

        for t in range(0, n_envs * timesteps, batch_size):
            A = advantages[t : t + batch_size]
            A = (A - A.mean()) / (A.std() + 1e-8)

            log_probs, dist = actor.get_log_probs(
                states[t : t + batch_size], actions[t : t + batch_size]
            )
            ratio = torch.exp(log_probs - old_log_probs[t : t + batch_size])

            t1 = A * ratio
            t2 = A * torch.clamp(ratio, 1 - eps, 1 + eps)
            ppo_loss = -torch.min(t1, t2).mean() - entropy_coef * dist.entropy().mean()

            actor_optimizer.zero_grad()
            ppo_loss.backward()
            actor_optimizer.step()

        # train critic
        for _ in range(n_critic_updates):
            v_preds = critic(episode.states)
            v_loss = F.mse_loss(v_targets, v_preds)
            # v_loss = F.smooth_l1_loss(v_targets, v_preds)

            critic_optimizer.zero_grad()
            v_loss.backward()
            critic_optimizer.step()

        if ((i + 1) % log_every) == 0:
            avg_reward = evaluate_policy(eval_env, actor, batch_size=4)
            train_rewards.append(avg_reward)
            print(
                f"[Epoch {i + 1}] Reward={avg_reward:.2f} Critic Loss={v_loss.item():.2f} Actor Loss={ppo_loss.item():.2f}"
            )

        if ((i + 1) % eval_every) == 0:
            record_episode(
                env_id=env_id,
                env_kwargs=env_kwargs,
                policy=actor,
                save_dir=os.path.join(save_dir, "videos"),
                prefix=f"eval-{i}",
            )

    plot_training_rewards(train_rewards, log_every, n_epochs, save_dir, eval_env)
    for i in range(4):
        record_episode(
            env_id=env_id,
            env_kwargs=env_kwargs,
            policy=actor,
            save_dir=os.path.join(save_dir, "videos"),
            prefix=f"final-{i}",
        )

    torch.save(actor.state_dict(), os.path.join(save_dir, "actor.pth"))
    torch.save(critic.state_dict(), os.path.join(save_dir, "critic.pth"))
