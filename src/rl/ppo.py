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
    n_envs: int = 1,
    n_updates: int = 2,
    timesteps: int = 512,
    gamma: float = 0.99,
    lmbda: float = 0.95,
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
            "n_updates": n_updates,
            "actor_lr": actor_lr,
            "critic_lr": critic_lr,
            "timesteps": timesteps,
            "n_epochs": n_epochs,
            "env_id": env_id,
            "actor": actor.config,
        },
        os.path.join(save_dir, "config.json"),
    )

    # record_episode(
    #     env_id=env.spec.id,
    #     env_kwargs=env_kwargs,
    #     policy=actor,
    #     save_dir=os.path.join(save_dir, "videos"),
    #     prefix="no_train",
    # )

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

        # train critic
        v_targets = advantages + values
        v_preds = critic(episode.states)
        v_loss = F.smooth_l1_loss(v_targets, v_preds)

        critic_optimizer.zero_grad()
        v_loss.backward()
        critic_optimizer.step()

        pbar.set_postfix(advantages=advantages.mean().item())

        # train actor
        for _ in range(n_updates):
            old_log_probs = episode.log_probs
            log_probs = actor.get_log_probs(episode.states, episode.actions)
            ratio = torch.exp(log_probs - old_log_probs)

            # normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            ppo_loss = -(
                torch.min(
                    advantages * ratio,
                    advantages * torch.clamp(ratio, 1 - eps, 1 + eps),
                )
            ).mean()

            actor_optimizer.zero_grad()
            ppo_loss.backward()
            actor_optimizer.step()

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
