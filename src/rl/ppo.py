import os
import torch
import gymnasium as gym
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm

from ..utils import record_episode, save_json, evaluate_policy, plot_training_rewards
from ..policy import Policy, action_pt_to_env
from ..critic import Critic


def get_episode_batch(env: gym.Env, actor: Policy, timesteps: int, gamma: float):
    states, next_states, actions, rewards, log_probs, dones, returns = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )

    obs, _ = env.reset()
    for i in range(timesteps):
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        states.append(obs_tensor)
        action, log_prob = actor(obs_tensor)

        raise RuntimeError

        obs, reward, done, _, __ = env.step(action_pt_to_env(action, env))
        next_states.append(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))

        actions.append(action)
        log_probs.append(log_prob)
        rewards.append(float(reward))
        dones.append(int(done))

        if done:
            obs, _ = env.reset()

    # calculate g_t = r_t + 1 + gamma * r_t+1 for each timestep
    R = 0
    for reward, done in zip(rewards[::-1], dones[::-1]):
        R = reward + gamma * R * (1 - done)
        returns.insert(0, R)

    return {
        "states": torch.cat(states, dim=0),
        "next_states": torch.cat(next_states, dim=0),
        "actions": torch.cat(actions, dim=0),
        "rewards": torch.tensor(rewards),
        "dones": torch.tensor(dones, dtype=torch.bool),
        "log_probs": torch.cat(log_probs, dim=0),
        "returns": torch.tensor(returns),
    }


def calculate_advantages(
    critic: Critic,
    states: torch.Tensor,
    next_states: torch.Tensor,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    lmbda: float,
    gamma: float,
):
    v_next, v_curr = critic(next_states), critic(states)
    deltas = rewards + gamma * v_next * (1 - dones.float()) - v_curr  # TD error

    A = torch.tensor(0.0)
    T = deltas.shape[0]
    advantages = torch.zeros_like(deltas)

    # calculate GAE for each timestep: A_t = delta_t + gamma * lmbda * A_t+1
    for t in range(T - 1, -1, -1):
        A = deltas[t] + gamma * lmbda * A * (1 - dones[t].float())
        advantages[t] = A

    return advantages


def train_ppo(
    env: gym.Env,
    actor: Policy,
    critic: Critic,
    actor_lr: float,
    critic_lr: float,
    n_epochs: int,
    save_dir: str,
    log_every: int,
    eval_every: int,
    num_envs: int = 1,
    rollout_length: int = 512,
    gamma: float = 0.99,
    seed: int = None,
    lmbda: float = 0.95,
):
    if seed:
        torch.manual_seed(seed)
    os.makedirs(save_dir, exist_ok=True)
    print(f"[INFO] A2C: env={env.spec.id} actor={actor.config}")
    print(f"[INFO] Saving to {save_dir}")
    print(
        f"[INFO] Training with gamma={gamma}, actor_lr={actor_lr}, critic_lr={critic_lr}, batch_size={batch_size}, n_epochs={n_epochs}"
    )
    save_json(
        {
            "gamma": gamma,
            "actor_lr": actor_lr,
            "critic_lr": critic_lr,
            "batch_size": batch_size,
            "n_epochs": n_epochs,
            "env_id": env.spec.id,
            "actor": actor.config,
        },
        os.path.join(save_dir, "config.json"),
    )

    record_episode(
        env_id=env.spec.id,
        policy=actor,
        save_dir=os.path.join(save_dir, "videos"),
        prefix="no_train",
    )

    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=actor_lr)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=critic_lr)
    train_rewards = []

    for i in tqdm(range(n_epochs)):
        episode_batch = get_episode_batch(
            env=env, actor=actor, batch_size=batch_size, gamma=gamma
        )

        states, next_states, returns, rewards, dones, log_probs = (
            episode_batch["states"],
            episode_batch["next_states"],
            episode_batch["returns"],
            episode_batch["rewards"],
            episode_batch["dones"],
            episode_batch["log_probs"],
        )

        # update critic, loss is squared loss b/w empirical Gt and V(s_t)
        v_states = critic(states)
        critic_loss = F.smooth_l1_loss(returns, v_states)

        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()

        # calculate advantage
        with torch.no_grad():
            advantages = calculate_advantages(
                critic=critic,
                states=states,
                next_states=next_states,
                rewards=rewards,
                dones=dones,
                lmbda=lmbda,
                gamma=gamma,
            )
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # update actor
        actor_loss = -(advantages * log_probs).mean()
        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()

        if ((i + 1) % log_every) == 0:
            avg_reward = evaluate_policy(env, actor, batch_size=4)
            train_rewards.append(avg_reward)
            print(
                f"[Epoch {i + 1}] Reward={avg_reward:.2f} Critic Loss={critic_loss:.2f} Actor Loss={actor_loss:.2f}"
            )

        if ((i + 1) % eval_every) == 0:
            record_episode(
                env_id=env.spec.id,
                policy=actor,
                save_dir=os.path.join(save_dir, "videos"),
                prefix=f"eval-{i}",
            )

    plot_training_rewards(train_rewards, log_every, n_epochs, save_dir, env)
    for i in range(4):
        record_episode(
            env_id=env.spec.id,
            policy=actor,
            save_dir=os.path.join(save_dir, "videos"),
            prefix=f"final-{i}",
        )

    torch.save(actor.state_dict(), os.path.join(save_dir, "actor.pth"))
    torch.save(critic.state_dict(), os.path.join(save_dir, "critic.pth"))
