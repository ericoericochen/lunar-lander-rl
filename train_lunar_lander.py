import gymnasium as gym
import torch
from coolname import generate_slug


from src import (
    create_discrete_policy,
    train_reinforce,
    train_a2c,
    train_ppo,
    DiscretePolicy,
    ContinuousPolicy,
    Critic,
)


if __name__ == "__main__":
    torch.manual_seed(42)
    run_name = generate_slug(2)
    # run_name = "reinforce"
    # train_reinforce(
    #     env_id="LunarLander-v3",
    #     env_kwargs={},
    #     policy=DiscretePolicy(obs_dim=8, n_hidden=16, n_acts=4),
    #     lr=1e-2,
    #     n_envs=16,
    #     gamma=0.99,
    #     n_epochs=2000,
    #     timesteps=512,
    #     save_dir=f"./training_runs/lunar_lander/{run_name}",
    #     log_every=25,
    #     eval_every=100,
    #     seed=42,
    # )

    # train_a2c(
    #     env_id="LunarLander-v3",
    #     env_kwargs={},
    #     actor=DiscretePolicy(obs_dim=8, n_hidden=16, n_acts=4),
    #     critic=Critic(obs_dim=8, n_hidden=16),
    #     actor_lr=1e-2,
    #     critic_lr=1e-2,
    #     n_epochs=2000,
    #     n_envs=32,
    #     timesteps=256,
    #     save_dir=f"./training_runs/lunar_lander/{run_name}",
    #     gamma=0.99,
    #     lmbda=0.999,
    #     log_every=25,
    #     eval_every=100,
    #     n_critic_updates=1,
    #     seed=42,
    # )

    train_ppo(
        env_id="LunarLander-v3",
        env_kwargs={"continuous": True},
        actor=ContinuousPolicy(obs_dim=8, n_hidden=16, n_acts=2),
        # actor=DiscretePolicy(obs_dim=8, n_hidden=16, n_acts=4),
        critic=Critic(obs_dim=8, n_hidden=16),
        actor_lr=1e-3,
        critic_lr=1e-3,
        n_epochs=2000,
        n_envs=4,
        timesteps=256,
        save_dir=f"./training_runs/lunar_lander_continuous/{run_name}",
        gamma=0.99,
        lmbda=0.999,
        log_every=25,
        eval_every=100,
        n_updates=4,
        seed=42,
    )
