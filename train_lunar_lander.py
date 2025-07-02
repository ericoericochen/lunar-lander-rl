import gymnasium as gym
import torch
from coolname import generate_slug


from src import (
    create_discrete_policy,
    create_continuous_policy,
    train_reinforce,
    train_a2c,
    train_ppo,
    create_critic,
)


if __name__ == "__main__":
    # env = gym.make("LunarLander-v3", continuous=True)
    env = gym.make("LunarLander-v3")
    # run_name = generate_slug(2)
    run_name = "kawwww"
    train_reinforce(
        env=env,
        policy=create_discrete_policy(env, n_hidden=16),
        lr=3e-4,
        # lr=1e-2,
        gamma=0.99,
        n_epochs=2000,
        batch_size=2048,
        save_dir=f"./training_runs/lunar_lander/{run_name}",
        log_every=25,
        seed=42,
    )

    # train_ppo(
    #     env=env,
    #     actor=create_continuous_policy(env, n_hidden=16),
    #     critic=create_critic(env, n_hidden=16),
    #     actor_lr=1e-3,
    #     critic_lr=1e-3,
    #     n_epochs=2000,
    #     batch_size=512,
    #     save_dir=f"./training_runs/lunar_lander/{run_name}",
    #     gamma=0.99,
    #     log_every=25,
    #     eval_every=100,
    #     seed=42,
    # )
    # train_a2c(
    #     env=env,
    #     actor=create_continuous_policy(env, n_hidden=32),
    #     critic=create_critic(env, n_hidden=32),
    #     actor_lr=3e-4,
    #     critic_lr=3e-4,
    #     n_epochs=10000,
    #     batch_size=512,
    #     save_dir=f"./training_runs/lunar_lander/{run_name}",
    #     gamma=0.99,
    #     lmbda=0.95,
    #     log_every=25,
    #     eval_every=100,
    #     seed=42,
    # )
