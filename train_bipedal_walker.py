import gymnasium as gym
import torch
from coolname import generate_slug


from src import (
    create_continuous_policy,
    create_critic,
    train_ppo,
    ContinuousPolicy,
    Critic,
)

if __name__ == "__main__":
    env = gym.make("BipedalWalker-v3")
    # run_name = generate_slug(2)
    run_name = "mook"
    train_ppo(
        env_id="BipedalWalker-v3",
        env_kwargs={},
        actor=ContinuousPolicy(obs_dim=24, n_hidden=32, n_acts=4),
        critic=Critic(obs_dim=24, n_hidden=32),
        n_envs=32,
        timesteps=512,
        n_updates=2,
        actor_lr=5e-3,
        critic_lr=5e-3,
        n_epochs=10000,
        gamma=0.99,
        lmbda=0.999,
        eps=0.2,
        log_every=25,
        eval_every=100,
        save_dir=f"./training_runs/bipedal_walker/{run_name}",
    )
