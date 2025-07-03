import gymnasium as gym
import torch
from coolname import generate_slug


from src import (
    train_ppo,
    ContinuousPolicy,
    Critic,
)

if __name__ == "__main__":
    env = gym.make("BipedalWalker-v3")
    # run_name = generate_slug(2)
    # run_name = "kfc"
    run_name = "kibba"
    torch.manual_seed(243)
    train_ppo(
        env_id="BipedalWalker-v3",
        env_kwargs={"hardcore": True},
        # env_kwargs={},
        actor=ContinuousPolicy(obs_dim=24, n_hidden=64, n_acts=4),
        critic=Critic(obs_dim=24, n_hidden=32),
        n_envs=16,
        timesteps=1024,
        batch_size=4096,
        entropy_coef=0.001,
        actor_lr=3e-3,
        critic_lr=3e-3,
        n_epochs=100000,
        gamma=0.99,
        lmbda=0.99,
        eps=0.1,
        log_every=25,
        eval_every=100,
        save_dir=f"./training_runs/bipedal_walker/{run_name}",
        seed=243,
    )
