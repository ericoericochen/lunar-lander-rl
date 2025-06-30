import gymnasium as gym
import torch
from coolname import generate_slug


from src import create_continuous_policy, create_critic, train_a2c

if __name__ == "__main__":
    env = gym.make("BipedalWalker-v3")
    # run_name = generate_slug(2)
    run_name = "mook"
    train_a2c(
        env=env,
        actor=create_continuous_policy(env, n_hidden=32),
        critic=create_critic(env, n_hidden=16),
        actor_lr=8e-3,
        critic_lr=3e-3,
        n_epochs=50000,
        gamma=0.99,
        log_every=25,
        eval_every=100,
        batch_size=1024,
        save_dir=f"./training_runs/bipedal_walker/{run_name}",
    )
