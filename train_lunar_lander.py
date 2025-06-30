import gymnasium as gym
import torch
from coolname import generate_slug


from src import create_discrete_policy, train_reinforce


if __name__ == "__main__":
    env = gym.make("LunarLander-v3")
    run_name = generate_slug(2)
    train_reinforce(
        env=env,
        policy=create_discrete_policy(env, n_hidden=16),
        lr=1e-2,
        gamma=0.99,
        n_epochs=2000,
        batch_size=2048,
        save_dir=f"./training_runs/lunar_lander/{run_name}",
        with_baseline=True,
        log_every=25,
        seed=42,
    )
