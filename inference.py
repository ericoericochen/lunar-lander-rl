import os
import json
import torch
import gymnasium as gym
from datetime import datetime
import argparse
from agent import RLAgent, create_mlp_from_env


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Run inference on a trained RL agent")
    parser.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="Directory containing the trained model and config",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save the recorded video",
    )
    args = parser.parse_args()

    config = load_config(os.path.join(args.run_dir, "config.json"))
    env = gym.make(config["env_id"])
    policy = create_mlp_from_env(env, config["n_hidden"])

    weights_path = os.path.join(args.run_dir, "policy.pth")
    if os.path.exists(weights_path):
        policy.load_state_dict(torch.load(weights_path))
        print(f"Loaded weights from {weights_path}")
    else:
        print(f"[ERROR] Weights file {weights_path} not found!")
        return

    agent = RLAgent(env=env, policy=policy)
    os.makedirs(args.output_dir, exist_ok=True)

    timestamp = str(int(datetime.now().timestamp()))
    agent.record_episode(args.output_dir, timestamp, greedy_sampling=True)


if __name__ == "__main__":
    main()
