import random
import numpy as np
import torch
import gymnasium as gym
from agent import RLAgent, create_mlp_from_env

NUM_EPOCHS = 10000
GAMMA = 0.99
LR = 1e-2
LOG_EVERY = 50
N_HIDDEN = 32
BATCH_SIZE = 4
SEED = 123

# seed everything
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# create env and seed
env = gym.make("LunarLander-v3")
env.action_space.seed(SEED)
env.observation_space.seed(SEED)


if __name__ == "__main__":
    save_dir = "./runs/13"
    policy_mlp = create_mlp_from_env(env, N_HIDDEN)
    lunar_lander_agent = RLAgent(env=env, policy=policy_mlp)
    lunar_lander_agent.train(
        num_epochs=NUM_EPOCHS,
        lr=LR,
        batch_size=BATCH_SIZE,
        gamma=GAMMA,
        log_every=LOG_EVERY,
        save_dir=save_dir,
        seed=SEED,
    )
