import argparse

import gymnasium as gym
import numpy as np
import torch

# importing from train.py so the preprocessing and network are exactly the same as in training
from train import QNetwork, make_env


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="dqn_model.pth")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--record", action="store_true", help="save mp4s to videos/ instead of opening a window")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.record:
        env = make_env(render_mode="rgb_array")
        env = gym.wrappers.RecordVideo(env, video_folder="videos", episode_trigger=lambda ep: True)
    else:
        env = make_env(render_mode="human")
    n_actions = env.action_space.n

    q_net = QNetwork(n_actions).to(device)
    q_net.load_state_dict(torch.load(args.model, map_location=device))
    q_net.eval()

    for episode in range(1, args.episodes + 1):
        state, _ = env.reset()
        total_reward, steps, done = 0.0, 0, False

        while not done:
            # no epsilon here, always take the best action the network knows
            with torch.no_grad():
                state_t = torch.as_tensor(np.asarray(state), device=device).unsqueeze(0)
                action = int(q_net(state_t).argmax(dim=1).item())

            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated

        print(f"episode {episode}: reward {total_reward:.0f}, length {steps}")

    env.close()


if __name__ == "__main__":
    main()
