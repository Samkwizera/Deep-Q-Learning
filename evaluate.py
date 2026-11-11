import argparse
import json

import numpy as np
import torch

# same environment and network as training
from train import QNetwork, make_env


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="dqn_model.pth")
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--out", type=str, default=None, help="optional path to save the scores as json")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = make_env()
    q_net = QNetwork(env.action_space.n).to(device)
    q_net.load_state_dict(torch.load(args.model, map_location=device))
    q_net.eval()

    rewards, lengths = [], []
    for i in range(args.episodes):
        # a different reset seed per game, so the no-op starts differ
        state, _ = env.reset(seed=1000 + i)
        total, steps, done = 0.0, 0, False
        while not done:
            # epsilon = 0: always the action the network rates highest, like play.py
            with torch.no_grad():
                state_t = torch.as_tensor(np.asarray(state), device=device).unsqueeze(0)
                action = int(q_net(state_t).argmax(dim=1).item())
            state, reward, terminated, truncated, _ = env.step(action)
            total += reward
            steps += 1
            done = terminated or truncated
        rewards.append(total)
        lengths.append(steps)
        print(f"game {i + 1}: reward {total:.0f}, length {steps}")
    env.close()

    result = dict(eval_episodes=args.episodes, eval_mean=float(np.mean(rewards)), eval_std=float(np.std(rewards)),
                  eval_rewards=rewards, eval_lengths=lengths)
    print(f"mean {result['eval_mean']:.1f} ± {result['eval_std']:.1f}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
