import argparse
import csv
import json
import os
import random
import time

import ale_py
import gymnasium as gym
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

gym.register_envs(ale_py)

ENV_ID = "ALE/Pong-v5"


def make_env(render_mode=None):
    # frameskip=1 because AtariPreprocessing already skips 4 frames.
    # sticky actions are off so the chosen action always gets executed.
    env = gym.make(ENV_ID, frameskip=1, repeat_action_probability=0.0, render_mode=render_mode)
    env = gym.wrappers.AtariPreprocessing(env, frame_skip=4, screen_size=84, grayscale_obs=True, noop_max=30)
    # one frame can't show which way the ball is moving, 4 stacked frames can
    env = gym.wrappers.FrameStackObservation(env, stack_size=4)
    return env


class QNetwork(nn.Module):
    def __init__(self, n_actions):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        # 84x84 input ends up as 64 feature maps of 7x7 after the convs
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )

    def forward(self, x):
        x = x.float() / 255.0
        return self.fc(self.conv(x))


class ReplayBuffer:
    def __init__(self, capacity, obs_shape):
        self.capacity = capacity
        # uint8 instead of float32 so the buffer uses 4x less memory
        self.states = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.next_states = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.index = 0
        self.size = 0

    def add(self, state, action, reward, next_state, done):
        self.states[self.index] = state
        self.actions[self.index] = action
        self.rewards[self.index] = reward
        self.next_states[self.index] = next_state
        self.dones[self.index] = done
        # once full, wrap around and overwrite the oldest experience
        self.index = (self.index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, device):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.as_tensor(self.states[idx], device=device),
            torch.as_tensor(self.actions[idx], device=device),
            torch.as_tensor(self.rewards[idx], device=device),
            torch.as_tensor(self.next_states[idx], device=device),
            torch.as_tensor(self.dones[idx], device=device),
        )


def get_epsilon(step, eps_start, eps_end, eps_decay):
    # linear decay over eps_decay steps, then stays at eps_end
    fraction = min(step / eps_decay, 1.0)
    return eps_start + fraction * (eps_end - eps_start)


def select_action(state, q_net, epsilon, n_actions, device):
    if random.random() < epsilon:
        return random.randrange(n_actions)
    with torch.no_grad():
        state_t = torch.as_tensor(np.asarray(state), device=device).unsqueeze(0)
        q_values = q_net(state_t)
    return int(q_values.argmax(dim=1).item())


def train_step(q_net, target_net, optimizer, batch, gamma):
    states, actions, rewards, next_states, dones = batch

    q_values = q_net(states)
    q_taken = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

    # bellman target uses the target network so the target doesn't shift every update.
    # if the episode ended there is no future reward, hence (1 - dones)
    with torch.no_grad():
        max_next_q = target_net(next_states).max(dim=1).values
        targets = rewards + gamma * max_next_q * (1.0 - dones)

    # huber loss instead of mse so a few big errors don't blow up the gradients
    loss = F.smooth_l1_loss(q_taken, targets)

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(q_net.parameters(), max_norm=10.0)
    optimizer.step()
    return loss.item()


def save_log(episodes, path):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=episodes[0].keys())
        writer.writeheader()
        writer.writerows(episodes)


def save_plot(episodes, path, title):
    rewards = [e["reward"] for e in episodes]
    lengths = [e["length"] for e in episodes]
    steps = [e["total_steps"] for e in episodes]
    window = 20
    moving_avg = [np.mean(rewards[max(0, i - window + 1): i + 1]) for i in range(len(rewards))]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    ax1.plot(steps, rewards, alpha=0.3, label="episode reward")
    ax1.plot(steps, moving_avg, label=f"mean of last {window} episodes")
    ax1.set_ylabel("reward")
    ax1.legend()
    ax1.set_title(title)
    ax2.plot(steps, lengths, color="tab:green")
    ax2.set_ylabel("episode length (steps)")
    ax2.set_xlabel("total steps")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--eps-start", type=float, default=1.0)
    p.add_argument("--eps-end", type=float, default=0.05)
    p.add_argument("--eps-decay", type=int, default=100_000, help="steps to go from eps-start to eps-end")
    p.add_argument("--total-steps", type=int, default=200_000)
    p.add_argument("--buffer-size", type=int, default=50_000)
    p.add_argument("--learning-starts", type=int, default=10_000)
    p.add_argument("--train-freq", type=int, default=4)
    p.add_argument("--target-update", type=int, default=1_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=str, default="runs/default")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"training on {device}")

    env = make_env()
    n_actions = env.action_space.n

    q_net = QNetwork(n_actions).to(device)
    target_net = QNetwork(n_actions).to(device)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    optimizer = torch.optim.Adam(q_net.parameters(), lr=args.lr)
    buffer = ReplayBuffer(args.buffer_size, env.observation_space.shape)

    episodes = []
    state, _ = env.reset(seed=args.seed)
    episode_reward, episode_length = 0.0, 0
    start_time = time.time()

    for step in range(1, args.total_steps + 1):
        epsilon = get_epsilon(step, args.eps_start, args.eps_end, args.eps_decay)
        action = select_action(state, q_net, epsilon, n_actions, device)
        next_state, reward, terminated, truncated, _ = env.step(action)

        # only store terminated as done, a time limit cut doesn't mean the future value is zero
        buffer.add(state, action, reward, next_state, float(terminated))
        state = next_state
        episode_reward += reward
        episode_length += 1

        # wait until the buffer has some random experience before learning from it
        if step > args.learning_starts and step % args.train_freq == 0:
            batch = buffer.sample(args.batch_size, device)
            train_step(q_net, target_net, optimizer, batch, args.gamma)

        if step % args.target_update == 0:
            target_net.load_state_dict(q_net.state_dict())

        if terminated or truncated:
            episodes.append({
                "episode": len(episodes) + 1,
                "total_steps": step,
                "length": episode_length,
                "reward": episode_reward,
                "epsilon": round(epsilon, 4),
            })
            recent = np.mean([e["reward"] for e in episodes[-20:]])
            speed = step / (time.time() - start_time)
            print(f"ep {len(episodes):4d} | step {step:7d} | reward {episode_reward:6.1f} | "
                  f"mean20 {recent:6.2f} | len {episode_length:5d} | eps {epsilon:.3f} | {speed:.0f} steps/s")
            state, _ = env.reset()
            episode_reward, episode_length = 0.0, 0

    env.close()

    torch.save(q_net.state_dict(), os.path.join(args.out_dir, "dqn_model.pth"))
    if episodes:
        save_log(episodes, os.path.join(args.out_dir, "log.csv"))
        save_plot(episodes, os.path.join(args.out_dir, "plot.png"), title=args.out_dir)
    print(f"saved to {args.out_dir}")


if __name__ == "__main__":
    main()
