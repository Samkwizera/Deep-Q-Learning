# Deep Q-Learning on Atari Pong (Group 12)

A Deep Q-Network (DQN) trained from scratch on `ALE/Pong-v5` with PyTorch and Gymnasium. Each member ran 10 hyperparameter experiments, and the best model is in `dqn_model.pth`.

Members: Ihimbazwe Samuel, Nkusi Dan

## Repository layout

| Path | What it is |
|---|---|
| `train.py` | Trains a DQN agent and saves the model, a per-episode log and a reward plot |
| `play.py` | Loads a trained model and plays Pong greedily (on screen, or recorded to mp4) |
| `evaluate.py` | Plays a few greedy games without a window and saves the scores to `eval.json` |
| `dqn_model.pth` | The group's best model (currently Samuel's experiment 9) |
| `notebooks/kaggle_training.ipynb` | Samuel's notebook for running his experiments on a Kaggle GPU |
| `notebooks/kaggle_experiments.ipynb` | Dan's notebook. It trains his 10 experiments on a Kaggle GPU and scores each model with `evaluate.py` |
| `Ihimbazwe_Samuel/expXX/`, `Nkusi_Dan/expXX/` | One folder per run: `config.json`, `dqn_model.pth`, `log.csv`, `plot.png` (Dan's runs also have `eval.json` with the greedy scores) |
| `*_Hyperparameter_experiment_results.md` | Each member's full results table and analysis |

## Setup

The project uses [uv](https://docs.astral.sh/uv/) and Python 3.11.

```bash
uv sync
```

This installs PyTorch, Gymnasium with the Atari ROMs (`ale-py`), NumPy, OpenCV, Matplotlib and MoviePy, at the versions pinned in `uv.lock`.

## Training

```bash
uv run python train.py --total-steps 500000 --out-dir runs/my_run
```

All hyperparameters are command-line flags:

| Flag | Default | Meaning |
|---|---|---|
| `--lr` | 0.0001 | Adam learning rate |
| `--gamma` | 0.99 | Discount factor |
| `--batch-size` | 32 | Transitions per gradient update |
| `--eps-start` | 1.0 | Epsilon at the start of training |
| `--eps-end` | 0.05 | Epsilon after the decay |
| `--eps-decay` | 100000 | Steps over which epsilon falls linearly from start to end |
| `--total-steps` | 200000 | Environment steps to train for (all of our experiments used 500000) |
| `--buffer-size` | 50000 | Replay buffer capacity |
| `--learning-starts` | 10000 | Random steps collected before learning begins |
| `--train-freq` | 4 | One gradient update every N steps |
| `--target-update` | 1000 | Copy the online network into the target network every N steps |
| `--seed` | 42 | Random seed |
| `--out-dir` | `runs/default` | Where the model, log and plot are saved |

A 500k-step run takes about 20 minutes on a Kaggle T4 GPU. On a laptop CPU it runs at about 25 steps/s, which is why we trained on Kaggle (see the two notebooks).

## Playing

```bash
uv run python play.py                       # opens a window and plays 3 games with dqn_model.pth
uv run python play.py --model Nkusi_Dan/exp04/dqn_model.pth --episodes 5
uv run python play.py --record              # saves mp4s to videos/ instead of opening a window
```

`play.py` always picks the action with the highest Q-value (no epsilon). It imports the environment and the network from `train.py`, so the preprocessing is exactly the same as in training.

To score a model without opening a window (this is how Dan's `eval.json` files were made):

```bash
uv run python evaluate.py --model Nkusi_Dan/exp04/dqn_model.pth --episodes 5 --out eval.json
```

## How the agent works

**Environment and preprocessing:**
- `ALE/Pong-v5` with sticky actions turned off.
- `AtariPreprocessing`: frame skip 4, 84×84 grayscale, and up to 30 random no-ops at reset.
- The last 4 frames are stacked so the network can see which way the ball is moving.

**Network:** the Nature-DQN CNN.
- Three conv layers: 32 filters 8×8 stride 4, 64 filters 4×4 stride 2, 64 filters 3×3 stride 1.
- A 512-unit fully connected layer, then one output per action (6 actions).
- Pixels are scaled to [0, 1] inside the network.

**Learning:**
- Experience replay: a uint8 buffer of 50k transitions, sampled uniformly.
- A separate target network, synced every 1000 steps.
- Huber loss on the Bellman target, gradient-norm clipping at 10, and the Adam optimiser.
- Only real episode ends (`terminated`) zero out the future value. A time-limit cut (`truncated`) does not.

## Hyperparameter experiments

Both members tuned exploration: epsilon start, end and decay. Learning rate (0.0001), gamma (0.99) and batch size (32) stayed fixed, and so did everything else in the training table above. Every run was 500k steps with seed 42.

Below is a summary. The full tables, with the observed behaviour of each run and each member's analysis, are in [Ihimbazwe_Samuel_Hyperparameter_experiment_results.md](Ihimbazwe_Samuel_Hyperparameter_experiment_results.md) and [Nkusi_Dan_Hyperparameter_experiment_results.md](Nkusi_Dan_Hyperparameter_experiment_results.md).

"Train" is the mean reward of the last 20 training episodes. "Greedy" is the mean score with epsilon = 0: Samuel played 3 games, and Dan played 5 games from different reset seeds. Pong scores run from -21 to +21.

### Ihimbazwe Samuel

| Exp | ε start | ε end | ε decay | Train | Greedy |
|---|---|---|---|---|---|
| 1 | 1.0 | 0.05 | 100k | -13.6 | -11.3 |
| 2 | 1.0 | 0.01 | 100k | -18.6 | -13.3 |
| 3 | 1.0 | 0.10 | 100k | -16.2 | -17.3 |
| 4 | 1.0 | 0.20 | 100k | -13.2 | -8.0 |
| 5 | 1.0 | 0.05 | 25k | -13.7 | -15.0 |
| 6 | 1.0 | 0.05 | 250k | -16.4 | -13.0 |
| 7 | 1.0 | 0.05 | 400k | -13.8 | -5.7 |
| 8 | 0.5 | 0.05 | 100k | -16.4 | -18.3 |
| **9** | **0.1** | **0.01** | **50k** | **+18.5** | **+20.0** |
| 10 | | | | | |

### Nkusi Dan

| Exp | ε start | ε end | ε decay | Train | Greedy |
|---|---|---|---|---|---|
| 1 | 1.0 | 0.05 | 250k | -12.6 | -8.2 |
| 2 | 1.0 | 0.05 | 50k | -19.1 | -20.2 |
| 3 | 1.0 | 0.05 | 450k | -18.3 | -15.8 |
| **4** | **1.0** | **0.01** | **250k** | **-5.7** | **-1.2** |
| 5 | 1.0 | 0.20 | 250k | -15.6 | -16.8 |
| 6 | 0.5 | 0.05 | 250k | -10.1 | -12.0 |
| 7 | 0.1 | 0.10 | constant | -10.2 | -18.6 |
| 8 | 0.2 | 0.01 | 125k | -11.0 | -9.6 |
| 9 | 1.0 | 0.01 | 50k | -6.4 | -10.8 |
| 10 | 1.0 | 0.10 | 375k | -16.9 | -19.4 |

### What we found

- **A low starting epsilon was the biggest win.** Samuel's experiment 9 (epsilon 0.1 to 0.01) was the only run that learned to win consistently: +18.5 in training, +20 greedy. Every run that started at 1.0 was still losing at 500k steps.
- **Ending at 0.01 gave mixed results.** It produced the next two best runs by training reward, Dan's experiments 4 and 9. It also produced one of the worst, Samuel's experiment 2. Ending high (0.1 to 0.2) or decaying very slowly was usually below the baseline. Samuel's experiments 4 and 7 were exceptions.
- **Starting with less exploration makes learning start earlier.** Runs that started at epsilon 0.1 to 0.5 began improving around 120k to 220k steps. Runs that started at 1.0 began around 200k steps or later. In Pong the agent gets a reward at the end of every point, so it does not need many random actions to find rewards.
- **One seed is not enough to rank close runs.** Samuel's experiment 6 and Dan's experiment 1 used exactly the same settings and the same seed. They still finished at -16.4 and -12.6, because GPU training is not bit-for-bit reproducible. Differences of a few points between runs are within this noise. The gap between Samuel's experiment 9 and every other run is not.
- **The greedy scores cover fewer games than they look.** With sticky actions off, Pong is almost deterministic, so a greedy agent often plays the same game again. In Dan's evaluations, games 2 to 5 were usually identical.

### Final model

`dqn_model.pth` is Samuel's experiment 9: learning rate 0.0001, gamma 0.99, batch size 32, epsilon 0.1 to 0.01 over 50k steps, 500k steps in total. It scored +20 in each of its 3 greedy games.
