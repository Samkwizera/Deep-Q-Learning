# Hyperparameter experiments

Member Name: Ihimbazwe Samuel

My focus is the exploration settings (epsilon start, end and decay). I kept learning rate, gamma and batch size at the baseline values so any change in behaviour comes from exploration.

Fixed for every run: ALE/Pong-v5, 500k steps, replay buffer 50k, learning starts after 10k steps, one update every 4 steps, target network synced every 1k steps, seed 42. Epsilon decay is the number of steps epsilon takes to go linearly from start to end.

Experiment 1 is the shared baseline, the same run as Orrie's experiment 1.

| Experiment | Learning rate | Gamma | Batch size | Epsilon start | Epsilon end | Epsilon decay | Observed behaviour |
|---|---|---|---|---|---|---|---|
| 1 | 0.0001 | 0.99 | 32 | 1.0 | 0.05 | 100000 | Stuck around -20.5 for the first 200k steps, then started returning the ball. Mean of the last 20 episodes got to -13.6 by 500k (best episode -7) and episodes got longer (about 900 to 2000 steps), so it was losing points more slowly. Still improving when training stopped. Greedy play with play.py: -9. |
| 2 | 0.0001 | 0.99 | 32 | 1.0 | 0.01 | 100000 | |
| 3 | 0.0001 | 0.99 | 32 | 1.0 | 0.10 | 100000 | |
| 4 | 0.0001 | 0.99 | 32 | 1.0 | 0.20 | 100000 | |
| 5 | 0.0001 | 0.99 | 32 | 1.0 | 0.05 | 25000 | |
| 6 | 0.0001 | 0.99 | 32 | 1.0 | 0.05 | 250000 | |
| 7 | 0.0001 | 0.99 | 32 | 1.0 | 0.05 | 400000 | |
| 8 | 0.0001 | 0.99 | 32 | 0.5 | 0.05 | 100000 | |
| 9 | 0.0001 | 0.99 | 32 | 0.1 | 0.01 | 50000 | |
| 10 | 0.0001 | 0.99 | 32 | | | | |

Models, logs and plots for each run are in `Ihimbazwe_Samuel/expXX/`.

## Best and worst

## What helped and what hurt

## Final configuration
