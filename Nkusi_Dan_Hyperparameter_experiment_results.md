# Hyperparameter experiments

Member Name: Nkusi Dan

My focus is the exploration settings (epsilon start, end and decay). I kept learning rate, gamma and batch size at the baseline values so any change in behaviour comes from exploration.

Fixed for every run: ALE/Pong-v5, 500k steps, replay buffer 50k, learning starts after 10k steps, one update every 4 steps, target network synced every 1k steps, seed 42. Epsilon decay is the number of steps epsilon takes to go linearly from start to end. Each run took about 20 minutes on a Kaggle T4.

For each run I looked at the mean reward of the last 20 training episodes, and I also played 5 games greedily (no epsilon) with the saved model, each starting from a different reset seed. Pong scores go from -21 (lose every point) to +21 (win every point). With sticky actions off, greedy Pong is almost deterministic, so games 2 to 5 usually came out identical. The greedy score is closer to 1 or 2 distinct games than to 5.

| Experiment | Learning rate | Gamma | Batch size | Epsilon start | Epsilon end | Epsilon decay | Observed behaviour |
|---|---|---|---|---|---|---|---|
| 1 | 0.0001 | 0.99 | 32 | 1.0 | 0.05 | 250000 | Baseline. Stuck around -20.5 until about 230k steps, then kept improving to -12.6 at the end (best episode -3). Rallies got long (about 3000 steps per episode), so it was returning the ball but still losing points. Still improving when training stopped. Greedy: -8.2 (-5 once, -9 in the other four). |
| 2 | 0.0001 | 0.99 | 32 | 1.0 | 0.05 | 50000 | Worst run. Epsilon reached 0.05 long before the network had learned anything, and it stayed near -21 for almost the whole run. Ended at -19.1 (best episode -15). Greedy: -20.2. |
| 3 | 0.0001 | 0.99 | 32 | 1.0 | 0.05 | 450000 | Too much random play for too long. Only started improving around 265k and ended at -18.3 (best episode -13). Greedy: -15.8. |
| 4 | 0.0001 | 0.99 | 32 | 1.0 | 0.01 | 250000 | Best run. Started improving around 205k and climbed fastest at the end, finishing at -5.7. Won 4 training episodes, including one 21-1 (+20). Still improving at 500k. Greedy: -1.2, but that is one +18 game and four -6 games. |
| 5 | 0.0001 | 0.99 | 32 | 1.0 | 0.20 | 250000 | 20% random actions to the end held it back. Peaked at -13.8 around 470k, then dropped to -15.6. Short episodes (about 1650 steps), so it lost points quickly. Greedy: -16.8. |
| 6 | 0.0001 | 0.99 | 32 | 0.5 | 0.05 | 250000 | Started learning earlier than the baseline (around 145k) and ended better, at -10.1 (one win, +3). Greedy was inconsistent: -8 in three games and -18 in two. |
| 7 | 0.0001 | 0.99 | 32 | 0.1 | 0.10 | constant | Learned the earliest of all (around 125k) and ended at -10.2 in training, with the longest rallies (about 3300 steps). Greedy it fell apart: -18.6. It seems to have needed the 10% random actions to get out of loops it gets stuck in when playing greedily. |
| 8 | 0.0001 | 0.99 | 32 | 0.2 | 0.01 | 125000 | Started early (around 180k) and peaked at -9.8 around 440k with 3 wins (best +4), then dropped a bit to -11.0. Greedy: -9.6 (+4 once, -13 in the rest). |
| 9 | 0.0001 | 0.99 | 32 | 1.0 | 0.01 | 50000 | Second best in training. Same fast decay as experiment 2 but ending at 0.01 instead of 0.05, and it learned instead of stalling. Started around 150k and ended at -6.4 with 3 wins (best +5). Greedy: -10.8. |
| 10 | 0.0001 | 0.99 | 32 | 1.0 | 0.10 | 375000 | Slow decay plus a high end value, so it explored the most, and it was the second worst. Only got going around 275k and ended at -16.9 (best episode -10). Greedy: -19.4. |

Models, logs, plots and greedy scores (`eval.json`) for each run are in `Nkusi_Dan/expXX/`.

## Best and worst

Best: experiment 4 (start 1.0, end 0.01, decay 250k). It had the best training score (-5.7), the best greedy score and the most wins, and it was still improving when training stopped.

Worst: experiment 2 (start 1.0, end 0.05, decay 50k). It never really left -21 and finished at -19.1 in training and -20.2 greedy.

## What helped and what hurt

A low final epsilon helped most. Both runs that ended at 0.01 after starting at 1.0 (experiments 4 and 9) were the top two in training. Once the agent can return the ball, every random action risks losing a point, so cutting those late helps.

Exploring a lot late in training hurt: a high end value (experiments 5 and 10) or a very long decay (experiments 3 and 10). Those runs spent the second half making random moves instead of improving on what they had learned.

Starting with less exploration (experiments 6, 7 and 8) made learning start earlier, around 125k to 180k steps instead of about 230k. The end results were only moderately better than the baseline, though, and constant 0.1 (experiment 7) did badly when played greedily.

The decay length on its own didn't give a clear pattern. A 50k decay was the worst run with end 0.05 (experiment 2) and the second best with end 0.01 (experiment 9). Every setting was run once with one seed, and DQN results change a lot between runs, so differences of a few points between experiments are probably not reliable.

## Final configuration

My best configuration is experiment 4: learning rate 0.0001, gamma 0.99, batch size 32, epsilon 1.0 to 0.01 over 250k steps. Samuel's experiment 9 (epsilon 0.1 to 0.01 over 50k steps) did much better (+20 greedy), so it is the group's current `dqn_model.pth`.
