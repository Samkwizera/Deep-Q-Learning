# Hyperparameter experiments

Member Name: Nkusi Dan

My focus is learning rate, gamma and batch size. Exploration is fixed at Samuel's best setting (epsilon 0.1 to 0.01 over 50k steps), so any change in behaviour comes from the parameter I changed. Experiment 1 is the baseline: it uses the same settings as Samuel's experiment 9, and every other run changes one or two values from it.

Fixed for every run: ALE/Pong-v5, 500k steps, epsilon 0.1 to 0.01 over 50k steps, replay buffer 50k, learning starts after 10k steps, one update every 4 steps, target network synced every 1k steps, seed 42.

For each run I looked at the mean reward of the last 20 training episodes, and I also played 5 games greedily (no epsilon) with the saved model, each starting from a different reset seed. Pong scores go from -21 (lose every point) to +21 (win every point). With sticky actions off, greedy Pong is almost deterministic, so games 2 to 5 usually came out identical. The greedy score is closer to 1 or 2 distinct games than to 5.

"Started improving" below means the first point where the mean of the last 20 training episodes rose above -19.5.

| Experiment | Learning rate | Gamma | Batch size | Epsilon start | Epsilon end | Epsilon decay | Observed behaviour |
|---|---|---|---|---|---|---|---|
| 1 | 0.0001 | 0.99 | 32 | 0.1 | 0.01 | 50000 | Baseline. Started improving around 140k steps and was still climbing at the end, finishing at -4.1 with 6 wins (best +14). Very long rallies at the end (about 4000 steps per episode). Greedy: 13.0 (+17 in four games, -3 in one). Much worse than Samuel's experiment 9, which used the same settings and seed (see the last section). |
| 2 | 0.0005 | 0.99 | 32 | 0.1 | 0.01 | 50000 | Never learned. Stayed at -21 for the whole run (best episode -19) with short games of about 780 steps. Greedy: -21 in every game. |
| 3 | 0.001 | 0.99 | 32 | 0.1 | 0.01 | 50000 | Same as experiment 2: never left -21 (best episode -19). Greedy: -21 in every game. The logs and models differ from experiment 2, but the outcome is the same. |
| 4 | 0.000025 | 0.99 | 32 | 0.1 | 0.01 | 50000 | Too slow. Only started improving around 300k steps, peaked at -16.3 around 370k and ended at -18.3 (best episode -8, no wins). Greedy: -15.0. |
| 5 | 0.0001 | 0.95 | 32 | 0.1 | 0.01 | 50000 | Started improving around 140k like the baseline, but improved slowly and ended at -16.1 (best episode -10, no wins). Long rallies (about 2900 steps), so it returned the ball but rarely scored. Greedy: -15.0. |
| 6 | 0.0001 | 0.9 | 32 | 0.1 | 0.01 | 50000 | Started around 160k and ended at -11.9 (best episode -6, no wins). Still improving at the end, and better than gamma 0.95 in training. Greedy: -14.4. |
| 7 | 0.0001 | 0.999 | 32 | 0.1 | 0.01 | 50000 | Close to the baseline in training. Started around 135k and ended at -2.5 with 10 wins (best +14), still improving, with the longest rallies of all runs (about 4800 steps). Greedy: -3.6, with one +2 game that hit the episode time limit (about 27k steps) and four -5 games. |
| 8 | 0.0001 | 0.99 | 64 | 0.1 | 0.01 | 50000 | Best greedy run. Started improving the earliest (around 75k), was winning on average from about 330k, and ended at +16.9 with 72 winning episodes (best +21). Greedy: 20.2 (21-0 once, 20-1 in the other four). |
| 9 | 0.0001 | 0.99 | 128 | 0.1 | 0.01 | 50000 | Best training run. Started around 100k, was winning on average from about 295k and ended at +18.7 with 100 winning episodes (best +21). Greedy: 14.8 (+14 once, +15 in the other four), lower than its training score. |
| 10 | 0.0005 | 0.99 | 64 | 0.1 | 0.01 | 50000 | The bigger batch did not rescue the higher learning rate. Stayed at -21 for the whole run, like experiment 2. Greedy: -21 in every game. |

Models, logs, plots and greedy scores (`eval.json`) for each run are in `Nkusi_Dan/expXX/`.

## Best and worst

Best: experiment 8 (batch size 64). It had the best greedy score (20.2, losing only 4 points in 5 games) and the second best training score (+16.9). Experiment 9 (batch size 128) was slightly better in training (+18.7) but worse greedy (14.8).

Worst: experiments 2, 3 and 10 (learning rate 0.0005 or 0.001). None of them ever left -21.

## What helped and what hurt

A learning rate of 0.0005 or higher broke training completely, with batch size 32 or 64. Lowering it to 0.000025 also hurt, because learning only started around 300k steps and there was not enough time left. 0.0001 was the only learning rate here that worked.

A bigger batch helped the most. Batch sizes 64 and 128 started learning earlier (75k to 100k steps instead of 140k), were winning on average by about 300k steps and finished far ahead of the batch-32 baseline. A bigger batch averages the gradient over more transitions, so each update is less noisy.

Lowering gamma hurt. With 0.95 and 0.9 the agent cares less about points that come many steps later, and in Pong a point is decided long after the paddle hits the ball. Both ended well below the baseline. Raising gamma to 0.999 was about the same as the baseline in training, but worse greedy.

Every setting was run once with one seed. My baseline (experiment 1) used exactly the same settings and seed as Samuel's experiment 9, but it finished at -4.1 in training against his +18.5, because GPU training is not bit-for-bit reproducible. So differences of even 10 or 20 points between two single runs can come from noise. The runs that never learned (learning rate 0.0005 and above) and the two big-batch runs both winning clearly are the results I trust most. Whether batch 64 or 128 is better is not something one run of each can settle.

## Final configuration

My best configuration is experiment 8: learning rate 0.0001, gamma 0.99, batch size 64, epsilon 0.1 to 0.01 over 50k steps. Its greedy score (20.2 over 5 games) is about the same as Samuel's experiment 9 (+20 over 3 games), which is the group's current `dqn_model.pth`.
