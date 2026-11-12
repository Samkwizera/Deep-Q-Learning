# Hyperparameter experiments

Member Name: Ihimbazwe Samuel

My focus is the exploration settings (epsilon start, end and decay). I kept learning rate, gamma and batch size at the baseline values so any change in behaviour comes from exploration.

Fixed for every run: ALE/Pong-v5, 500k steps, replay buffer 50k, learning starts after 10k steps, one update every 4 steps, target network synced every 1k steps, seed 42. Epsilon decay is the number of steps epsilon takes to go linearly from start to end.

Experiment 1 is the baseline with the default settings.

For each run I looked at the mean reward of the last 20 training episodes, and I also played 3 games greedily (no epsilon) with the saved model. Pong scores go from -21 (lose every point) to +21 (win every point).

| Experiment | Learning rate | Gamma | Batch size | Epsilon start | Epsilon end | Epsilon decay | Observed behaviour |
|---|---|---|---|---|---|---|---|
| 1 | 0.0001 | 0.99 | 32 | 1.0 | 0.05 | 100000 | Stuck around -20.5 for the first 200k steps, then started returning the ball. Last 20 episodes averaged -13.6 (best episode -7) and episodes got longer (about 900 to 2100 steps), so it was losing points more slowly. Still improving when training stopped. Greedy: -11.3. |
| 2 | 0.0001 | 0.99 | 32 | 1.0 | 0.01 | 100000 | Slowest of all. Stayed at about -20.8 until around 330k steps and only reached -18.7 at the end (best episode -12). Once epsilon hit 0.01 it was mostly repeating its own bad policy, so it took a long time to find anything better. Greedy: -13.3. |
| 3 | 0.0001 | 0.99 | 32 | 1.0 | 0.10 | 100000 | Started improving around 260k steps, ended at -16.3 (best episode -10). Slower than the baseline, the extra random moves late in training didn't help it learn faster. Greedy: -17.3. |
| 4 | 0.0001 | 0.99 | 32 | 1.0 | 0.20 | 100000 | Better than I expected. Started improving around 200k and ended at -13.3, even with 20% random actions still happening. Greedy it does better: -8.0. |
| 5 | 0.0001 | 0.99 | 32 | 1.0 | 0.05 | 25000 | Very close to the baseline (-13.7 at the end, best episode -3). Learning only kicked in around 250k anyway, so reaching 0.05 at 25k instead of 100k didn't change much. Greedy: -15.0. |
| 6 | 0.0001 | 0.99 | 32 | 1.0 | 0.05 | 250000 | Slow start (only got going around 290k), peaked at -13.8 then dropped to -16.4. Greedy was inconsistent: -1 in one game and -20 in the other two. |
| 7 | 0.0001 | 0.99 | 32 | 1.0 | 0.05 | 400000 | I thought this would be the worst because it explores for almost the whole run, but it was the best of the runs starting at 1.0. Peaked at -12.2, longest rallies of all (about 2900 steps per episode). Greedy: -5.7, and it won one game 21-20. |
| 8 | 0.0001 | 0.99 | 32 | 0.5 | 0.05 | 100000 | Learned a bit earlier than the baseline (around 220k) and peaked at -13.3, but got worse near the end (-16.4). Worst greedy score: -18.3. |
| 9 | 0.0001 | 0.99 | 32 | 0.1 | 0.01 | 50000 | Best by far. Started improving around 120k, jumped from about -12 to +13 between 250k and 300k steps, and ended at +18.5 (best episode +21). Episodes got shorter after that because it wins points quickly instead of long rallies. Greedy: +20 in all 3 games. |
| 10 | 0.0001 | 0.99 | 32 | 0.01 | 0.01 | 50000 | Wanted to see if even less exploration than experiment 9 is better. It isn't. It started improving early (around 145k) but levelled off around -10.6 from 300k onwards (best episode +3). The rallies got very long (about 3900 steps per episode), so it learned to keep the ball in play but not how to win points. Greedy: -3.3 (+2 once, -6 twice). |

Models, logs and plots for each run are in `Ihimbazwe_Samuel/expXX/`.

## Best and worst

Best: experiment 9 (start 0.1, end 0.01). It is the only run that actually learned to win, +18.5 in training and +20 greedy.

Worst: experiment 2 (start 1.0, end 0.01). It barely moved off -21 for most of the run and finished at -18.7. Experiment 8 had the worst greedy score (-18.3) because it got worse near the end.

## What helped and what hurt

Starting with very little exploration helped the most. In Pong the agent gets a reward every time a point ends, so it doesn't need lots of random actions to find rewards. With epsilon at 0.1 from the start, most of what went into the replay buffer came from the agent's own policy, so it learned from situations that actually matter (tracking and hitting the ball) much earlier.

For the runs that started at 1.0, changing the end value or the decay length didn't make a clear difference. They all ended between -13 and -19, and the order didn't follow a pattern (0.2 at the end did better than 0.1, the longest decay did better than the shortest). I only ran each setting once with one seed, and DQN results change a lot between runs, so I don't think those differences are reliable.

Ending at 0.01 hurt when it came after a full random start (experiment 2), but worked in experiment 9. So a low final epsilon isn't bad on its own, what mattered more was the low starting value.

Going all the way down to 0.01 from the start (experiment 10) was too little though. It learned to return the ball but got stuck there, so a small amount of exploration early on (0.1 in experiment 9) is still needed to find how to actually win points.

## Final configuration

Experiment 9: learning rate 0.0001, gamma 0.99, batch size 32, epsilon 0.1 to 0.01 over 50k steps.

Because experiment 9 was so far ahead of everything else, I wanted to check it wasn't just a lucky run. I trained the same settings again with a different seed (7 instead of 42) for 1M steps. It was slower (+10.5 at 500k compared to +18.5) but it also learned to win, and finished at +16.6 over the last 20 episodes with a best episode of +21. So the result holds up, and the gap in speed between the two seeds shows how much DQN runs can vary.
