# GTA VI Twitch CCU forecast — vocaine entry

Forecasts hourly Twitch concurrent viewers in the Grand Theft Auto VI
category for hours 2..24 of release day (2026-11-19 UTC), as five quantiles
per hour, from the hour-1 print and the case's comparables.

## Approach: nearest-neighbour analogs

For the target launch, the program scores every comparable past launch by
similarity and takes the K=8 nearest, with weights 1/(distance + 0.05):

- **(a) scale** — |log p1 − log h1|, weight 2.0: how big the launch's first
  hour was;
- **(b) pre-release shape** — log viewers at anchors 14/10/7/5/3/2/1 days
  before release day, each relative to the day −1 level (scale-free), plus
  the 3-day log slope and the launch jump log(p1) − log(day −1 level);
  weights 0.5 (shape) / 0.5 (jump) / 0.25 (slope);
- **(c) platform/unlock** — console-only vs multi-platform penalty 0.5
  (GTA VI is PS5 + Xbox Series X|S, midnight-local rolling unlock);
  unlock-class penalty tuned to 0.

Each selected analog contributes its own launch-day curve as a ratio to its
own hour-1 print. The forecast median for hour h is p1 × exp(weighted mean
of the analogs' log ratios); the spread is the weighted dispersion of those
log ratios at that hour (floor 0.20, ×1.25, cap 0.90). Everything is
computed at runtime from the case's own data — no answer is hardcoded.
The Grand Theft Auto VI pre-release comparable is excluded automatically
(it has no launch-day curve).

## Validation (local, leave-one-out on the kit's 38 comparables)

Each comparable was replayed as a simulated case (its pre-release history +
hour-1 print; answer = its real hours 2..24) and scored with the exact
pinball loss on the log scale from RULES.txt:

- **this entry: 0.3910** mean pinball loss (37 cases; big launches
  with hour-1 ≥ 20k viewers: **0.2496**)
- **house starter baseline, same set: 0.9297** → 57.9% better
- worst LOO cases are tiny/sparse launches (Black Myth: Wukong 1.75,
  Marathon 1.17); best are big console launches (Modern Warfare III 0.114,
  Elden Ring Nightreign 0.124, Hogwarts Legacy 0.131)

Practice cases p00–p03 run valid (23 rows, hours 2..24, quantiles > 0,
≤ 1e9, non-decreasing, ≤ 65,536 bytes), byte-identical across two runs,
~0.1 s per case (limit 8 s).

## Run

```
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Python 3.13, standard library only, no network, deterministic.
