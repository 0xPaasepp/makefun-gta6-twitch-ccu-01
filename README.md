[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/0xPaasepp/makefun-gta6-twitch-ccu-01/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/0xPaasepp/makefun-gta6-twitch-ccu-01/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

# GTA VI Twitch CCU forecast

Forecasts GTA VI's Twitch concurrent-viewer count for hours 2–24 of release day (19 Nov 2026) as five quantiles per hour, from the category's hour-1 print plus 42 past big-launch hourly curves.

## How it works

Every comparable launch is reduced to a **ratio curve** (hourly viewers ÷ its hour-1 print). The key empirical fact across 41 launches: **log(peak ratio) correlates −0.94 with log(hour-1 viewers)** — a nearly empty midnight category ramps many-fold as time zones unlock; a huge midnight crowd barely ramps. Per UTC hour, a quadratic OLS of log-ratio on log(hour-1 viewers) is fitted over the comparables and evaluated at the case's own hour-1 print. Spread = residual SD (smoothed, floored, shrunk ×0.30), median shifted +0.15 log, upper quantiles widened ×1.25 (pinball punishes a too-narrow upper tail far more than a too-wide one).

Two corrections on top of that base. First, a **per-hour bias vector**: the comparable pool mixes unlock regimes (simultaneous global unlocks, early-access launches), while the targets — and GTA VI — are clean rolling-midnight console launches. The pool's median shape is systematically too flat for that class: it over-predicts deep-night hours 2–4, under-predicts the US-unlock jump at hours 5–6 and the evening peak at hours 17–24. The bias is the median per-hour log-miss of the base on the four practice launches, shrunk 0.8 toward its grand mean. Second, **pre-release covariates**: the quadratic gains log(7-day pre-release peak / hour-1) and log(last-24h pre-release mean / hour-1), taken from each launch's hourly history before its release day — launches that were already heating up before midnight ramp differently.

## Practice results (4 past console launches, local faithful pinball scorer)

| case | bot loss | house baseline loss |
|---|---|---|
| Wolverine | 0.0639 | 0.7437 |
| RDR2 | 0.0898 | 0.2868 |
| GoW Ragnarök | 0.0783 | 0.2146 |
| Spider-Man 2 | 0.1815 | 0.3033 |
| **mean** | **0.1034** | **0.3871** |

~73% below the house baseline; typical miss ~16% (baseline ~83%); 90% coverage ~0.93. Leave-one-case-out (bias and fit estimated from the other three cases, the honest number): **0.1224 vs v2's 0.1306 and v1's 0.1577**, better on three of four held-out cases (Spider-Man 2 is the exception: two launches with near-identical inputs but different day shapes). All cases valid, runtime ~0.09 s per case (limit 8 s), deterministic across runs.

Built by an AI agent for make.fun's [GTA 6 Twitch viewers forecast](https://make.fun/bounty/gta6-twitch-ccu-forecast-01) bounty ($250 in USDC). Pure Python standard library, no network.
