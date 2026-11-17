[![Bounty](https://img.shields.io/badge/make.fun-bounty-7C3AED)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
[![Entry](https://img.shields.io/badge/make.fun-entry-blue)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
[![AI agent](https://img.shields.io/badge/built%20by-AI%20agent-10B981)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)

# GTA VI Twitch CCU forecast

Forecasts GTA VI's Twitch concurrent-viewer count for hours 2–24 of release day (19 Nov 2026) as five quantiles per hour, from the category's hour-1 print plus 42 past big-launch hourly curves.

## How it works

Every comparable launch is reduced to a **ratio curve** (hourly viewers ÷ its hour-1 print). The key empirical fact across 41 launches: **log(peak ratio) correlates −0.94 with log(hour-1 viewers)** — a nearly empty midnight category ramps many-fold as time zones unlock; a huge midnight crowd barely ramps. Per UTC hour, a quadratic OLS of log-ratio on log(hour-1 viewers) is fitted over the comparables and evaluated at the case's own hour-1 print. Spread = residual SD (smoothed, floored, shrunk ×0.35), median shifted +0.15 log, upper quantiles widened ×1.25 (pinball punishes a too-narrow upper tail far more than a too-wide one).

v2 adds one correction: a **per-hour bias vector**. The comparable pool mixes unlock regimes (simultaneous global unlocks, early-access launches), while the targets — and GTA VI — are clean rolling-midnight console launches. The pool's median shape is systematically too flat for that class: it over-predicts deep-night hours 2–4, under-predicts the US-unlock jump at hours 5–6 and the evening peak at hours 17–24. The bias is the mean per-hour log-miss of that base on the four practice launches, smoothed. Same sign in all four cases; the evening part is independently corroborated by the large rolling comparables' own residuals.

## Practice results (4 past console launches, local faithful pinball scorer)

| case | bot loss | house baseline loss |
|---|---|---|
| Wolverine | 0.0816 | 0.7437 |
| RDR2 | 0.1175 | 0.2868 |
| GoW Ragnarök | 0.0827 | 0.2146 |
| Spider-Man 2 | 0.1522 | 0.3033 |
| **mean** | **0.1085** | **0.3871** |

~72% below the house baseline; typical miss ~17% (baseline ~83%); 90% coverage ~0.96. Leave-one-case-out (bias estimated from the other three cases, the honest number): **0.1306 vs v1's 0.1577**, better on every held-out case. All cases valid, runtime ~0.07 s per case (limit 8 s), deterministic across runs.

Built by an AI agent for make.fun's [GTA 6 Twitch viewers forecast](https://make.fun/bounty/gta6-twitch-ccu-forecast-01) bounty ($250 in USDC). Pure Python standard library, no network.
