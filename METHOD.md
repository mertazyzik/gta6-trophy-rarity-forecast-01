# Method

## The quantity

For each of Grand Theft Auto VI's PlayStation 5 trophies, forecast its earn
rate on Exophase at the case's target: the percent of the players Exophase
tracks for the game who have earned it. Output: five quantiles
(0.05, 0.25, 0.5, 0.75, 0.95) per trophy. Scored by mean pinball loss on the
logit scale.

## Decomposition

A rate is a ratio, `r_i(t) = E_i(t) / N(t)`:

- `N(t)`: players tracked. Grows as players sync profiles; Exophase refreshes
  its counts on its own schedule (hours apart), so the recorded history is a
  step function.
- `E_i(t)`: earners of trophy `i`. Grows two ways: new players arrive having
  already earned it, and tracked players earn it as they play.

## Tracked players

Distinct page states (consecutive identical reads collapsed) give refresh
points `(t_k, N_k)`. Two shapes are fitted in log space:

- power law: `log N = a + b * log(t + 1)`, `t` hours since release;
- saturating exponential: `N = N_inf * (1 - exp(-t / tau))`, tau gridded.

They are blended by leave-last-out error on the most recent refreshes, and
extrapolated to the target; the blend's error sets the lognormal spread
(`sigma_logN`, floored at 0.12, capped at 0.6). With fewer than three distinct
states (flat history, as on launch night) a prior is used instead: steep
launch growth, a power law with exponent near 1 (the documented first refresh
took tracked players 638 -> 3868), slowing with time since release.

## Earners: the flow

Per trophy, over each refresh transition `k`:

```
dE_i = q_i * dN_k + h_i * dt_k * (N - E_i)
```

`q_i` is the share of newly tracked players who already earned the trophy;
`h_i` is the hourly hazard for a tracked non-earner. `(q_i, h_i)` are fitted
by weighted least squares over the transitions (recency half-life 24h),
shrunk toward priors by two pseudo-transitions:

- `q` prior: new players are fresh players. The documented first refresh
  added 3230 players and only 103 non-opening trophies between them (~0.03
  a player); the opening trophy (at 100%) takes `q = 0.95`, the rest share a
  small mass in proportion to `r^0.7` (early trophies first). The mass grows
  with time since release, as later joiners have played longer before
  syncing (up to 3x).
- `h` prior: asymptotic hazard `0.005 * r/(r+0.02)` per hour, lower for
  silver/gold/platinum, times a ramp `(a/(a+tau))^2` at the population's
  mean age `a` (about half the game's age — players join over time, so the
  average tracked player is much younger than the game). `tau` is the
  typical hours of play before the trophy can be earned: chapters 9–24h in
  order, final mission 35h, grinds 80–120h, 100% completion 200h, platinum
  250h, opening trophy 0.5h.

## Ceiling (reach)

Earners do not grow toward `N`: not everyone finishes the story or grinds.
Each trophy has an eventual reach (chapter 1: 0.70, chapter 2: 0.60, chapter 3:
0.50, chapter 4: 0.42, final mission: 0.35, silver: 0.22, bronze: 0.30, 100%
completion: 0.04, platinum: 0.03, opening: 1.0), raised to at least 1.5x the
rate already reached. In the simulation, earners grow toward `reach * N`
(the fitted hazard rescales by `1/reach`), and the reach itself is sampled
(logit-normal, sd 0.25–0.4).

## Refresh lag

The answering read shows the page as of the last refresh before the target,
not the target itself. Refreshes are projected forward on their observed
schedule (median gap between estimated refresh times, with jitter); the
forecast is the state at the last projected refresh at or before the target.
With a flat history the gap prior is ~4.5h (the documented first refresh).

## Quantiles

A Monte Carlo (320 samples, fixed seed `20261119`, so two runs agree byte
for byte) over: the growth-curve uncertainty, `(q, h)` sampling error
(lognormal, so the median stays at the estimate), the reach, the refresh
schedule, and process noise on the logit scale (`0.05 + 0.05*sqrt(days)`,
wider for flat histories). Empirical quantiles at the five levels. A trophy
everyone holds stays (almost) held. Any failure degrades to the persistence
baseline (median the last rate, spread `0.5*sqrt(days)` logits), so the
program always writes a valid forecast.

## Validation

- Practice cases (90-minute horizon, flat history): loss ~0.52–0.75 across
  four reconstructed answers, vs ~1.40–1.58 for the persistence baseline
  (the house's documented practice loss is ~1.51).
- Synthetic final cases (a simulated Exophase world: power-law player
  growth, per-trophy saturating earn curves, stale reads refreshed every
  2–8h; nine worlds varying growth, reach and timescales): mean loss ~0.11
  vs ~0.36 for the persistence baseline, 90% coverage ~0.96–1.00.
- Deterministic (byte-identical outputs across runs), ~1s per case, valid
  output on edge cases (single read, null rates, empty history).
