"""GTA VI trophy rarity forecaster (make.fun gta6-trophy-rarity-forecast-01).

Reads one case per *.in file (Exophase trophy earn rates for Grand Theft Auto
VI on PS5 as the house recorded them) and writes, for every trophy, five
quantiles of its earn rate at the case's target, as /output/<id>.out.

Model
-----
An earn rate is a ratio: r_i(t) = E_i(t) / N(t), the earners of trophy i over
the players Exophase tracks. Exophase refreshes its counts on its own
schedule (hours apart), so the history is a step function: long flat stretches
(the page as of its last refresh) punctuated by jumps. The forecaster works
with the jumps, because that is where the information is:

* N(t), the tracked players, is fitted with two growth curves (a power law in
  hours since release and a saturating exponential), blended by how well each
  predicted the most recent refreshes, and extrapolated to the target.
* For each trophy, earners follow the flow
      dE_i = q_i * dN + h_i * (N - E_i) * dt
  i.e. newly tracked players arrive having already earned the trophy with
  probability q_i, and tracked players who have not earned it yet do so at
  hazard h_i per hour. (q_i, h_i) are fitted per trophy by ridge regression
  over the refresh transitions, shrunk toward priors built from the trophy's
  current rate and tier, with recency weighting.
* The forecast is a Monte Carlo (fixed seed, so two runs agree byte for byte)
  over the growth-curve uncertainty, the (q_i, h_i) sampling error, the
  refresh lag (the answering read shows the page as of the last refresh
  before the target, not the target itself) and process noise, giving
  empirical quantiles at 0.05 / 0.25 / 0.5 / 0.75 / 0.95.

When the history is flat (no refresh observed, as on launch night) the flow
is unidentifiable from the case alone, so the forecaster falls back on priors
calibrated on the one refresh the competition's rules document: the page
unchanged from the first probe through 04:00Z, then refreshed by 04:30Z with
tracked players 638 -> 3868 while every rate but the 100% one fell, i.e. a
stale page, a large jump when it refreshes, and new players who have earned
about one trophy each (the opening one) and almost nothing else. Any failure
anywhere degrades to the persistence baseline, so the program always writes a
valid forecast.

Standard library only; no network. Deterministic: fixed RNG seed, sorted
inputs, no wall-clock or hash-order dependence.
"""

import json
import math
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]
# Persistence baseline spread (logit units per sqrt day), used as fallback
# and as a floor on the forecast spread.
BASE_SIGMA = 0.5
SEED = 20261119


def clamp(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    c = clamp(pct)
    return math.log(c / (100.0 - c))


def inv_logit(l):
    return 100.0 / (1.0 + math.exp(-l))


def round4(x):
    """x rounded to 4 decimals, half up, as JavaScript's Math.round does it."""
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if value == int(value) else value


def parse_time(s):
    """ISO 8601 UTC timestamp -> epoch seconds (float)."""
    if not s:
        return None
    try:
        from datetime import datetime, timezone
        return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()
    except Exception:
        return None


# --------------------------------------------------------------------------
# Case history: collapse the reads into distinct states (refreshes).
# --------------------------------------------------------------------------

class State(object):
    __slots__ = ("t", "n", "earned", "rates")

    def __init__(self, t, n, earned, rates):
        self.t = t            # hours since release, first read showing it
        self.n = n            # players tracked
        self.earned = earned  # trophies earned (page total)
        self.rates = rates    # list of float|None, per trophy


def distinct_states(case):
    """Collapse history into the distinct page states, oldest first.

    Consecutive reads with identical (players_tracked, rates) are one state:
    the page had not refreshed between them. A state is dated at the first
    read that shows it, which is when the refresh that produced it is first
    seen (the refresh itself happened between the previous read and this one).

    Also returns refresh info: the estimated refresh time of each state
    after the first (the midpoint between the previous read and the first
    read showing the state) and the median read cadence, in hours since
    release.
    """
    hist = case.get("history") or []
    release = parse_time((case.get("game") or {}).get("release_date") or "")
    if release is None:
        # Fall back to the first read minus a day; only the differences
        # between times matter for the fits.
        first = parse_time(hist[0].get("at")) if hist else 0.0
        release = (first or 0.0) - 86400.0
    states = []
    refresh_times = []
    read_times = []
    prev_key = None
    prev_read_t = None
    for read in hist:
        at = parse_time(read.get("at") or read.get("slot"))
        if at is None:
            continue
        n = read.get("players_tracked")
        rates = read.get("rates") or []
        if not isinstance(n, (int, float)) or n <= 0:
            continue
        t = (at - release) / 3600.0
        read_times.append(t)
        key = (n, tuple(rates))
        if key == prev_key:
            prev_read_t = t
            continue
        prev_key = key
        if states and prev_read_t is not None:
            refresh_times.append((prev_read_t + t) / 2.0)
        prev_read_t = t
        states.append(State(t, float(n),
                            read.get("trophies_earned"), list(rates)))
    cadence = 0.5
    if len(read_times) >= 2:
        diffs = sorted(read_times[k + 1] - read_times[k]
                       for k in range(len(read_times) - 1))
        diffs = [d for d in diffs if d > 0]
        if diffs:
            cadence = diffs[len(diffs) // 2]
    return states, release, {"refresh_times": refresh_times,
                             "cadence": cadence,
                             "last_read_t": read_times[-1] if read_times else None}


def earner_counts(states, n_trophies):
    """Per-state earner counts E_i = rate * N / 100 (rounded to integers).

    Rates are printed to 2 decimals, so for the player counts seen here the
    rounding recovers the integer earner count well. Returns a list (per
    state) of lists (per trophy) of float|None.
    """
    out = []
    for s in states:
        row = []
        for i in range(n_trophies):
            r = s.rates[i] if i < len(s.rates) else None
            if r is None:
                row.append(None)
            else:
                row.append(round(float(r) * s.n / 100.0))
        out.append(row)
    return out


def last_known(values):
    """Last non-None entry of a list, or None."""
    for v in reversed(values):
        if v is not None:
            return v
    return None


# --------------------------------------------------------------------------
# Tracked-player growth N(t).
# --------------------------------------------------------------------------

def _ols(xs, ys):
    """Least squares y ~ a + b*x. Returns (a, b, sse)."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return my, 0.0, sum((y - my) ** 2 for y in ys)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = my - b * mx
    sse = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    return a, b, sse


def fit_player_growth(states, t_target_abs):
    """Fit N(t) from the distinct states; forecast log N at the target.

    Two shapes are fitted in log space and blended by leave-last-out error:
    a power law in hours since release, and a saturating exponential. With
    fewer than three distinct states there is nothing to fit, so a prior is
    used instead: launch-night growth was steep (the documented first
    refresh took tracked players 638 -> 3868), slowing like a power law
    with exponent ~1.5.

    Returns (log_n_target, sigma_log_n, log_n_last, t_last_state).
    """
    times = [s.t for s in states]
    logn = [math.log(max(s.n, 1.0)) for s in states]
    t_last = times[-1]
    log_n_last = logn[-1]
    target_t = t_target_abs

    def power_law_predict(ts, tt):
        a, b, _ = _ols([math.log(max(t, 0.0) + 1.0) for t in ts],
                       [math.log(max(s.n, 1.0)) for s in states[:len(ts)]])
        return a + b * math.log(max(tt, 0.0) + 1.0)

    def saturating_predict(ts, tt, taus=(6.0, 12.0, 24.0, 48.0, 96.0, 192.0, 384.0, 768.0)):
        ns = [s.n for s in states[:len(ts)]]
        best = None
        for tau in taus:
            z = [1.0 - math.exp(-max(t, 0.0) / tau) for t in ts]
            zz = sum(v * v for v in z)
            if zz <= 0:
                continue
            ninf = sum(v * n for v, n in zip(z, ns)) / zz
            if ninf <= 0:
                continue
            sse = sum((math.log(max(n, 1.0)) - math.log(max(ninf * v, 1.0))) ** 2
                      for v, n in zip(z, ns))
            if best is None or sse < best[0]:
                best = (sse, tau, ninf)
        if best is None:
            return None
        _, tau, ninf = best
        return math.log(max(ninf * (1.0 - math.exp(-max(tt, 0.0) / tau)), 1.0))

    if len(states) < 3:
        # Prior: steep launch growth, slowing like a power law.
        alpha = 1.5
        log_target = log_n_last + alpha * (
            math.log(max(target_t, 0.0) + 1.0) - math.log(max(t_last, 0.0) + 1.0))
        return log_target, 0.45, log_n_last, t_last

    # Leave-last-out: how well does each shape predict the last states?
    k_hold = min(3, len(states) - 1)
    train = states[:-k_hold]
    hold = states[-k_hold:]
    errs = {"power": [], "sat": []}
    for s in hold:
        p = power_law_predict([x.t for x in train], s.t)
        errs["power"].append(math.log(max(s.n, 1.0)) - p)
        q = saturating_predict([x.t for x in train], s.t)
        if q is not None:
            errs["sat"].append(math.log(max(s.n, 1.0)) - q)
    rms = {}
    for name, e in errs.items():
        rms[name] = math.sqrt(sum(v * v for v in e) / len(e)) if e else 0.35
    # Blend weights: inverse squared error, with a floor so neither shape
    # is ever discarded outright.
    w_power = 1.0 / (rms["power"] ** 2 + 0.01)
    w_sat = 1.0 / (rms["sat"] ** 2 + 0.01)
    p_full = power_law_predict(times, target_t)
    s_full = saturating_predict(times, target_t)
    if s_full is None:
        log_target = p_full
        sigma = max(0.12, rms["power"])
    else:
        log_target = (w_power * p_full + w_sat * s_full) / (w_power + w_sat)
        sigma = max(0.12, min(rms["power"], rms["sat"]))
    # Longer horizons are less certain; and never let the forecast go below
    # the last observed count.
    sigma = min(0.6, sigma + 0.03 * math.sqrt(max(target_t - t_last, 0.0) / 24.0))
    log_target = max(log_target, log_n_last)
    return log_target, sigma, log_n_last, t_last


# --------------------------------------------------------------------------
# Per-trophy flow: dE = q*dN + h*(N - E)*dt.
# --------------------------------------------------------------------------

def mean_age(t):
    """Mean age (hours since they started playing) of the tracked
    population at time t since release.

    Players join over time, so the average tracked player is much younger
    than the game: with power-law joining the mean age is roughly half the
    game's age. Using t itself would pretend every player has played since
    launch, which wildly overstates how fast trophies are earned.
    """
    return 0.45 * max(t, 0.0) + 0.5


def reach_prior(trophy, last_rate):
    """Prior for the eventual share of tracked players who will ever earn
    a trophy (its ceiling).

    Not everyone finishes the story or grinds: chapter trophies top out
    well below 100%, 100% completion and the platinum in the low single
    digits. The ceiling is at least 1.5x the rate already reached.
    """
    desc = (trophy.get("description") or "") + " " + (trophy.get("name") or "")
    if trophy.get("id") == 2 or (last_rate is not None and last_rate >= 99.9):
        base = 1.0
    elif "Chapter 1" in desc:
        base = 0.70
    elif "Chapter 2" in desc:
        base = 0.60
    elif "Chapter 3" in desc:
        base = 0.50
    elif "Chapter 4" in desc:
        base = 0.42
    elif "final mission" in desc:
        base = 0.35
    elif "100% Completion" in desc:
        base = 0.04
    elif trophy.get("tier") == "platinum":
        base = 0.03
    elif trophy.get("tier") == "gold":
        base = 0.08
    elif trophy.get("tier") == "silver":
        base = 0.22
    else:
        base = 0.30
    if last_rate is not None:
        base = max(base, min(1.5 * last_rate / 100.0, 0.95))
    return min(max(base, 0.005), 1.0)


def tau_hours(trophy):
    """Typical hours of play before a trophy can be earned, from its text.

    Story chapters come in order (later chapters take longer to reach);
    grinds and 100% completion take far longer; the opening trophy is
    earned in the first minutes.
    """
    desc = (trophy.get("description") or "") + " " + (trophy.get("name") or "")
    if trophy.get("id") == 2:
        return 0.5
    tau = None
    for ch in (1, 2, 3, 4):
        if "Chapter %d" % ch in desc:
            tau = 4.0 + 5.0 * ch
    if "final mission" in desc:
        tau = max(tau or 0.0, 35.0)
    if tau is not None:
        return tau
    if "100% Completion" in desc:
        return 200.0
    tier = trophy.get("tier")
    if tier == "platinum":
        return 250.0
    if tier == "gold":
        return 150.0
    if tier == "silver":
        return 80.0
    for phrase in ("250 Stunts", "100 vehicles", "100 locations",
                   "50 Challenges", "Collectible Card", "all Shooting Range"):
        if phrase in desc:
            return 120.0
    return 30.0


def ramp(t, tau):
    """How much of a trophy's earning hazard has switched on by age t.

    Nobody earns a trophy that takes tau hours in their first minutes; the
    hazard ramps up as the population ages into it. Squared, so the start
    is slow.
    """
    if t <= 0 or tau <= 0:
        return 0.0
    f = t / (t + tau)
    return f * f


def trophy_priors(case, states, earners, t_now=None):
    """Prior (q, h) per trophy from its current rate and tier.

    q_i is the share of newly tracked players who already earned the trophy.
    New players are fresh players: they have earned the opening trophy
    (the one at 100%) and almost nothing else -- the documented first
    refresh added 3230 players and only 103 non-opening trophies between
    them, about 0.03 a player. So the non-opening q mass is small and spread
    in proportion to r^0.7 (early trophies first), and the opening trophy
    takes q = 0.95; the mass grows with time since release, as later
    joiners have played longer before syncing. h_i is the hourly hazard for
    a tracked non-earner at the checkpoint's age: the asymptotic hazard,
    higher for trophies many players are earning now (high current rate)
    and lower for silver/gold/platinum grinds, times the ramp at the
    checkpoint's age.
    """
    trophies = case["trophies"]
    n_tr = len(trophies)
    t_last = states[-1].t if states else 0.0
    if t_now is None:
        t_now = t_last
    t_last = t_now
    last_rates = []
    for i in range(n_tr):
        col = [e[i] for e in earners]
        last_e = last_known(col)
        last_n = states[-1].n if states else 1.0
        last_rates.append(100.0 * last_e / last_n if last_e is not None else 0.0)
    tier_h = {"bronze": 1.0, "silver": 0.5, "gold": 0.25, "platinum": 0.1}
    weights = []
    for i, t in enumerate(trophies):
        r = last_rates[i]
        weights.append((max(r, 0.01) ** 0.7))
    wsum = sum(weights) or 1.0
    # Non-opening q mass: ~0.15 a player at launch, more later.
    q_mass = 0.15 * min(1.0 + max(t_last, 0.0) / 24.0, 3.0)
    priors = []
    for i, t in enumerate(trophies):
        r = last_rates[i]
        if r >= 99.9:
            q0 = 0.95
        else:
            q0 = q_mass * weights[i] / wsum
        h_asym = 0.005 * (r / (r + 0.02)) * tier_h.get(t.get("tier"), 1.0)
        h0 = h_asym * ramp(mean_age(max(t_last, 0.0)), tau_hours(t))
        priors.append((min(max(q0, 0.0), 1.0), min(max(h0, 0.0), 0.05)))
    return priors, last_rates


def fit_flow(states, earners, priors, t_now=None):
    """Ridge-fit (q_i, h_i) per trophy over the refresh transitions.

    For each transition k: dE_i = q_i * dN_k + h_i * dt_k * (N - E_i) + noise.
    Weighted by recency (24h half-life); shrunk toward the priors with a
    strength worth about two transitions. Returns per trophy
    (q, h, sd_q, sd_h, t_ref), t_ref the recency-weighted mean transition
    time the fitted hazard belongs to.
    """
    n_tr = len(priors)
    transitions = []
    for k in range(1, len(states)):
        dt = states[k].t - states[k - 1].t
        dn = states[k].n - states[k - 1].n
        if dt <= 0:
            continue
        transitions.append((k, dt, dn))
    t_last = states[-1].t if states else 0.0
    if t_now is None:
        t_now = t_last
    out = []
    for i in range(n_tr):
        q0, h0 = priors[i]
        # Recency-weighted mean transition time: the age the fitted hazard
        # belongs to (the simulation ramps it from there).
        t_ref = t_now
        if len(transitions) >= 3:
            wsum_ref = 0.0
            tsum_ref = 0.0
            for (k, dt, dn) in transitions:
                w = 0.5 ** ((t_last - states[k].t) / 24.0)
                wsum_ref += w
                tsum_ref += w * states[k].t
            if wsum_ref > 0:
                t_ref = tsum_ref / wsum_ref
        if len(transitions) < 3:
            out.append((q0, h0, 0.5 * max(q0, 0.02), 0.6 * max(h0, 0.004), t_ref))
            continue
        # Normal equations for [q, h]. The prior enters as pseudo-rows, so
        # its strength scales with the regressors: one pseudo-transition
        # saying "dN_typ new players earned q0 of them" and one saying
        # "exposure_typ non-earner-hours earned h0 of them", each worth
        # two real transitions.
        rows = []
        for (k, dt, dn) in transitions:
            e_prev = earners[k - 1][i]
            e_now = earners[k][i]
            if e_prev is None or e_now is None:
                continue
            x1 = dn
            x2 = dt * max(states[k - 1].n - e_prev, 0.0)
            y = e_now - e_prev
            w = 0.5 ** ((t_last - states[k].t) / 24.0)
            rows.append((x1, x2, y, w))
        if not rows:
            out.append((q0, h0, 0.5 * max(q0, 0.02), 0.6 * max(h0, 0.004), t_ref))
            continue
        dn_typ = sorted(abs(r[0]) for r in rows)[len(rows) // 2] or 1.0
        ex_typ = sorted(abs(r[1]) for r in rows)[len(rows) // 2] or 1.0
        w_prior = 2.0
        rows.append((dn_typ, 0.0, q0 * dn_typ, w_prior))
        rows.append((0.0, ex_typ, h0 * ex_typ, w_prior))
        s11 = s12 = s22 = t1 = t2 = 0.0
        for (x1, x2, y, w) in rows:
            s11 += w * x1 * x1
            s12 += w * x1 * x2
            s22 += w * x2 * x2
            t1 += w * x1 * y
            t2 += w * x2 * y
        det = s11 * s22 - s12 * s12
        if abs(det) < 1e-12:
            out.append((q0, h0, 0.5 * max(q0, 0.02), 0.6 * max(h0, 0.004), t_ref))
            continue
        q = (t1 * s22 - t2 * s12) / det
        h = (s11 * t2 - s12 * t1) / det
        # Residual variance over the real rows only -> sampling sds
        # (floored, so the Monte Carlo never trusts a fit more than the
        # data warrants).
        resid = 0.0
        cnt = 0
        for (x1, x2, y, w) in rows[:-2]:
            resid += w * (y - (q * x1 + h * x2)) ** 2
            cnt += 1
        dof = max(cnt - 2, 1)
        sig2 = resid / dof
        sd_q = math.sqrt(max(sig2 * s22 / det, 0.0)) if det > 0 else 0.5 * max(q0, 0.02)
        sd_h = math.sqrt(max(sig2 * s11 / det, 0.0)) if det > 0 else 0.6 * max(h0, 0.004)
        q = min(max(q, 0.0), 1.0)
        h = min(max(h, 0.0), 0.2)
        sd_q = max(sd_q, 0.30 * max(q, 0.02), 0.01)
        sd_h = max(sd_h, 0.40 * max(h, 0.004), 0.002)
        out.append((q, h, sd_q, sd_h, t_ref))
    return out


# --------------------------------------------------------------------------
# Forecast: Monte Carlo over growth, flow and refresh lag.
# --------------------------------------------------------------------------

def _lognormal_sample(rng, median, sd, cap):
    """Sample around `median` with lognormal noise of sd `sd` (in linear
    units), capped at `cap`. The median of the samples is `median`."""
    if median <= 0:
        return min(max(abs(rng.gauss(0.0, sd)), 0.0), cap)
    s = min(sd / median, 1.0)
    return min(max(median * math.exp(rng.gauss(0.0, s)), 0.0), cap)


def persistence_rows(case, last_rates):
    """House baseline rows: median the last rate, spread 0.5*sqrt(days)."""
    spread = BASE_SIGMA * math.sqrt(max(case.get("horizon_minutes", 0), 0) / 1440.0)
    rows = []
    for i, t in enumerate(case["trophies"]):
        r = last_rates[i] if i < len(last_rates) else 0.0
        lg = logit(r if r is not None else 0.0)
        rows.append({"id": t["id"],
                     "q": [round4(inv_logit(lg + z * spread)) for z in Z]})
    return rows


def forecast_case(case):
    trophies = case.get("trophies") or []
    horizon_min = case.get("horizon_minutes", 0) or 0
    horizon_h = horizon_min / 60.0
    n_tr = len(trophies)
    if n_tr == 0 or horizon_h < 0:
        raise ValueError("case lists no trophies")
    states, release, refresh_info = distinct_states(case)
    if not states:
        raise ValueError("case has no history")
    earners = earner_counts(states, n_tr)
    # Time base: the checkpoint is the case's `at`, not the last distinct
    # state -- a stale page's last state can predate the checkpoint by
    # hours. The target is the checkpoint plus the horizon.
    at_sec = parse_time(case.get("at"))
    t_check = ((at_sec - release) / 3600.0 if at_sec is not None
               else (refresh_info["last_read_t"] or states[-1].t))
    t_target_abs = t_check + horizon_h
    priors, last_rates = trophy_priors(case, states, earners, t_now=t_check)
    flow = fit_flow(states, earners, priors, t_now=t_check)
    log_n_target, sigma_log_n, log_n_last, t_last_state = fit_player_growth(
        states, t_target_abs)
    n_last = math.exp(log_n_last)
    last_e = []
    for i in range(n_tr):
        v = last_known([e[i] for e in earners])
        last_e.append(float(v) if v is not None else 0.0)

    # Refresh schedule: Exophase refreshes on its own clock, hours apart,
    # so the answering read shows the page as of the LAST refresh before
    # the target -- not a memoryless lag. Estimate the refresh gap from the
    # gaps between the estimated refresh times (median and spread); the
    # last refresh is the last estimated one. With no refresh observed
    # (flat history) use the documented first refresh as the prior: the
    # page was unchanged for hours, then refreshed, a gap near 4.5 hours.
    refresh_times = refresh_info["refresh_times"]
    if len(refresh_times) >= 2:
        rgaps = sorted(refresh_times[k + 1] - refresh_times[k]
                       for k in range(len(refresh_times) - 1))
        rgaps = [g for g in rgaps if g > 0]
        gap_med = rgaps[len(rgaps) // 2] if rgaps else 4.5
        gap_sd = max((rgaps[-1] - rgaps[0]) / 4.0 if len(rgaps) > 1 else 0.0,
                     0.25 * gap_med)
        last_refresh = refresh_times[-1]
    elif len(refresh_times) == 1:
        gap_med, gap_sd = 4.5, 1.5
        last_refresh = refresh_times[0]
    else:
        gap_med, gap_sd = 4.5, 1.5
        last_refresh = None
    cadence = refresh_info["cadence"]

    # Flat history: the page is stale. Its state was already showing at the
    # first probe, so it is at least as old as the flat span, and the next
    # refresh can jump the counts a lot (the documented first refresh:
    # 638 -> 3868 players while every rate but the 100% one fell). Model
    # that by sampling the state's effective age, which stretches the
    # growth curve, and by widening the spread.
    flat = len(states) < 3
    t_first = states[0].t
    rng = random.Random(SEED)
    n_samples = 320
    n_steps = 64
    days = max(horizon_h, 0.0) / 24.0
    # Process noise on the logit scale, growing with the horizon.
    sigma_proc = 0.05 + 0.05 * math.sqrt(days)
    if flat:
        sigma_proc += 0.25
    samples = [[] for _ in range(n_tr)]
    t_abs_target = t_target_abs

    for _ in range(n_samples):
        # --- when the displayed state was fresh, and N at the target ---
        if flat:
            span = max(t_check - t_first, 0.0)
            age = rng.uniform(span, min(span + 3.5, 12.0))
            t_eff = max(t_check - age, -1.0)
            # Launch-night growth was steep (N 638 -> 3868 across the
            # first refresh, a power-law exponent near 1) and slows with
            # time since release.
            alpha_mean = 0.45 + 0.6 * math.exp(-max(t_check, 0.0) / 24.0)
            alpha = min(max(rng.gauss(alpha_mean, 0.25), 0.5), 1.6)
            log_n_t = log_n_last + alpha * (
                math.log(max(t_abs_target, 0.0) + 1.0)
                - math.log(max(t_eff, 0.0) + 1.0))
            log_n_t += rng.gauss(0.0, 0.25)
        else:
            # The last state became fresh at the last refresh (its
            # midpoint estimate), just before its first read.
            t_eff = (refresh_times[-1] if refresh_times else t_last_state)
            log_n_t = rng.gauss(log_n_target, sigma_log_n)
        n_target = max(math.exp(log_n_t), n_last)
        # --- the answering read shows the page as of the last refresh
        # before the target: project the refresh schedule forward from
        # the last refresh (a roughly periodic clock, with jitter in the
        # gap and in the last refresh's time) and take the last refresh
        # at or before the target (+15 min for the answering read). ---
        gap_s = min(max(rng.gauss(gap_med, gap_sd), 0.5), 24.0)
        base_refresh = last_refresh if last_refresh is not None else t_eff
        base_refresh += rng.gauss(0.0, max(cadence, 0.1) / 2.0)
        cutoff = t_abs_target + 0.25
        if cutoff <= base_refresh:
            t_ans = max(base_refresh, t_eff)
        else:
            k = int(math.floor((cutoff - base_refresh) / gap_s))
            t_ans = base_refresh + k * gap_s
        t_ans = min(max(t_ans, t_eff), cutoff)
        dur_total = max(t_abs_target - t_eff, 0.0)
        frac = (t_ans - t_eff) / dur_total if dur_total > 0 else 0.0
        log_n_ans = log_n_last + (log_n_t - log_n_last) * frac
        n_ans = math.exp(log_n_ans)
        dur = max(t_ans - t_eff, 0.0)
        steps = max(int(n_steps * dur / dur_total), 0) if dur_total > 0 else 0
        dt = dur / steps if steps else 0.0
        # N path: log-linear from the last count to the answering count.
        log_n_path = [log_n_last + (log_n_ans - log_n_last)
                      * (j + 1) / steps for j in range(steps)]
        n_path = [math.exp(v) for v in log_n_path]
        for i in range(n_tr):
            q0, h0, sd_q, sd_h, t_ref = flow[i]
            # Ceiling: earners grow toward reach * N, not toward N -- not
            # everyone finishes the story or grinds, so a trophy's rate
            # saturates below 100%. Sample the reach around its prior.
            rinf0 = reach_prior(trophies[i], last_rates[i])
            if rinf0 >= 0.999:
                rinf = 1.0
            else:
                lr = math.log(rinf0 / (1.0 - rinf0)) + rng.gauss(0.0, 0.25)
                rinf = 1.0 / (1.0 + math.exp(-lr))
                rinf = min(max(rinf, 0.005), 0.995)
            # Lognormal sampling: the sampling sds are often larger than
            # the estimates themselves for rare trophies, and a normal
            # truncated at zero would shift the mean up. Lognormal keeps
            # the median at the estimate.
            q = min(_lognormal_sample(rng, q0, sd_q, 1.0), rinf)
            h = _lognormal_sample(rng, h0, sd_h, 0.2)
            # The fitted h was against (N - E) exposure; against the
            # ceiling exposure (reach*N - E) it rescales by 1/reach.
            h_ceil = h / max(rinf, 0.05)
            tau_i = tau_hours(trophies[i])
            ramp_ref = ramp(mean_age(max(t_ref, 0.0)), tau_i)
            e = last_e[i]
            n_prev = n_last
            for j in range(steps):
                n_now = n_path[j]
                dn = max(n_now - n_prev, 0.0)
                # The hazard ramps up as the population ages into the
                # trophy: scale the fitted hazard (which belongs to the
                # fitting window's mean age) by the ramp ratio at this step.
                if ramp_ref > 0:
                    ratio = ramp(mean_age(t_eff + dt * (j + 0.5)), tau_i) / ramp_ref
                    h_step = h_ceil * min(max(ratio, 0.2), 3.0)
                else:
                    h_step = h_ceil
                e = e + q * dn + h_step * dt * max(rinf * n_prev - e, 0.0)
                n_prev = n_now
            n_end = n_path[-1] if n_path else n_last
            r = 100.0 * e / max(n_end, 1.0) if n_end > 0 else 0.0
            # Process noise on the logit scale.
            r = inv_logit(logit(r) + rng.gauss(0.0, sigma_proc))
            samples[i].append(min(max(r, 0.0), 100.0))

    rows = []
    for i, t in enumerate(trophies):
        s = sorted(samples[i])
        qs = []
        for level in LEVELS:
            pos = level * (len(s) - 1)
            lo = int(math.floor(pos))
            hi = min(lo + 1, len(s) - 1)
            frac = pos - lo
            qs.append(s[lo] * (1.0 - frac) + s[hi] * frac)
        # A trophy everyone holds stays (almost) held: hedge the top end
        # slightly instead of printing a bare 100.
        if last_rates[i] is not None and last_rates[i] >= 99.9:
            qs = [99.5, 99.8, 99.95, 100.0, 100.0]
        else:
            qs = [round4(min(max(v, 0.01), 99.99)) for v in qs]
            for j in range(1, 5):
                if qs[j] < qs[j - 1]:
                    qs[j] = qs[j - 1]
        rows.append({"id": t["id"], "q": qs})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        try:
            forecast = forecast_case(case)
            # Validate the shape before writing anything.
            rows = forecast["trophies"]
            assert len(rows) == len(case["trophies"])
            seen = set()
            for row in rows:
                assert set(row.keys()) == {"id", "q"}
                assert row["id"] not in seen
                seen.add(row["id"])
                q = row["q"]
                assert len(q) == 5
                assert all(isinstance(v, (int, float)) and math.isfinite(v)
                           and 0 <= v <= 100 for v in q)
                assert all(q[j] <= q[j + 1] for j in range(4))
        except Exception:
            # Never leave a case without a valid forecast: the persistence
            # baseline is always well-formed.
            try:
                states, _, _ = distinct_states(case)
                earners = earner_counts(states, len(case["trophies"]))
                _, last_rates = trophy_priors(case, states, earners)
            except Exception:
                last_rates = [0.0] * len(case["trophies"])
            forecast = {"trophies": persistence_rows(case, last_rates)}
        text = json.dumps(forecast, separators=(",", ":")) + "\n"
        assert len(text.encode("utf-8")) <= 65536
        (OUTPUT_DIR / f"{path.stem}.out").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
