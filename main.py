"""GTA VI Twitch concurrent-viewer forecast v3 (gta6-twitch-ccu-forecast-01).

Forecasts hours 2..24 of release day (2026-11-19 UTC) as five quantiles each,
from the case's comparables: past big launches' hourly Twitch curves.

Design
-------
Every comparable launch is reduced to a ratio curve (hourly viewers divided
by its hour-1 print). For each UTC hour a quadratic is fitted in log space:
log-ratio as a function of centred/scaled log(hour-1 viewers), plus two
pre-release covariates when they exist for the target -- x2, the log of the
7-day pre-release peak over the hour-1 print, and x3, the log of the last-24h
pre-release mean over the hour-1 print. The fit is ordinary least squares
through the normal equations; the spread is the fit's residual standard
deviation (floored, smoothed). The median is shifted up by MED_SHIFT and
corrected by HOUR_BIAS, a per-hour vector measured on the four practice
launches as the median per-hour log-miss of this same model (shrunk 0.8
toward its grand mean). The bias is a structural correction, not a fudge
factor: the comparable pool mixes unlock regimes (simultaneous global
unlocks, early-access launches), while the practice targets -- and the
final, GTA VI -- are clean rolling-midnight console launches. The pool's
median shape is systematically too flat for that class: it over-predicts
the deep-night hours 2-4, under-predicts the US-unlock jump at hours 5-6
and the evening peak at hours 17-24. The same pattern has the same sign in
all four practice launches, and the evening part is independently
corroborated by the large rolling comparables' own leave-one-out residuals
(about +0.3 log at hours 17-24).

Practice (pinball loss, log scale, lower is better; house baseline 0.387):
v3 in-sample 0.103, leave-one-case-out 0.122 (v2: 0.109 / 0.131; v1: 0.162 /
0.158). The LOO number is the honest one: v3 wins on three of four held-out
cases and on the mean. One stated risk: the bias was measured at hour-1
prints of 23K-269K and the final will be ~700K; two attempts to damp the
bias by size were built and both made LOO worse, so the full-strength bias
is the best-supported choice -- but if the final's hour-1 comes in very
large, the median may overshoot the morning jump, and the q95 tail (about
2.3x at 700K) is what covers that miss.

Safety: any failure (too few comparables, a bad fit, a missing hour-1
print, unreadable input) falls back to the house baseline formula, so the
program always writes a valid forecast and never crashes. Pure standard
library, deterministic, no network, no randomness.
"""

import json
import math
import os
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# v3 parameters. Practice mean pinball loss 0.103 (in-sample, bias fit on the
# same four cases) and 0.122 leave-one-case-out, vs v2's 0.109 / 0.131.
SPREAD = 0.30         # shrink factor on the residual spread (v2: 0.35)
MED_SHIFT = 0.15      # log-units added to the fitted median curve
UPPER_MULT = 1.25     # extra spread multiplier on the 0.75 / 0.95 quantiles
SIGMA_FLOOR = 0.15    # minimum log-spread per hour
MU_MIN = -1.2         # clamp on fitted log-ratio (ratio >= 0.30)
MU_MAX = 2.6          # clamp on fitted log-ratio (ratio <= 13.5)
SMOOTH_WIN = 3        # moving-average window for mu / sigma across hours

# Per-hour log-bias of the covariate model, measured on the four practice
# launches (median log-miss per hour, shrunk 0.8 toward its grand mean
# 0.060). Index h-1 is hour h; hour 1 is never forecast but kept so the
# indexing matches. Hours 2-4 negative: the pool over-predicts the
# deep-night dip for clean rolling launches. Hours 5-6 positive: the pool
# under-predicts the US-unlock jump. Hours 17-24 positive: the pool
# under-predicts the evening peak. See REPORT.md for the measurement and
# the leave-one-case-out validation.
HOUR_BIAS = [
    0.012,   # h1 (unused)
    -0.171,  # h2
    -0.226,  # h3
    -0.228,  # h4
    0.303,   # h5
    0.317,   # h6
    0.235,   # h7
    0.074,   # h8
    0.048,   # h9
    -0.035,  # h10
    -0.025,  # h11
    -0.143,  # h12
    -0.148,  # h13
    -0.086,  # h14
    0.007,   # h15
    0.021,   # h16
    0.163,   # h17
    0.284,   # h18
    0.205,   # h19
    0.206,   # h20
    0.191,   # h21
    0.180,   # h22
    0.154,   # h23
    0.111,   # h24
]


def _solve_linear(mat, vec):
    """Gaussian elimination with partial pivoting. Returns x or None."""
    n = len(vec)
    aug = [row[:] + [vec[i]] for i, row in enumerate(mat)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[piv][col]) < 1e-12:
            return None
        aug[col], aug[piv] = aug[piv], aug[col]
        for r in range(col + 1, n):
            f = aug[r][col] / aug[col][col]
            for c in range(col, n + 1):
                aug[r][c] -= f * aug[col][c]
    out = [0.0] * n
    for i in range(n - 1, -1, -1):
        out[i] = (aug[i][n] - sum(aug[i][j] * out[j] for j in range(i + 1, n))) / aug[i][i]
    return out


def _wls(rows, targets, weights):
    """Weighted least squares via the normal equations. rows: feature lists."""
    k = len(rows[0])
    ata = [[0.0] * k for _ in range(k)]
    atb = [0.0] * k
    for xi, yi, wi in zip(rows, targets, weights):
        for i in range(k):
            atb[i] += wi * xi[i] * yi
            for j in range(k):
                ata[i][j] += wi * xi[i] * xi[j]
    return _solve_linear(ata, atb)


def _median(vals):
    v = sorted(vals)
    n = len(v)
    if n == 0:
        return 0.0
    return v[n // 2] if n % 2 else 0.5 * (v[n // 2 - 1] + v[n // 2])


def _pstdev(vals, mean):
    if len(vals) < 2:
        return 0.0
    return math.sqrt(sum((v - mean) ** 2 for v in vals) / len(vals))


def _smooth(vals, win):
    if win <= 1:
        return list(vals)
    r = win // 2
    out = []
    for i in range(len(vals)):
        seg = vals[max(0, i - r):min(len(vals), i + r + 1)]
        out.append(sum(seg) / len(seg))
    return out


def _pre_release(hourly, day_start):
    """(log 7-day pre-peak / p1 is computed by the caller; here return the
    raw 7-day pre-peak and the last-24h pre-mean, or (None, None)."""
    import datetime
    start = datetime.datetime.fromisoformat(day_start.replace("Z", "+00:00"))
    peak7, last24 = [], []
    for pt in hourly:
        v = pt.get("viewers")
        if v is None:
            continue
        t = datetime.datetime.fromisoformat(pt["from"].replace("Z", "+00:00"))
        if t < start:
            if (start - t).total_seconds() <= 7 * 86400:
                peak7.append(v)
            if (start - t).total_seconds() <= 24 * 3600:
                last24.append(v)
    if not peak7:
        return None, None
    mean24 = sum(last24) / len(last24) if last24 else None
    return max(peak7), mean24


def _day_curve(comp):
    """Hourly viewers for the comparable's release day, hours 0..23, or None
    if the day has no hour-0 print."""
    day = comp["day_start"][:10]
    by_hour = {}
    for pt in comp["hourly"]:
        if pt["from"][:10] == day:
            by_hour[int(pt["from"][11:13])] = pt["viewers"]
    if 0 not in by_hour or not by_hour[0]:
        return None
    return [by_hour.get(h) for h in range(24)]


def _build_pool(case):
    """Comparables as {p1, curve, x2, x3}. The GTA VI category's own history
    is a comparable in every case but has no launch-day curve, so it is
    skipped by name (its day curve is incomplete)."""
    pool = []
    for comp in case.get("comparables", []):
        if "Grand Theft Auto VI" in str(comp.get("name", "")):
            continue
        curve = _day_curve(comp)
        if curve is None:
            continue
        peak7, mean24 = _pre_release(comp["hourly"], comp["day_start"])
        x2 = math.log(peak7 / curve[0]) if peak7 else None
        x3 = math.log(mean24 / curve[0]) if mean24 else None
        pool.append({"p1": curve[0], "curve": curve, "x2": x2, "x3": x3})
    return pool


def _fit_hour(pool, hour, log_p1_eval, x2_eval, x3_eval, use_cov):
    """Fit one hour's log-ratio model. Returns (mu, sigma). Falls back to
    the no-covariate fit when covariates are too sparse, then to the
    pool median, then to 0.0."""
    log_p1s = [math.log(p["p1"]) for p in pool]
    xbar = sum(log_p1s) / len(log_p1s)
    xsd = _pstdev(log_p1s, xbar) or 1.0

    def _fit(cov_keys, ev):
        pts = []
        for p in pool:
            v = p["curve"][hour]
            if v is None or v <= 0:
                continue
            if any(p[k] is None for k in cov_keys):
                continue
            pts.append(p)
        if len(pts) < 6:
            return None
        rows, ys = [], []
        for p in pts:
            u = (math.log(p["p1"]) - xbar) / xsd
            rows.append([1.0, u, u * u] + [p[k] for k in cov_keys])
            ys.append(math.log(p["curve"][hour] / p["p1"]))
        beta = _wls(rows, ys, [1.0] * len(rows))
        if beta is None:
            return None
        ue = (log_p1_eval - xbar) / xsd
        row_e = [1.0, ue, ue * ue] + list(ev)
        mu = sum(b * v for b, v in zip(beta, row_e))
        resid = []
        for xi, yi in zip(rows, ys):
            resid.append(yi - sum(b * v for b, v in zip(beta, xi)))
        sig = max(_pstdev(resid, sum(resid) / len(resid)), SIGMA_FLOOR)
        return mu, sig

    if use_cov:
        got = _fit(["x2", "x3"], (x2_eval, x3_eval))
        if got is not None:
            return got
    got = _fit([], ())
    if got is not None:
        return got
    vals = [math.log(p["curve"][hour] / p["p1"]) for p in pool
            if p["curve"][hour] is not None and p["curve"][hour] > 0]
    if vals:
        m = _median(vals)
        return m, max(_pstdev(vals, m), SIGMA_FLOOR)
    return 0.0, 0.5


def _baseline_rows(p1):
    """House baseline formula: flat at the hour-1 print, spread widening
    over the day. Used whenever the model cannot run."""
    rows = []
    for h in range(2, 25):
        s = 0.15 * math.sqrt(h - 1)
        q = sorted(max(1.0, p1 * math.exp(z * s)) for z in Z)
        rows.append({"hour": h, "q": [round(v, 2) for v in q]})
    return {"hours": rows}


def _forecast(case):
    known = case.get("known", [])
    p1 = None
    for k in known:
        if k.get("hour") == 1:
            p1 = k.get("viewers")
    if not p1 or p1 <= 0:
        return None
    p1 = max(1, p1)
    pool = _build_pool(case)
    if len(pool) < 6:
        return _baseline_rows(p1)

    peak7, mean24 = _pre_release(case.get("history_hourly", []), case["day_start"])
    use_cov = peak7 is not None and mean24 is not None
    x2_eval = math.log(peak7 / p1) if use_cov else None
    x3_eval = math.log(mean24 / p1) if use_cov else None
    log_p1 = math.log(p1)

    mu, sig = [], []
    for h in range(24):
        m, s = _fit_hour(pool, h, log_p1, x2_eval, x3_eval, use_cov)
        mu.append(m)
        sig.append(s)
    mu = _smooth(mu, SMOOTH_WIN)
    sig = _smooth(sig, SMOOTH_WIN)

    rows = []
    for h in range(2, 25):
        m = min(max(mu[h - 1] + MED_SHIFT + HOUR_BIAS[h - 1], MU_MIN), MU_MAX)
        s = sig[h - 1] * SPREAD
        q = []
        for j, z in enumerate(Z):
            mult = UPPER_MULT if j >= 3 else 1.0
            v = p1 * math.exp(m + z * s * mult)
            q.append(min(max(v, 1.0), 1e9))
        q = sorted(round(v, 2) for v in q)
        for j in range(1, 5):
            if q[j] < q[j - 1]:
                q[j] = q[j - 1]
        rows.append({"hour": h, "q": q})
    return {"hours": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        try:
            case = json.loads(path.read_text())
            out = _forecast(case)
            if out is None:
                raise ValueError("no hour-1 print")
        except Exception:
            # Never crash and never leave a case without output: fall back
            # to a flat valid forecast. If even the hour-1 print cannot be
            # read, use 1.0 so the output stays well-formed.
            try:
                case = json.loads(path.read_text())
                p1 = max(1, next(k["viewers"] for k in case.get("known", [])
                                 if k.get("hour") == 1))
            except Exception:
                p1 = 1
            out = _baseline_rows(p1)
        (OUTPUT_DIR / (path.stem + ".out")).write_text(json.dumps(out))


if __name__ == "__main__":
    main()
