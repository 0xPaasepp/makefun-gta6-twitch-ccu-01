"""GTA VI Twitch concurrent-viewer forecast v2 (gta6-twitch-ccu-forecast-01).

Forecasts hours 2..24 of release day (2026-11-19 UTC) as five quantiles each,
from the case's comparables: past big launches' hourly Twitch curves.

Design
-------
Same ratio-curve base as v1: every comparable is reduced to r_h =
viewers_h / viewers_hour1, and for each UTC hour h a quadratic in
log(hour-1 viewers) is fit over the comparables and evaluated at the case's
own hour-1 print (clamped to the pool range, never extrapolated). The spread
is the residual standard deviation of that fit (floored, smoothed), shrunk
by SPREAD; the median is shifted up by MED_SHIFT; the two upper quantiles
get a 1.25x wider spread.

v2 adds one correction on top of that base: a per-hour bias vector,
HOUR_BIAS. It was measured on the four practice launches (Wolverine, RDR2,
GoW Ragnarok, Spider-Man 2): for each practice case, the v1 base was run and
its per-hour log-miss (log actual - log median forecast) recorded; HOUR_BIAS
is the mean of those four miss vectors, smoothed across neighbouring hours.
It is a structural correction, not a fudge factor: the comparable pool is a
mix of unlock regimes (simultaneous global unlocks, early-access launches,
unverified), while the practice targets — and the final, GTA VI — are clean
rolling-midnight console launches. The pool's median shape is systematically
too flat for that class: it over-predicts the deep-night hours 2-4 (bias
about -0.3 log), under-predicts the US-unlock jump at hours 5-6 (+0.2 to
+0.4) and the evening peak at hours 17-24 (+0.2 to +0.4). The same pattern
appears in all four practice cases with the same sign at those hours, and
leave-one-case-out validation (bias estimated from three cases, tested on
the fourth) improves the mean pinball loss from 0.158 (v1) to 0.131.

The bias is applied at full strength with no size scaling: an attempt to
scale it by hour-1 size (joint fit on practice targets + rolling
comparables) was tested and REJECTED — it scored worse under
leave-one-case-out (0.144-0.156) than the size-blind bias, because the size
slope is not reliably estimable from this little data. The evening part of
the bias is independently corroborated by the large rolling comparables
themselves (their leave-one-out residuals at hours 17-24 average about
+0.3 log), so the full-strength correction is the honest choice even for a
~700K hour-1 print.

Safety: any failure (too few comparables, a bad fit, a missing hour-1 print)
falls back to the house baseline formula, so the program always writes a
valid forecast and never crashes. Pure standard library, deterministic, no
network, no randomness.
"""

import json
import math
import os
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# v2 parameters. Practice mean pinball loss 0.109 (in-sample, bias fit on the
# same four cases) and 0.131 leave-one-case-out, vs v1's 0.162 / 0.158.
# House baseline is 0.387 on the same cases.
SPREAD = 0.35        # shrink factor on the residual spread (v1: 0.5)
MED_SHIFT = 0.15     # log-units added to the fitted median curve (as v1)
UPPER_MULT = 1.25    # extra spread multiplier on the 0.75 / 0.95 quantiles
SIGMA_FLOOR = 0.15   # minimum log-spread per hour
MU_MIN = -1.2        # clamp on fitted log-ratio (ratio >= 0.30)
MU_MAX = 2.6         # clamp on fitted log-ratio (ratio <= 13.5)
SMOOTH_WIN = 3       # moving-average window for mu / sigma across hours

# Per-hour log-bias of the v1 base, measured on the four practice launches
# (mean log-miss per hour, smoothed with a window of 3). Index h-1 is hour h;
# hour 1 is never forecast but kept so the smoothing matches the measurement.
# Hours 2-4 negative: the pool over-predicts the deep-night dip for clean
# rolling launches. Hours 5-6 positive: the pool under-predicts the
# US-unlock jump. Hours 17-24 positive: the pool under-predicts the evening
# peak. See REPORT.md for the measurement and the validation.
HOUR_BIAS = [
    -0.146,  # h1 (unused)
    -0.231,  # h2
    -0.370,  # h3
    -0.201,  # h4
    0.053,   # h5
    0.274,   # h6
    0.241,   # h7
    0.135,   # h8
    0.040,   # h9
    -0.022,  # h10
    -0.125,  # h11
    -0.163,  # h12
    -0.149,  # h13
    -0.030,  # h14
    0.067,   # h15
    0.160,   # h16
    0.252,   # h17
    0.295,   # h18
    0.289,   # h19
    0.245,   # h20
    0.229,   # h21
    0.208,   # h22
    0.166,   # h23
    0.140,   # h24
]


def round2(x):
    """Round to 2 decimals, half up, as JavaScript's Math.round(x * 100) / 100."""
    scaled = x * 100.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    v = whole / 100.0
    return int(v) if v == int(v) else v


def hour1_print(case):
    """Hour-1 print: `known` hour 1, else the hour-1 whole-hour mean, else the
    mean of the minute reads inside hour 1, else the last history point."""
    known = case.get("known") or []
    for row in known:
        if row.get("hour") == 1 and row.get("viewers") is not None:
            return max(1.0, float(row["viewers"]))
    day_start = str(case.get("day_start", ""))[:19]
    for row in case.get("history_hourly") or []:
        if str(row.get("from", ""))[:19] == day_start and row.get("viewers") is not None:
            return max(1.0, float(row["viewers"]))
    checkpoint = str(case.get("checkpoint", ""))
    vals = []
    for pt in case.get("history") or []:
        at = str(pt.get("at", ""))
        if day_start <= at < checkpoint and pt.get("viewers") is not None:
            vals.append(float(pt["viewers"]))
    if vals:
        return max(1.0, sum(vals) / len(vals))
    hist = case.get("history") or []
    if hist and hist[-1].get("viewers") is not None:
        return max(1.0, float(hist[-1]["viewers"]))
    return 1.0


def forecast_hours(case):
    """Hours to forecast, from the case's `forecast` list, else 2..24."""
    hours = []
    for row in case.get("forecast") or []:
        h = row.get("hour")
        if isinstance(h, int) and 2 <= h <= 24:
            hours.append(h)
    return sorted(set(hours)) or list(range(2, 25))


def day_curve(comp):
    """The comparable's 24 hourly viewer counts on its own release day, or
    None when its hour-1 print is missing or non-positive."""
    day = str(comp.get("day_start", ""))[:10]
    by_hour = {}
    for pt in comp.get("hourly") or []:
        if str(pt.get("from", ""))[:10] == day:
            try:
                h = int(str(pt["from"])[11:13])
            except (ValueError, IndexError):
                continue
            v = pt.get("viewers")
            if v is not None:
                by_hour[h] = float(v)
    if 0 not in by_hour or by_hour[0] <= 0:
        return None
    return [by_hour.get(h) for h in range(24)]


def solve3(mat, vec):
    """Solve a 3x3 linear system by Gaussian elimination with pivoting."""
    a = [row[:] for row in mat]
    b = list(vec)
    for i in range(3):
        piv = max(range(i, 3), key=lambda r: abs(a[r][i]))
        if abs(a[piv][i]) < 1e-12:
            return None
        a[i], a[piv] = a[piv], a[i]
        b[i], b[piv] = b[piv], b[i]
        for r in range(i + 1, 3):
            f = a[r][i] / a[i][i]
            for c in range(i, 3):
                a[r][c] -= f * a[i][c]
            b[r] -= f * b[i]
    x = [0.0, 0.0, 0.0]
    for i in (2, 1, 0):
        x[i] = (b[i] - sum(a[i][j] * x[j] for j in range(i + 1, 3))) / a[i][i]
    return x


def quad_fit(xs, ys):
    """Least squares for y = a + b*x + c*x^2. Returns (a, b, c) or None."""
    s = [0.0] * 5
    t = [0.0] * 3
    for x, y in zip(xs, ys):
        p = 1.0
        for k in range(5):
            s[k] += p
            if k < 3:
                t[k] += p * y
            p *= x
    mat = [[s[0], s[1], s[2]], [s[1], s[2], s[3]], [s[2], s[3], s[4]]]
    return solve3(mat, t)


def median(vals):
    v = sorted(vals)
    n = len(v)
    if n == 0:
        return 0.0
    return v[n // 2] if n % 2 else 0.5 * (v[n // 2 - 1] + v[n // 2])


def pstdev(vals, mean):
    if len(vals) < 2:
        return 0.0
    return math.sqrt(sum((v - mean) ** 2 for v in vals) / len(vals))


def smooth(vals, win):
    r = win // 2
    out = []
    for i in range(len(vals)):
        seg = vals[max(0, i - r):min(len(vals), i + r + 1)]
        out.append(sum(seg) / len(seg))
    return out


def baseline_rows(p1, hours):
    """House baseline: flat at hour-1 with a widening random-walk spread."""
    rows = []
    for h in hours:
        q = sorted(max(1.0, p1 * math.exp(z * 0.15 * math.sqrt(h - 1))) for z in Z)
        rows.append({"hour": h, "q": [round2(v) for v in q]})
    return rows


def forecast(case):
    p1 = hour1_print(case)
    hours = forecast_hours(case)
    log_p1 = math.log(p1)

    pool = []
    for comp in sorted(case.get("comparables") or [], key=lambda c: str(c.get("name", ""))):
        # The target's own pre-release history is not a launch curve.
        if "Grand Theft Auto VI" in str(comp.get("name", "")):
            continue
        curve = day_curve(comp)
        if curve is None:
            continue
        pool.append((curve[0], curve))
    if len(pool) < 6:
        return {"hours": baseline_rows(p1, hours)}

    # Never extrapolate the quadratic past the hour-1 sizes seen in the pool.
    xs_all = [math.log(c0) for c0, _ in pool]
    x_lo, x_hi = min(xs_all), max(xs_all)
    x_eval = min(max(log_p1, x_lo), x_hi)

    mu, sig = [], []
    for h in range(24):
        pts = []
        for c0, curve in pool:
            v = curve[h]
            if v is not None and v > 0:
                pts.append((math.log(c0), math.log(v / c0)))
        if len(pts) < 6:
            mu.append(0.0)
            sig.append(0.5)
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        fit = quad_fit(xs, ys)
        if fit is None:
            m0 = median(ys)
            mu.append(m0)
            sig.append(max(pstdev(ys, m0), SIGMA_FLOOR))
            continue
        a, b, c = fit
        mu.append(a + b * x_eval + c * x_eval * x_eval)
        resid = [y - (a + b * x + c * x * x) for x, y in pts]
        rmean = sum(resid) / len(resid)
        sig.append(max(pstdev(resid, rmean), SIGMA_FLOOR))
    mu = smooth(mu, SMOOTH_WIN)
    sig = smooth(sig, SMOOTH_WIN)

    rows = []
    for h in hours:
        # v1 median shift plus the per-hour bias correction (v2).
        m = min(max(mu[h - 1] + MED_SHIFT + HOUR_BIAS[h - 1], MU_MIN), MU_MAX)
        s = sig[h - 1] * SPREAD
        q = []
        for j, z in enumerate(Z):
            sj = s * (UPPER_MULT if j >= 3 else 1.0)
            v = p1 * math.exp(m + z * sj)
            q.append(min(max(v, 1.0), 1e9))
        q = sorted(round2(v) for v in q)
        rows.append({"hour": h, "q": q})
    return {"hours": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        try:
            case = json.loads(path.read_text(encoding="utf-8"))
            out = forecast(case)
        except Exception:
            try:
                out = {"hours": baseline_rows(hour1_print(case), forecast_hours(case))}
            except Exception:
                out = {"hours": baseline_rows(1.0, list(range(2, 25)))}
        (OUTPUT_DIR / (path.stem + ".out")).write_text(
            json.dumps(out, separators=(",", ":")), encoding="utf-8")


if __name__ == "__main__":
    main()
