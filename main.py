"""GTA VI launch-day Twitch CCU forecaster (v7).

Given one case file (schema makefun/twitch-ccu-case/v1) this program
writes the forecast for hours 2..24 as five quantiles per hour.

How it works (v7 = v6 F2b, ridge 0.5 -> 1.6, upper width 1.25 -> 1.20):
for every hour of the launch day, a ridge regression predicts
log(viewers / hour-1) from a quadratic in standardised log hour-1 plus
five pre-release covariates [x2, x3, x4, x10, x11], fitted on the past
big launches listed in the case file itself. Evening hours (17-24) are
fit without shrinkage. The fitted median is shifted +0.15 in log space
and dressed with normal quantiles whose width follows the per-hour
residual scale. A fixed practice correction (the median log-miss of
this model on the four published practice launches) is then applied:
full strength at hours 5-6 with size damping for big targets, and a
blend of the practice correction with the pool's own rolling-miss
elsewhere (pure practice correction for small targets). If anything
goes wrong the house baseline is emitted instead, so the output is
always a valid 23-row forecast.

Standard library only. No clock, no randomness: the same input always
gives the same bytes. About half a second per case.

Usage: python3 main.py <case.in> <case.out>
   or: python3 main.py            (reads /input/*.in, writes /output/*.out)
"""
import datetime
import json
import math
import os
import sys

QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]
Z_SCORES = [-1.645, -0.674, 0.0, 0.674, 1.645]

RIDGE = 1.6              # shrinkage on the quadratic + covariate terms
RIDGE_EVENING = 0.0      # shrinkage for hours 17-24 (fit unshrunk)
EVENING_FROM = 16        # 0-based hour index where the evening starts
WIDTH = 0.32             # quantile half-width multiplier on residual scale
UPPER_WIDTH = 1.20       # extra width on the two upper quantiles
MEDIAN_SHIFT = 0.15      # constant log-shift on the fitted median
PRACTICE_WEIGHT = 0.8    # practice-correction weight outside hours 5-6
DAMP_STRENGTH = 2.0      # size damping for the hours 5-6 correction
DAMP_REFERENCE = 120000.0
SMALL_TARGET = 50000.0   # below this hour-1 the correction is used pure
FEATURES = ["x2", "x3", "x4", "x10", "x11"]
EXCLUDED = ["ELDEN RING", "Cyberpunk 2077", "Monster Hunter Wilds"]

# Median per-hour log-miss (log actual - log q50) of this model on the
# four published practice launches, smoothed and shrunk 0.8 toward its
# grand mean. Entries are hours 1..24.
PRACTICE_BIAS = [0.012, -0.176, -0.236, -0.253, 0.284, 0.32, 0.215, 0.098,
                 0.025, -0.059, -0.054, -0.178, -0.184, -0.122, -0.026,
                 -0.016, 0.128, 0.27, 0.228, 0.239, 0.264, 0.272, 0.214, 0.205]


# --------------------------------------------------------------------- small
def mid(values):
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        return 0.0
    if n % 2:
        return ordered[n // 2]
    return 0.5 * (ordered[n // 2 - 1] + ordered[n // 2])


def sd(values, mean):
    if len(values) < 2:
        return 0.0
    return math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))


def moving_avg(values, win):
    if win <= 1:
        return list(values)
    half = win // 2
    smoothed = []
    for i in range(len(values)):
        lo = max(0, i - half)
        hi = min(len(values), i + half + 1)
        smoothed.append(sum(values[lo:hi]) / (hi - lo))
    return smoothed


def gauss_solve(matrix, rhs):
    """Solve a small dense linear system by Gaussian elimination."""
    n = len(rhs)
    aug = [row[:] + [rhs[i]] for i, row in enumerate(matrix)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot][col]) < 1e-12:
            return None
        aug[col], aug[pivot] = aug[pivot], aug[col]
        for row in range(col + 1, n):
            factor = aug[row][col] / aug[col][col]
            for c in range(col, n + 1):
                aug[row][c] -= factor * aug[col][c]
    sol = [0.0] * n
    for i in range(n - 1, -1, -1):
        sol[i] = (aug[i][n] - sum(aug[i][j] * sol[j] for j in range(i + 1, n))) / aug[i][i]
    return sol


def ridge_fit(design, targets, weights, penalties):
    """Weighted ridge regression; returns the coefficient vector or None."""
    width = len(design[0])
    normal = [[0.0] * width for _ in range(width)]
    moment = [0.0] * width
    for row, y, w in zip(design, targets, weights):
        for i in range(width):
            moment[i] += w * row[i] * y
            for j in range(width):
                normal[i][j] += w * row[i] * row[j]
    if penalties:
        for i in range(width):
            normal[i][i] += penalties[i]
    return gauss_solve(normal, moment)


# --------------------------------------------------------------------- input
def launch_day(comp):
    """The 24 hourly viewer counts of a comparable's launch day."""
    day = comp["day_start"][:10]
    hours = {}
    for point in comp["hourly"]:
        if point["from"][:10] == day:
            hours[int(point["from"][11:13])] = point["viewers"]
    if 0 not in hours or not hours[0]:
        return None
    return [hours.get(h) for h in range(24)]


def history_stats(hourly, day_start):
    """Pre-release shape: 7-day peak/mean/slope/vol, 3-day slope/vol."""
    start = datetime.datetime.fromisoformat(day_start.replace("Z", "+00:00"))
    week, three = [], []
    for point in hourly:
        viewers = point.get("viewers")
        if viewers is None or viewers <= 0:
            continue
        when = datetime.datetime.fromisoformat(point["from"].replace("Z", "+00:00"))
        if when < start:
            age = (start - when).total_seconds()
            if age <= 7 * 86400:
                week.append((age / 86400.0, math.log(viewers), viewers))
            if age <= 3 * 86400:
                three.append((age / 86400.0, math.log(viewers), viewers))
    if not week:
        return None, None, None, None, None, None

    def trend(points):
        n = len(points)
        if n < 2:
            return 0.0, 0.0
        mx = sum(d for d, _, _ in points) / n
        my = sum(y for _, y, _ in points) / n
        sxx = sum((d - mx) ** 2 for d, _, _ in points)
        sxy = sum((d - mx) * (y - my) for d, y, _ in points)
        slope = sxy / sxx if sxx > 1e-9 else 0.0
        spread = math.sqrt(sum((y - my) ** 2 for _, y, _ in points) / n)
        return slope, spread

    peak = max(v for _, _, v in week)
    recent = [v for d, _, v in week if d <= 1.0]
    mean = sum(recent) / len(recent) if recent else None
    slope7, vol7 = trend(week)
    slope3, vol3 = trend(three)
    return peak, mean, slope7, vol7, slope3, vol3


def unlock_kind(text):
    lowered = str(text).lower()
    if "rolling" in lowered or "midnight local" in lowered:
        return "rolling"
    if "single global" in lowered or "simultaneous global" in lowered or "simultaneous across" in lowered:
        return "simultaneous"
    return "unknown"


def collect_launches(case):
    """Past launches from the case file's comparables (the GTA VI
    category itself is not a launch: its premiere day has no hour 0)."""
    launches = []
    for comp in case["comparables"]:
        name = str(comp.get("name", ""))
        if "Grand Theft Auto VI" in name:
            continue
        curve = launch_day(comp)
        if curve is None:
            continue
        peak, mean, slope7, vol7, _slope3, _vol3 = history_stats(comp["hourly"], comp["day_start"])
        hour1 = curve[0]
        pre_peak = math.log(peak / hour1) if peak else None
        pre_mean = math.log(mean / hour1) if mean else None
        interaction = (slope7 * vol7) if (slope7 is not None and vol7 is not None) else None
        launches.append({
            "name": name, "p1": hour1, "curve": curve,
            "regime": unlock_kind(comp.get("unlock", "")),
            "x2": pre_peak, "x3": pre_mean, "x4": slope7, "x10": interaction,
            "x11": (pre_peak * pre_peak) if pre_peak is not None else None,
            "x8": 0.0 if "PC" in str(comp.get("platforms", "")) else 1.0,
        })
    return launches


def target_stats(case):
    """The target's hour-1 viewers and pre-release covariates."""
    peak, mean, slope7, vol7, _slope3, _vol3 = history_stats(case["history_hourly"], case["day_start"])
    hour1 = max(1, next(k["viewers"] for k in case["known"] if k["hour"] == 1))
    pre_peak = math.log(peak / hour1) if peak else None
    pre_mean = math.log(mean / hour1) if mean else None
    interaction = (slope7 * vol7) if (slope7 is not None and vol7 is not None) else None
    return {"p1": hour1, "x2": pre_peak, "x3": pre_mean, "x4": slope7,
            "x10": interaction,
            "x11": (pre_peak * pre_peak) if pre_peak is not None else None}


# --------------------------------------------------------------------- model
def hourly_fit(launches, log_p1, covariates):
    """Fit log(curve[h]/p1) for each hour; returns (median[24], scale[24]).

    A missing target covariate is filled with the pool mean, never 0.
    """
    log_sizes = [math.log(p["p1"]) for p in launches]
    centre = sum(log_sizes) / len(log_sizes)
    scale = sd(log_sizes, centre) or 1.0
    pool_avg = {}
    for key in FEATURES:
        present = [p[key] for p in launches if p[key] is not None]
        pool_avg[key] = sum(present) / len(present) if present else 0.0
    filled = [covariates[i] if covariates[i] is not None else pool_avg[FEATURES[i]]
              for i in range(len(FEATURES))]
    u_eval = (log_p1 - centre) / scale
    medians, scales = [], []
    for h in range(24):
        usable = [(p, math.log(p["curve"][h] / p["p1"])) for p in launches
                  if p["curve"][h] is not None and p["curve"][h] > 0
                  and all(p[key] is not None for key in FEATURES)]
        if len(usable) < 6:
            medians.append(0.0)
            scales.append(0.5)
            continue
        design, targets = [], []
        for launch, log_ratio in usable:
            u = (math.log(launch["p1"]) - centre) / scale
            design.append([1.0, u, u * u] + [launch[key] for key in FEATURES])
            targets.append(log_ratio)
        penalty = RIDGE if h < EVENING_FROM else RIDGE_EVENING
        coeffs = ridge_fit(design, targets, [1.0] * len(design),
                           [0.0, penalty, penalty] + [penalty] * len(FEATURES))
        if coeffs is None:
            middle = mid(targets)
            medians.append(middle)
            scales.append(max(sd(targets, middle), 0.15))
            continue
        probe = [1.0, u_eval, u_eval * u_eval] + filled
        medians.append(sum(b * v for b, v in zip(coeffs, probe)))
        errors = [y - sum(b * v for b, v in zip(coeffs, row))
                  for row, y in zip(design, targets)]
        err_mean = sum(errors) / len(errors)
        variance = sum((e - err_mean) ** 2 for e in errors) / len(errors)
        scales.append(max(math.sqrt(variance), 0.15))
    return moving_avg(medians, 3), moving_avg(scales, 3)


def quantile_table(p1, medians, scales):
    """Turn a fitted (median, scale) pair into the 23 output rows."""
    rows = []
    for h in range(2, 25):
        quants = []
        for j, z in enumerate(Z_SCORES):
            offset = z * scales[h - 1] * WIDTH * (UPPER_WIDTH if j >= 3 else 1.0)
            value = p1 * math.exp(min(max(medians[h - 1] + offset, -1.2), 2.6))
            quants.append(min(max(value, 1.0), 1e9))
        quants = sorted(round(v, 2) if v != int(v) else v for v in quants)
        for j in range(1, 5):
            if quants[j] < quants[j - 1]:
                quants[j] = quants[j - 1]
        rows.append({"hour": h, "q": quants})
    return {"hours": rows}


def leave_one_miss(launches):
    """Median per-hour log-miss of the base model on the pool's rolling
    launches, each predicted from all the others (mixed excluded)."""
    rolling = [p for p in launches if p["regime"] == "rolling" and p["name"] not in EXCLUDED]
    if len(rolling) < 3:
        return None
    per_hour = [[] for _ in range(24)]
    for held in rolling:
        rest = [p for p in launches if p["name"] != held["name"]]
        if len(rest) < 6:
            continue
        medians, _scales = hourly_fit(rest, math.log(held["p1"]), [held[key] for key in FEATURES])
        medians = [m + MEDIAN_SHIFT for m in medians]
        for hh in range(2, 25):
            actual = held["curve"][hh - 1]
            if actual is None or actual <= 0:
                continue
            guess = held["p1"] * math.exp(min(max(medians[hh - 1], -1.2), 2.6))
            per_hour[hh - 1].append(math.log(max(1, actual)) - math.log(max(1e-9, guess)))
    return [mid(v) if v else 0.0 for v in per_hour]


def nudge_rows(table, p1, miss):
    """Apply the practice correction to finished rows (log-multiplicative).

    Hours 5-6 take the full correction, damped for large targets; other
    hours blend it with the pooled rolling miss (pure correction for
    small targets)."""
    shifted = []
    for row in table["hours"]:
        h = row["hour"]
        if h in (5, 6):
            amount = PRACTICE_BIAS[h - 1]
            amount = amount / (1.0 + DAMP_STRENGTH * max(0.0, math.log(p1 / DAMP_REFERENCE)))
        elif p1 < SMALL_TARGET:
            amount = PRACTICE_BIAS[h - 1]
        else:
            amount = PRACTICE_WEIGHT * PRACTICE_BIAS[h - 1] + (1.0 - PRACTICE_WEIGHT) * miss[h - 1]
        factor = math.exp(amount)
        shifted.append({"hour": h, "q": [max(1.0, min(1e9, v * factor)) for v in row["q"]]})
    return {"hours": shifted}


def baseline_table(p1):
    """House baseline: flat at hour 1, spread widening with sqrt(h-1)."""
    rows = []
    for h in range(2, 25):
        rows.append({"hour": h, "q": [max(1.0, min(1e9, p1 * math.exp(z * 0.15 * math.sqrt(h - 1)))) for z in Z_SCORES]})
    return {"hours": rows}


def predict_case(case):
    """The whole forecast for one parsed case file."""
    launches = collect_launches(case)
    if len(launches) < 6:
        raise ValueError("too few comparables")
    target = target_stats(case)
    p1 = target["p1"]
    medians, scales = hourly_fit(launches, math.log(p1), [target[key] for key in FEATURES])
    medians = [m + MEDIAN_SHIFT for m in medians]
    table = quantile_table(p1, medians, scales)
    miss = leave_one_miss(launches)
    if miss is None:
        miss = [0.0] * 24
    return nudge_rows(table, p1, miss)


# --------------------------------------------------------------------- driver
def process_file(in_path, out_path):
    with open(in_path) as handle:
        case = json.load(handle)
    try:
        result = predict_case(case)
    except Exception:
        hour1 = 1
        try:
            hour1 = max(1, next(k["viewers"] for k in case["known"] if k["hour"] == 1))
        except Exception:
            pass
        result = baseline_table(hour1)
    try:
        assert sorted(r["hour"] for r in result["hours"]) == list(range(2, 25))
        for row in result["hours"]:
            assert len(row["q"]) == 5
            assert all(0 < v <= 1e9 and math.isfinite(v) for v in row["q"])
            assert all(row["q"][j] <= row["q"][j + 1] for j in range(4))
    except Exception:
        hour1 = max(1, next(k["viewers"] for k in case["known"] if k["hour"] == 1))
        result = baseline_table(hour1)
    with open(out_path, "w") as handle:
        json.dump(result, handle)


def main(argv):
    if len(argv) == 3:
        process_file(argv[1], argv[2])
        return 0
    if len(argv) == 1:
        os.makedirs("/output", exist_ok=True)
        for name in sorted(os.listdir("/input")):
            if name.endswith(".in"):
                process_file(os.path.join("/input", name),
                             os.path.join("/output", name[:-3] + ".out"))
        return 0
    sys.stderr.write("usage: main.py [case.in case.out]\n")
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
