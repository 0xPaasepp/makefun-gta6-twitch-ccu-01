#!/usr/bin/env python3
"""Hourly Twitch viewership forecast for a game launch (hours 2-24)."""

import datetime
import glob
import json
import math
import os
import sys

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]
FEATS = ["x2", "x3", "x4", "x10", "x11"]
LAM = 0.5
LAM_EVENING = 0.0
EVENING_FROM = 16
SHIFT = 0.15
SPREAD = 0.32
UPPER = 1.25
BLEND = 0.8
SMALL_P1 = 50000.0
DAMP_K = 2.0
DAMP_REF = 120000.0
FULL_HOURS = (5, 6)
MIXED = ("ELDEN RING", "Cyberpunk 2077", "Monster Hunter Wilds")
PRACTICE_BIAS = [
    0.012, -0.176, -0.229, -0.244, 0.279, 0.323, 0.203, 0.094,
    0.02, -0.055, -0.072, -0.182, -0.19, -0.129, -0.034, -0.011,
    0.124, 0.27, 0.228, 0.239, 0.264, 0.271, 0.213, 0.205,
]


def run():
    src = find_case()
    if len(sys.argv) > 2:
        dst = sys.argv[2]
    else:
        stem = os.path.splitext(os.path.basename(src))[0]
        dst = os.path.join("/output", stem + ".out")
        if not os.path.isdir("/output"):
            dst = stem + ".out"
    try:
        with open(src) as fh:
            case = json.load(fh)
        p1 = hour1(case)
        result = predict(case, p1)
    except Exception:
        try:
            with open(src) as fh:
                case = json.load(fh)
            p1 = hour1(case)
        except Exception:
            p1 = 1000.0
        result = flat_forecast(p1)
    check(result)
    folder = os.path.dirname(dst)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(dst, "w") as fh:
        json.dump(result, fh, separators=(",", ":"))
        fh.write("\n")


def find_case():
    if len(sys.argv) > 1:
        return sys.argv[1]
    for pattern in ("/input/*.in", "*.in", "input/*.in"):
        found = sorted(glob.glob(pattern))
        if found:
            return found[0]
    raise FileNotFoundError("no case file found")


def hour1(case):
    for k in case.get("known") or []:
        if k.get("hour") == 1 and k.get("viewers"):
            return max(1, k["viewers"])
    hist = case.get("history_hourly") or []
    if hist and hist[0].get("viewers"):
        return max(1, hist[0]["viewers"])
    raise ValueError("no hour-1 viewers in case")


def predict(case, p1):
    pool = build_pool(case)
    if len(pool) < 6:
        raise ValueError("comparable pool too small")
    feats = target_feats(case, p1)
    means = pool_means(pool)
    cov = [feats[k] if feats[k] is not None else means[k] for k in FEATS]
    base = base_rows(pool, p1, cov)
    return apply_bias(base, bias_for(pool, p1))


def flat_forecast(p1):
    rows = []
    for h in range(2, 25):
        q = []
        for z in Z:
            v = p1 * math.exp(z * 0.15 * math.sqrt(h - 1))
            q.append(min(max(round(v, 2), 1.0), 1e9))
        rows.append({"hour": h, "q": sorted(q)})
    return {"hours": rows}


def check(result):
    hours = result["hours"]
    if len(hours) != 23:
        raise ValueError("must be 23 rows")
    for i, r in enumerate(hours):
        if r["hour"] != i + 2:
            raise ValueError("hours must be 2..24")
        q = r["q"]
        if len(q) != 5:
            raise ValueError("five quantiles per hour")
        for j, v in enumerate(q):
            if not math.isfinite(v) or v < 1.0 or v > 1e9:
                raise ValueError("quantile out of range")
            if j and v < q[j - 1]:
                raise ValueError("quantiles must be non-decreasing")


def to_time(text):
    return datetime.datetime.fromisoformat(str(text).replace("Z", "+00:00"))


def regime_of(unlock_text):
    u = str(unlock_text).lower()
    if "rolling" in u or "midnight local" in u:
        return "rolling"
    if "single global" in u or "simultaneous global" in u or "simultaneous across" in u:
        return "simultaneous"
    return "unknown"


def curve_of(comp):
    day = str(comp.get("day_start", ""))[:10]
    by_hour = {}
    for pt in comp.get("hourly") or []:
        if str(pt.get("from", ""))[:10] == day:
            try:
                by_hour[int(str(pt["from"])[11:13])] = pt.get("viewers")
            except (ValueError, IndexError):
                continue
    if not by_hour.get(0):
        return None
    return [by_hour.get(h) for h in range(24)]


def features_of(hourly, day_start, p1):
    ds = to_time(day_start)
    pts = []
    for pt in hourly or []:
        v = pt.get("viewers")
        if v is None or v <= 0:
            continue
        try:
            t = to_time(pt["from"])
        except ValueError:
            continue
        if t < ds:
            dt = (ds - t).total_seconds() / 86400.0
            if dt <= 7.0:
                pts.append((dt, math.log(v), v))
    if not pts:
        return {"x2": None, "x3": None, "x4": None, "x5": None, "x10": None, "x11": None}
    peak = max(v for _, _, v in pts)
    last24 = [v for d, _, v in pts if d <= 1.0]
    mean24 = sum(last24) / len(last24) if last24 else None
    n = len(pts)
    if n < 2:
        slope, vol = 0.0, 0.0
    else:
        mx = sum(d for d, _, _ in pts) / n
        my = sum(y for _, y, _ in pts) / n
        sxx = sum((d - mx) ** 2 for d, _, _ in pts)
        sxy = sum((d - mx) * (y - my) for d, y, _ in pts)
        slope = sxy / sxx if sxx > 1e-9 else 0.0
        vol = math.sqrt(sum((y - my) ** 2 for _, y, _ in pts) / n)
    x2 = math.log(peak / p1) if peak else None
    x3 = math.log(mean24 / p1) if mean24 else None
    x10 = slope * vol if slope is not None and vol is not None else None
    return {"x2": x2, "x3": x3, "x4": slope, "x5": vol, "x10": x10,
            "x11": x2 * x2 if x2 is not None else None}


def build_pool(case):
    pool = []
    for comp in case.get("comparables") or []:
        name = str(comp.get("name", ""))
        if "Grand Theft Auto VI" in name:
            continue
        curve = curve_of(comp)
        if curve is None:
            continue
        pool.append({
            "name": name, "p1": curve[0], "curve": curve,
            "feats": features_of(comp.get("hourly"), comp.get("day_start"), curve[0]),
            "regime": regime_of(comp.get("unlock", "")),
        })
    pool.sort(key=lambda p: p["name"])
    return pool


def target_feats(case, p1):
    return features_of(case.get("history_hourly"), case.get("day_start"), p1)


def pool_means(pool):
    means = {}
    for k in FEATS:
        vals = [p["feats"][k] for p in pool if p["feats"][k] is not None]
        means[k] = sum(vals) / len(vals) if vals else 0.0
    return means


def gauss_solve(mat, vec):
    n = len(vec)
    a = [row[:] + [vec[i]] for i, row in enumerate(mat)]
    for i in range(n):
        piv = max(range(i, n), key=lambda r: abs(a[r][i]))
        if abs(a[piv][i]) < 1e-12:
            return None
        a[i], a[piv] = a[piv], a[i]
        for r in range(i + 1, n):
            f = a[r][i] / a[i][i]
            for c in range(i, n + 1):
                a[r][c] -= f * a[i][c]
    out = [0.0] * n
    for i in range(n - 1, -1, -1):
        out[i] = (a[i][n] - sum(a[i][j] * out[j] for j in range(i + 1, n))) / a[i][i]
    return out


def fit_ridge(rows, ys, ridge):
    k = len(rows[0])
    mat = [[0.0] * k for _ in range(k)]
    vec = [0.0] * k
    for xi, yi in zip(rows, ys):
        for i in range(k):
            vec[i] += xi[i] * yi
            for j in range(k):
                mat[i][j] += xi[i] * xi[j]
    for i in range(k):
        mat[i][i] += ridge[i]
    return gauss_solve(mat, vec)


def med(vals):
    v = sorted(vals)
    n = len(v)
    if n == 0:
        return 0.0
    return v[n // 2] if n % 2 else 0.5 * (v[n // 2 - 1] + v[n // 2])


def sd(vals, mean):
    if len(vals) < 2:
        return 0.0
    return math.sqrt(sum((v - mean) ** 2 for v in vals) / len(vals))


def smooth(vals):
    out = []
    for i in range(len(vals)):
        lo = max(0, i - 1)
        hi = min(len(vals), i + 2)
        out.append(sum(vals[lo:hi]) / (hi - lo))
    return out


def fit_mu_sigma(pool, log_p1, cov):
    xs = [math.log(p["p1"]) for p in pool]
    xbar = sum(xs) / len(xs)
    xsd = sd(xs, xbar) or 1.0
    # usable (row, log-ratio) pairs per hour, built in one pass
    per_hour = [[] for _ in range(24)]
    for p in pool:
        if not all(p["feats"][k] is not None for k in FEATS):
            continue
        u = (math.log(p["p1"]) - xbar) / xsd
        row = [1.0, u, u * u] + [p["feats"][k] for k in FEATS]
        for h in range(24):
            v = p["curve"][h]
            if v is not None and v > 0:
                per_hour[h].append((row, math.log(v / p["p1"])))
    mu, sig = [], []
    for h in range(24):
        pts = per_hour[h]
        if len(pts) < 6:
            mu.append(0.0)
            sig.append(0.5)
            continue
        lh = LAM if h < EVENING_FROM else LAM_EVENING
        ridge = [0.0, lh, lh] + [lh] * len(FEATS)
        beta = fit_ridge([r for r, _ in pts], [y for _, y in pts], ridge)
        if beta is None:
            ys = [y for _, y in pts]
            m0 = med(ys)
            mu.append(m0)
            sig.append(max(sd(ys, m0), 0.15))
            continue
        ue = (log_p1 - xbar) / xsd
        ev = [1.0, ue, ue * ue] + list(cov)
        mu.append(sum(b * v for b, v in zip(beta, ev)))
        resid = [y - sum(b * v for b, v in zip(beta, r)) for r, y in pts]
        rm = sum(resid) / len(resid)
        var = sum((r - rm) ** 2 for r in resid) / len(resid)
        sig.append(max(math.sqrt(var), 0.15))
    return smooth(mu), smooth(sig)


def base_rows(pool, p1, cov):
    mu, sig = fit_mu_sigma(pool, math.log(p1), cov)
    mu = [m + SHIFT for m in mu]
    rows = []
    for h in range(2, 25):
        q = []
        for j, z in enumerate(Z):
            off = z * sig[h - 1] * SPREAD * (UPPER if j >= 3 else 1.0)
            v = p1 * math.exp(min(max(mu[h - 1] + off, -1.2), 2.6))
            q.append(min(max(v, 1.0), 1e9))
        q = sorted(round(v, 2) for v in q)
        for j in range(1, 5):
            if q[j] < q[j - 1]:
                q[j] = q[j - 1]
        rows.append({"hour": h, "q": q})
    return {"hours": rows}


def rolling_miss(pool):
    rolling = [p for p in pool if p["regime"] == "rolling" and p["name"] not in MIXED]
    if len(rolling) < 3:
        return None
    per_hour = [[] for _ in range(24)]
    for target in rolling:
        sub = [p for p in pool if p["name"] != target["name"]]
        if len(sub) < 6:
            continue
        means = pool_means(sub)
        cov = [target["feats"][k] if target["feats"][k] is not None else means[k]
               for k in FEATS]
        out = base_rows(sub, target["p1"], cov)
        rows = {r["hour"]: r["q"] for r in out["hours"]}
        for h in range(2, 25):
            actual = target["curve"][h - 1]
            if actual is None or actual <= 0:
                continue
            per_hour[h - 1].append(
                math.log(max(1, actual)) - math.log(max(1e-9, rows[h][2])))
    return [med(v) if v else 0.0 for v in per_hour]


def bias_for(pool, p1):
    miss = rolling_miss(pool)
    damp = 1.0 / (1.0 + DAMP_K * max(0.0, math.log(p1 / DAMP_REF)))
    w = 1.0 if p1 < SMALL_P1 else BLEND
    weights = []
    for h in range(24):
        if miss is None or (h + 1) in FULL_HOURS:
            weights.append((1.0, 0.0, damp if (h + 1) in FULL_HOURS and miss is not None else 1.0))
        else:
            weights.append((w, 1.0 - w, 1.0))
    return [wp * PRACTICE_BIAS[h] * d + wm * (miss[h] if miss is not None else 0.0)
            for h, (wp, wm, d) in enumerate(weights)]


def apply_bias(rows, bias):
    out = []
    for r in rows["hours"]:
        h = r["hour"]
        q = [min(max(v * math.exp(bias[h - 1]), 1.0), 1e9) for v in r["q"]]
        q = sorted(q)
        for j in range(1, 5):
            if q[j] < q[j - 1]:
                q[j] = q[j - 1]
        out.append({"hour": h, "q": q})
    return {"hours": out}


if __name__ == "__main__":
    run()
