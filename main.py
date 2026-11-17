"""gta6-twitch-ccu-forecast-01 -- vocaine entry: nearest-neighbour analog forecast.

Approach: for the target launch, find the K past launches (comparables) most
similar to it -- by (a) log size of the hour-1 print, (b) the shape of the
pre-release hype curve (log viewers at fixed anchors before release day,
scale-free, relative to the day-before level, plus the launch jump
log(p1) - log(pre-release day -1 level)), and (c) platform/unlock class
(console-only + midnight-local rolling vs multi-platform / single global).
Each selected analog contributes its own launch-day curve as a ratio to its
own hour-1 print; the forecast median for hour h is p1 times the
distance-weighted mean of the analogs' log ratios, and the spread is the
weighted dispersion of those log ratios at that hour. Everything is computed
at runtime from the case's own data -- no answer is hardcoded.

Guardrails (RULES.txt): Python 3.13 stdlib only, no network; reads every
*.in in $MAKEFUN_INPUT_DIR (default /input), writes <id>.out to
$MAKEFUN_OUTPUT_DIR (default /output); exactly 23 rows, hours 2..24, five
quantiles each, > 0, <= 1e9, non-decreasing; output <= 65,536 bytes;
deterministic (no randomness, all orderings sorted); well under 8 s.
"""

import json
import math
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]

# --- tuning constants (chosen by leave-one-out validation on the kit's
# comparables; see validation notes in README) ---
K_ANALOGS = 8
W_SCALE = 2.0        # weight of |log p1 - log h1| distance
W_SHAPE = 0.5        # weight of pre-release shape distance
W_JUMP = 0.5         # weight of launch-jump distance
PENALTY_PLATFORM = 0.5   # added distance when platform class differs
PENALTY_UNLOCK = 0.0     # added distance when unlock class differs (tuned: 0)
WEIGHT_EPS = 0.05    # 1 / (dist + eps) analog weights
TRIM_FACTOR = 0.0    # >0: drop analogs with dist > factor * best dist (0 = off)
USE_MEDIAN = False   # weighted median of analog log-ratios (else weighted mean)
SHRINK = 0.0         # shrink log-ratio median toward 0 (persistence) by this share
SIGMA_FLOOR = 0.20   # minimum log spread per hour
SIGMA_CAP = 0.90     # maximum log spread per hour
SIGMA_GROWTH = 0.0   # extra spread per sqrt(hour - 1), log units (tuned: 0)
SIGMA_MULT = 1.25    # global multiplier on the per-hour log spread

# Pre-release anchors, in days before the release day's 00:00 UTC.
ANCHORS_DAYS = [-14, -10, -7, -5, -3, -2, -1]
ANCHOR_TOL_HOURS = 12.0


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def round2(x):
    """Round to 2 decimals, half up (same convention as the starter)."""
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value.is_integer() else value


def hourly_series(points):
    """points: list of {'from'/'at', 'viewers'} -> sorted list of (time, log viewers)."""
    out = []
    for p in points:
        t = p.get("from") or p.get("at")
        v = p.get("viewers")
        if t is None or v is None:
            continue
        v = max(1.0, float(v))
        out.append((parse_time(t), math.log(v)))
    out.sort(key=lambda x: x[0])
    return out


def nearest_log(series, target, tol_hours):
    """Log viewers of the point nearest to target within tol_hours, else None."""
    if not series:
        return None
    best = None
    best_dt = None
    for t, lv in series:
        dt = abs((t - target).total_seconds()) / 3600.0
        if best_dt is None or dt < best_dt:
            best_dt = dt
            best = lv
    if best_dt is not None and best_dt <= tol_hours:
        return best
    return None


def shape_features(series, day_start):
    """Scale-free pre-release shape: log viewers at anchors relative to the
    day -1 anchor, plus the 3-day log slope. Returns (dict anchor->value,
    slope or None, day-minus-1 log level or None)."""
    anchors = {}
    for d in ANCHORS_DAYS:
        anchors[d] = nearest_log(series, day_start + timedelta(days=d),
                                 ANCHOR_TOL_HOURS)
    base = anchors.get(-1)
    shape = {}
    if base is not None:
        for d, lv in anchors.items():
            if lv is not None:
                shape[d] = lv - base
    # Slope of log viewers over the last 3 pre-release days (per day).
    window = [(t, lv) for t, lv in series
              if day_start - timedelta(days=3) <= t < day_start]
    slope = None
    if len(window) >= 4:
        t0 = window[0][0]
        xs = [(t - t0).total_seconds() / 86400.0 for t, _ in window]
        ys = [lv for _, lv in window]
        n = len(xs)
        mx = sum(xs) / n
        my = sum(ys) / n
        den = sum((x - mx) ** 2 for x in xs)
        if den > 0:
            slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den
    return shape, slope, base


def platform_class(platforms):
    """True when the launch is console-only (no PC)."""
    if not platforms:
        return None
    return "PC" not in str(platforms)


def unlock_class(unlock):
    """'rolling' for midnight-local rolling unlocks, 'global' for a single
    simultaneous unlock, else None."""
    if not unlock:
        return None
    u = str(unlock).lower()
    if "midnight local" in u or "rolling" in u:
        return "rolling"
    if "simultaneous" in u or "single global" in u or "global unlock" in u:
        return "global"
    return None


def build_curve(hourly_points, day_start):
    """Map hour (1..72, relative to day_start) -> viewers for a comparable."""
    curve = {}
    for p in hourly_points:
        t = p.get("from")
        v = p.get("viewers")
        if t is None or v is None:
            continue
        h = int((parse_time(t) - day_start).total_seconds() // 3600) + 1
        if 1 <= h <= 72:
            curve[h] = max(1.0, float(v))
    return curve


def forecast_case(case, name):
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    first = next((k for k in case["known"] if k["hour"] == 1), None)
    if first is None:
        raise ValueError(f"{name} has no hour-1 print")
    p1 = max(1.0, float(first["viewers"]))
    log_p1 = math.log(p1)
    day_start = parse_time(case["day_start"])

    # Target pre-release shape from its hourly history.
    target_series = hourly_series(case.get("history_hourly") or [])
    t_shape, t_slope, t_base = shape_features(target_series, day_start)
    t_jump = (log_p1 - t_base) if t_base is not None else None

    # Score every comparable as a potential analog.
    scored = []
    for comp in case.get("comparables") or []:
        ds = parse_time(comp["day_start"])
        curve = build_curve(comp.get("hourly") or [], ds)
        h1 = curve.get(1)
        if h1 is None:
            continue  # not a launch curve (e.g. the GTA VI pre-release entry)
        if any(h not in curve for h in range(2, 25)):
            continue  # needs a full release-day curve
        log_h1 = math.log(h1)
        d_scale = abs(log_p1 - log_h1)
        c_series = hourly_series(comp.get("hourly") or [])
        c_shape, c_slope, c_base = shape_features(c_series, ds)
        shared = [d for d in t_shape if d in c_shape]
        if shared:
            d_shape = math.sqrt(sum((t_shape[d] - c_shape[d]) ** 2
                                    for d in shared) / len(shared))
        else:
            d_shape = 0.0
        d_jump = 0.0
        if t_jump is not None and c_base is not None:
            d_jump = abs(t_jump - (log_h1 - c_base))
        dist = W_SCALE * d_scale + W_SHAPE * d_shape + W_JUMP * d_jump
        if t_slope is not None and c_slope is not None:
            dist += 0.25 * abs(t_slope - c_slope)
        plat_t = platform_class("PS5, Xbox Series X|S")  # GTA VI: console-only
        plat_c = platform_class(comp.get("platforms"))
        if plat_t is not None and plat_c is not None and plat_t != plat_c:
            dist += PENALTY_PLATFORM
        unl_t = "rolling"  # GTA VI unlocks midnight local, rolling
        unl_c = unlock_class(comp.get("unlock"))
        if unl_c is not None and unl_t != unl_c:
            dist += PENALTY_UNLOCK
        scored.append((dist, comp["name"], curve, log_h1))

    # Fall back to the house persistence shape if no usable analog exists.
    if not scored:
        return {"hours": [{"hour": h, "q": [
            round2(p1 * math.exp(z * 0.15 * math.sqrt(h - 1))) for z in Z]}
            for h in range(2, 25)]}

    scored.sort(key=lambda x: (x[0], x[1]))
    chosen = scored[:K_ANALOGS]
    if TRIM_FACTOR > 0 and chosen:
        best_d = chosen[0][0]
        chosen = [c for c in chosen if c[0] <= TRIM_FACTOR * best_d + 1e-9]
    weights = [1.0 / (d + WEIGHT_EPS) for d, _, _, _ in chosen]
    wsum = sum(weights)

    hours = []
    for h in range(2, 25):
        logs = []
        ws = []
        for (d, _nm, curve, log_h1), w in zip(chosen, weights):
            logs.append(math.log(curve[h]) - log_h1)
            ws.append(w)
        tw = sum(ws)
        if USE_MEDIAN:
            order = sorted(range(len(logs)), key=lambda i: logs[i])
            acc = 0.0
            mean = logs[order[-1]]
            for i in order:
                acc += ws[i]
                if acc >= tw / 2.0:
                    mean = logs[i]
                    break
        else:
            mean = sum(w * l for w, l in zip(ws, logs)) / tw
        mean = (1.0 - SHRINK) * mean
        var = sum(w * (l - mean) ** 2 for w, l in zip(ws, logs)) / tw
        sigma = math.sqrt(var)
        sigma = min(SIGMA_CAP, max(SIGMA_FLOOR,
                                    math.sqrt(sigma * sigma +
                                              (SIGMA_GROWTH ** 2) * (h - 1))))
        sigma *= SIGMA_MULT
        med = p1 * math.exp(mean)
        qs = []
        prev = 0.0
        for z in Z:
            q = round2(med * math.exp(z * sigma))
            q = max(1.0, min(1e9, float(q)))
            if q < prev:
                q = prev
            prev = q
            qs.append(q)
        hours.append({"hour": h, "q": qs})
    return {"hours": hours}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
