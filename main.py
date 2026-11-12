"""Predict Roblox's Up-and-Coming top 10 (make.fun bounty
roblox-up-and-coming-forecast-01).

Method: rank-diffusion Monte Carlo (standard library only, deterministic).
Each game on the chart has a latent score (higher = better rank). It evolves
hour by hour as a random walk: a per-game drift from its recent rank velocity
(shrunk toward zero, fading over the horizon) plus Gaussian noise whose scale
is estimated from the history's own rank changes. Every simulated hour we rank
the games by latent score and record the top 10; averaging over all paths and
hours gives P(game at place k) for the 72-hour scored window. The forecast is
mixed with a broad prior over the current top 40 and leaves 25% of each place
unnamed, which the scorer spreads over the games on each read's chart that we
did not name there -- that covers new entrants and deep climbers, and keeps
any single place from scoring the -9.21 floor.

Reads /input/history.ndjson and /input/task.json, writes the forecast to
$MAKEFUN_OUTPUT_DIR/forecast.json (or /output/forecast.json).
"""
import json
import math
import os
import random
from datetime import datetime, timezone

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

SEED = 12345
N_PATHS = 3000
VEL_WINDOW_H = 72.0      # hours of history used to estimate rank velocity
SHRINK = 0.3             # shrink velocity toward 0 (velocity persists ~0.2)
DRIFT_DECAY_H = 48.0     # velocity fades over the horizon
SIGMA_SCALE = 0.5  # halve the estimated hourly rank noise: the chart's ranks
# are more persistent than consecutive sparse reads suggest, and a tighter
# diffusion scored better on all three validation windows (two backtests
# plus the real practice window, -34.8 vs -36.4)
PRIOR_W = 0.10           # weight of the broad prior over the top 40
UNNAMED = 0.25           # mass left unnamed at each place
CAND_N = 40              # candidate games: the chart's current top 40
MIN_P = 0.001            # rows below this are dropped


def parse_ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def load_history(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("ok") is False:
                continue
            if not r.get("up_and_coming"):
                continue
            rows.append(r)
    rows.sort(key=lambda r: r["at"])
    return rows


def estimate_drift(rows):
    """Per-game rank velocity (ranks/hour, + = worsening) from recent reads."""
    t_end = parse_ts(rows[-1]["at"]) / 3600.0
    series = {}
    for r in rows:
        t = parse_ts(r["at"]) / 3600.0
        for c in r["up_and_coming"]:
            series.setdefault(c["universe_id"], []).append((t, c["rank"]))
    drift = {}
    for u, pts in series.items():
        recent = [(t, rk) for (t, rk) in pts if t >= t_end - VEL_WINDOW_H]
        if len(recent) >= 3:
            n = len(recent)
            mt = sum(t for t, _ in recent) / n
            mr = sum(rk for _, rk in recent) / n
            den = sum((t - mt) ** 2 for t, _ in recent)
            v = sum((t - mt) * (rk - mr) for t, rk in recent) / den if den else 0.0
            drift[u] = max(-3.0, min(3.0, v * SHRINK))
        else:
            drift[u] = 0.0
    return drift


def estimate_sigma(rows):
    """Std of hourly rank changes, pooled over games ranked <= 40."""
    diffs = []
    for i in range(len(rows) - 1):
        dt = (parse_ts(rows[i + 1]["at"]) - parse_ts(rows[i]["at"])) / 3600.0
        if dt <= 0 or dt > 24:
            continue
        a = {c["universe_id"]: c["rank"] for c in rows[i]["up_and_coming"]}
        b = {c["universe_id"]: c["rank"] for c in rows[i + 1]["up_and_coming"]}
        for u in a:
            if u in b and a[u] <= 40:
                diffs.append((b[u] - a[u]) / math.sqrt(dt))
    if len(diffs) >= 10:
        m = sum(diffs) / len(diffs)
        var = sum((d - m) ** 2 for d in diffs) / len(diffs)
        return math.sqrt(var) * SIGMA_SCALE
    return 0.6 * SIGMA_SCALE


def age_out_fraction(rows, window_from, window_until):
    """Fraction of the scored window each game stays on the chart.

    Games leave the Up-and-Coming chart 28 days after they were created.
    Uses the details of the latest read when present; 1.0 when unknown.
    """
    last = rows[-1]
    details = last.get("details") or {}
    out = {}
    span = max(1.0, window_until - window_from)
    for u, d in details.items():
        try:
            created = parse_ts(d["created"])
        except (KeyError, TypeError, ValueError):
            continue
        leave = created + 28 * 86400
        if leave <= window_from:
            out[int(u)] = 0.0
        elif leave < window_until:
            out[int(u)] = (leave - window_from) / span
        else:
            out[int(u)] = 1.0
    return out


def main():
    with open(os.path.join(INPUT_DIR, "task.json"), encoding="utf-8") as f:
        task = json.load(f)
    places = int(task.get("places", 10))
    hours = int(task.get("window", {}).get("slots", 72))
    window = task.get("window", {})
    w_from = parse_ts(window["from"]) if window.get("from") else 0.0
    w_until = parse_ts(window["until"]) if window.get("until") else w_from + hours * 3600

    rows = load_history(os.path.join(INPUT_DIR, "history.ndjson"))
    last = rows[-1]
    chart = sorted(last["up_and_coming"], key=lambda c: c["rank"])
    cand = [c["universe_id"] for c in chart[:CAND_N]]
    rank0 = {c["universe_id"]: c["rank"] for c in chart}
    drift = estimate_drift(rows)
    sigma = estimate_sigma(rows)
    elig = age_out_fraction(rows, w_from, w_until)
    cand = [u for u in cand if elig.get(u, 1.0) > 0.0]
    if len(cand) < places:
        cand = [c["universe_id"] for c in chart[:max(places, CAND_N)]]

    rng = random.Random(SEED)
    counts = {}
    for _ in range(N_PATHS):
        s = {u: -float(rank0[u]) for u in cand}
        for h in range(hours):
            decay = math.exp(-h / DRIFT_DECAY_H)
            for u in cand:
                s[u] += -drift.get(u, 0.0) * decay + rng.gauss(0.0, sigma)
            order = sorted(cand, key=lambda u: s[u], reverse=True)
            for place, u in enumerate(order[:places], start=1):
                counts[(place, u)] = counts.get((place, u), 0) + 1
    total = N_PATHS * hours

    # Broad prior over candidates, decaying with current rank.
    prior = {u: 1.0 / (rank0[u] ** 0.7) for u in cand}
    psum = sum(prior.values())
    for u in cand:
        prior[u] /= psum

    out = {}
    for place in range(1, places + 1):
        for u in cand:
            p = (1 - PRIOR_W) * counts.get((place, u), 0) / total
            p += PRIOR_W * prior[u] * 0.5
            p *= elig.get(u, 1.0)
            if p > 0.0005:
                out[(place, u)] = p
    for place in range(1, places + 1):
        ssum = sum(p for (pl, u), p in out.items() if pl == place)
        if ssum > 0:
            scale = (1.0 - UNNAMED) / ssum
            for k in [k for k in out if k[0] == place]:
                out[k] *= scale

    # Top 10: order by expected place, tie-break by total probability mass.
    tot, exp_place = {}, {}
    for (place, u), p in out.items():
        tot[u] = tot.get(u, 0.0) + p
        exp_place[u] = exp_place.get(u, 0.0) + p * place
    order = sorted(tot, key=lambda u: (exp_place[u] / tot[u], -tot[u]))
    top10 = order[:places]
    # Fill any shortfall with the chart's current order.
    for u in [c["universe_id"] for c in chart]:
        if len(top10) >= places:
            break
        if u not in top10:
            top10.append(u)

    probabilities = [
        {"universe_id": u, "place": pl, "p": round(p, 6)}
        for (pl, u), p in sorted(out.items())
        if p >= MIN_P
    ]
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, task.get("output", "forecast.json")),
              "w", encoding="utf-8") as f:
        f.write(json.dumps({"probabilities": probabilities, "top10": top10}) + "\n")


if __name__ == "__main__":
    main()
