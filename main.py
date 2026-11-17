"""Forecast GTA VI launch-day Twitch CCU hour by hour for make.fun.

This model predicts the 24-hour concurrent viewer trajectory of GTA VI on Twitch
(category 1867279146) from hour 1 print (p1) using an empirical diurnal log-ratio
distribution modeled from comparable AAA midnight-local release curves.
"""

import json
import math
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# Calibrated prior parameters for hourly log-ratio relative to hour 1 (hours 2..24).
# Based on historical rolling-release AAA game launch curves on Twitch.
DEFAULT_PRIORS = {
    2:  (-0.112, 0.180),
    3:  (-0.333, 0.320),
    4:  (-0.386, 0.320),
    5:  (-0.211, 0.350),
    6:  ( 0.019, 0.480),
    7:  ( 0.041, 0.550),
    8:  ( 0.018, 0.580),
    9:  (-0.028, 0.600),
    10: (-0.032, 0.600),
    11: (-0.007, 0.600),
    12: ( 0.055, 0.600),
    13: ( 0.195, 0.600),
    14: ( 0.338, 0.600),
    15: ( 0.455, 0.600),
    16: ( 0.530, 0.600),
    17: ( 0.607, 0.600),
    18: ( 0.668, 0.620),
    19: ( 0.717, 0.640),
    20: ( 0.729, 0.650),
    21: ( 0.735, 0.640),
    22: ( 0.707, 0.630),
    23: ( 0.646, 0.620),
    24: ( 0.546, 0.640),
}


def round2(x: float) -> float | int:
    """Round to 2 decimal places half-up matching JavaScript Math.round."""
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    val = round(whole / 100, 2)
    return int(val) if val.is_integer() else val


def bound_quantile(v: float) -> float | int:
    """Bound quantile to valid range (0, 1e9], non-zero positive, rounded to 2 decimals."""
    if not math.isfinite(v) or v < 0.01:
        v = 0.01
    elif v > 1e9:
        v = 1e9
    res = round2(v)
    if res <= 0:
        res = 0.01
    elif res > 1e9:
        res = 1_000_000_000
    return res


def extract_comparable_log_ratios(case: dict) -> dict[int, list[float]]:
    """Extract valid launch-day hourly log-ratios from comparables.

    Prioritizes rolling midnight-local releases matching GTA VI's launch structure.
    Falls back to general comparables if insufficient rolling releases are available.
    """
    comparables = case.get("comparables", [])
    rolling_ratios: dict[int, list[float]] = {h: [] for h in range(2, 25)}
    general_ratios: dict[int, list[float]] = {h: [] for h in range(2, 25)}

    for comp in comparables:
        day_start_str = comp.get("day_start")
        if not day_start_str:
            continue
        try:
            dt_start = datetime.fromisoformat(day_start_str.replace("Z", "+00:00"))
        except Exception:
            continue

        pts = {p.get("from"): p.get("viewers") for p in comp.get("hourly", [])}
        h1 = pts.get(day_start_str)
        if not h1 or h1 < 1000:
            continue

        unlock = str(comp.get("unlock", "")).lower()
        is_rolling = "rolling" in unlock or "midnight local" in unlock

        curve: dict[int, float] = {}
        valid = True
        for h in range(2, 25):
            t_str = (dt_start + timedelta(hours=h - 1)).strftime("%Y-%m-%dT%H:%M:%SZ")
            v = pts.get(t_str)
            if v is not None and v > 0:
                curve[h] = math.log(v / h1)
            else:
                valid = False
                break

        if valid:
            for h in range(2, 25):
                if is_rolling:
                    rolling_ratios[h].append(curve[h])
                else:
                    general_ratios[h].append(curve[h])

    # If rolling curves are present across the day, use them to avoid simultaneous-release artifacts
    hours_with_rolling = sum(1 for h in range(2, 25) if len(rolling_ratios[h]) >= 3)
    if hours_with_rolling == 23:
        return rolling_ratios

    # Fall back to combined sample
    combined: dict[int, list[float]] = {h: [] for h in range(2, 25)}
    for h in range(2, 25):
        combined[h] = rolling_ratios[h] + general_ratios[h]
    return combined


def forecast_case(case: dict, filename: str) -> dict:
    if case.get("levels") != LEVELS:
        raise ValueError(f"{filename} asks for levels {case.get('levels')}, expected {LEVELS}")

    known = case.get("known", [])
    first_hour = next((k for k in known if k.get("hour") == 1), None)
    if first_hour is None:
        raise ValueError(f"{filename} has no hour-1 print")

    v1 = first_hour.get("viewers")
    p1 = max(1.0, float(v1)) if v1 is not None else 1.0
    extracted = extract_comparable_log_ratios(case)

    hours_output = []
    forecast_rows = case.get("forecast", [])
    if not forecast_rows:
        forecast_rows = [{"hour": h} for h in range(2, 25)]

    for row in forecast_rows:
        h = row.get("hour")
        prior_mean, prior_std = DEFAULT_PRIORS.get(h, (0.0, 0.15 * math.sqrt(h - 1)))

        sample = extracted.get(h, [])
        n = len(sample)
        if n >= 3:
            sample_mean = sum(sample) / n
            sample_var = sum((s - sample_mean) ** 2 for s in sample) / (n - 1)
            sample_std = max(0.1, math.sqrt(sample_var))

            # Empirical Bayes shrinkage towards calibrated diurnal prior (N0 = 10 pseudo-observations)
            N0 = 10.0
            weight_sample = n / (n + N0)
            mu = weight_sample * sample_mean + (1.0 - weight_sample) * prior_mean
            sigma = weight_sample * sample_std + (1.0 - weight_sample) * prior_std
        else:
            mu = prior_mean
            sigma = prior_std

        sigma = max(0.05, min(2.0, sigma))

        # Compute quantiles: q_tau = p1 * exp(mu + z_tau * sigma), strictly bounded and rounded
        quantiles = [
            bound_quantile(p1 * math.exp(max(-50.0, min(50.0, mu + z * sigma))))
            for z in Z
        ]

        # Enforce strict non-decreasing monotonicity: q05 <= q25 <= q50 <= q75 <= q95
        for i in range(1, len(quantiles)):
            if quantiles[i] < quantiles[i - 1]:
                quantiles[i] = quantiles[i - 1]

        hours_output.append({"hour": h, "q": quantiles})

    return {"hours": hours_output}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    case_paths = sorted(INPUT_DIR.glob("*.in"))
    for path in case_paths:
        content = path.read_text(encoding="utf-8")
        case = json.loads(content)
        forecast = forecast_case(case, path.name)
        out_path = OUTPUT_DIR / f"{path.stem}.out"
        out_path.write_text(json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
