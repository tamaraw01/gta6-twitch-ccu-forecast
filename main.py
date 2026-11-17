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
    val = whole / 100
    return int(val) if val.is_integer() else val


def extract_comparable_log_ratios(case: dict) -> dict[int, list[float]]:
    """Extract valid launch-day hourly log-ratios from comparables."""
    comparables = case.get("comparables", [])
    hourly_log_ratios: dict[int, list[float]] = {h: [] for h in range(2, 25)}

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
            weight = 3 if is_rolling else 1
            for h in range(2, 25):
                for _ in range(weight):
                    hourly_log_ratios[h].append(curve[h])

    return hourly_log_ratios


def forecast_case(case: dict, filename: str) -> dict:
    if case.get("levels") != LEVELS:
        raise ValueError(f"{filename} asks for levels {case.get('levels')}, expected {LEVELS}")

    known = case.get("known", [])
    first_hour = next((k for k in known if k.get("hour") == 1), None)
    if first_hour is None:
        raise ValueError(f"{filename} has no hour-1 print")

    p1 = max(1, first_hour.get("viewers", 1))
    extracted = extract_comparable_log_ratios(case)

    hours_output = []
    for row in case.get("forecast", []):
        h = row.get("hour")
        prior_mean, prior_std = DEFAULT_PRIORS.get(h, (0.0, 0.15 * math.sqrt(h - 1)))

        sample = extracted.get(h, [])
        if len(sample) >= 5:
            sample_mean = sum(sample) / len(sample)
            sample_var = sum((s - sample_mean) ** 2 for s in sample) / (len(sample) - 1)
            sample_std = max(0.1, math.sqrt(sample_var))

            # Bayesian shrinkage towards established prior
            weight_sample = min(1.0, len(sample) / 20.0)
            mu = weight_sample * sample_mean + (1.0 - weight_sample) * prior_mean
            sigma = weight_sample * sample_std + (1.0 - weight_sample) * prior_std
        else:
            mu = prior_mean
            sigma = prior_std

        # Compute quantiles: q_tau = p1 * exp(mu + z_tau * sigma), bounded to valid range
        quantiles = [max(0.01, min(1e9, round2(p1 * math.exp(mu + z * sigma)))) for z in Z]

        # Enforce non-decreasing monotonicity
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
