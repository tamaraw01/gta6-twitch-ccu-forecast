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

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z_SCORES = [-1.64485, -0.67449, 0.0, 0.67449, 1.64485]

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


def bound_quantile(val: float) -> float:
    """Clamp quantile to reasonable bounds and round to 2 decimals.
    
    Guarantees positive values suitable for viewer count distributions.
    """
    clamped = max(1.0, min(100_000_000.0, val))
    return round(clamped, 2)


def extract_comparable_ratios(case: dict) -> dict[int, list[float]]:
    """Extract hour-by-hour log ratios from comparable launches in input case."""
    extracted: dict[int, list[float]] = {h: [] for h in range(2, 25)}
    history = case.get("history") or case.get("comparables") or case.get("series") or []
    if isinstance(history, dict):
        history = list(history.values())

    for item in history:
        if not isinstance(item, dict):
            continue
        series = item.get("ccu") or item.get("viewers") or item.get("series") or []
        if isinstance(series, list) and len(series) >= 2:
            h1 = series[0]
            if isinstance(h1, (int, float)) and h1 > 0:
                for idx, ccu in enumerate(series[1:], start=2):
                    if idx in extracted and isinstance(ccu, (int, float)) and ccu > 0:
                        extracted[idx].append(math.log(ccu / h1))
    return extracted


def forecast_case(case: dict) -> dict:
    """Produce probabilistic forecast quantiles for a single test case."""
    p1 = float(case.get("hour_1_ccu") or case.get("p1") or case.get("h1") or 100_000.0)
    p1 = max(10.0, p1)

    extracted = extract_comparable_ratios(case)
    hours_output = []

    forecast_hours = case.get("forecast_hours") or list(range(2, 25))
    if isinstance(forecast_hours, list) and forecast_hours and isinstance(forecast_hours[0], dict):
        hours_list = [row.get("hour") for row in forecast_hours if row.get("hour")]
    else:
        hours_list = list(forecast_hours)

    for h in hours_list:
        prior_mean, prior_std = DEFAULT_PRIORS.get(h, (0.0, 0.15 * math.sqrt(max(1, h - 1))))
        sample = extracted.get(h, [])
        n = len(sample)

        if n >= 3:
            sample_mean = sum(sample) / n
            sample_var = sum((s - sample_mean) ** 2 for s in sample) / (n - 1)
            sample_std = max(0.1, math.sqrt(sample_var))

            # Empirical Bayes shrinkage towards calibrated diurnal prior (N0 = 10 pseudo-observations)
            N0 = 10.0
            weight = n / (n + N0)
            mu = weight * sample_mean + (1.0 - weight) * prior_mean
            sigma = weight * sample_std + (1.0 - weight) * prior_std
        else:
            mu = prior_mean
            sigma = prior_std

        sigma = max(0.05, min(2.0, sigma))

        # Compute quantiles: q_tau = p1 * exp(mu + z_tau * sigma)
        quantiles = [
            bound_quantile(p1 * math.exp(max(-50.0, min(50.0, mu + z * sigma))))
            for z in Z_SCORES
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
        forecast = forecast_case(case)
        out_path = OUTPUT_DIR / f"{path.stem}.out"
        out_path.write_text(json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
