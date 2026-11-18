# GTA VI Twitch CCU Forecaster

Hour-by-hour probabilistic forecast model for GTA VI's launch-day Twitch CCU (concurrent viewers), built for the `make.fun` competition `gta6-twitch-ccu-forecast-01`.

## Approach

1. **Prior Diurnal Modeling**: Calibrated empirical log-ratio diurnal curves relative to Hour 1 (midnight local launch print).
2. **Empirical Bayes Shrinkage**: Shrinks small-sample comparable series towards stable category priors.
3. **Quantile Calibration & Monotonicity**: Generates strictly monotonic quantiles at $\tau \in \{0.05, 0.25, 0.5, 0.75, 0.95\}$.

## Execution

The model runs against cases in `$MAKEFUN_INPUT_DIR` and writes outputs to `$MAKEFUN_OUTPUT_DIR`.

