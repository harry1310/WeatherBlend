"""
Probabilistic precipitation scoring — mirrors src/WeatherBlend/Evaluate/Precip/PrecipMetrics.cs
so the Python side can cross-check the .NET implementation and produce plots.

Public functions:
  brier(prob, truth_binary) -> float
  brier_skill_score(brier, reference) -> float
  reliability_table(prob, truth_binary, n_bins=10) -> pd.DataFrame
  frequency_bias(forecast_binary, truth_binary) -> float
  contingency(forecast_binary, truth_binary) -> dict
  crps_sample(forecast_samples, truth) -> float   # proper CRPS for ensemble forecasts

Usage from a notebook or the report generator:
  import sys; sys.path.insert(0, 'scripts')
  from precip_eval.metrics import brier, reliability_table
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _pair(prob, truth):
    p = np.asarray(prob, dtype=float)
    y = np.asarray(truth, dtype=float)
    if p.shape != y.shape:
        raise ValueError(f"Shape mismatch: prob={p.shape}, truth={y.shape}")
    mask = ~(np.isnan(p) | np.isnan(y))
    return p[mask], y[mask]


def brier(prob, truth_binary) -> float:
    """Brier score = mean((p - y)^2). y must be 0 or 1."""
    p, y = _pair(prob, truth_binary)
    if p.size == 0:
        return float("nan")
    return float(np.mean((p - y) ** 2))


def brier_skill_score(score: float, reference: float) -> float:
    if reference == 0:
        return float("nan")
    return 1.0 - score / reference


def reliability_table(prob, truth_binary, n_bins: int = 10) -> pd.DataFrame:
    """
    Returns a DataFrame with columns:
      bin_lower, bin_upper, mean_forecast, observed_frequency, n
    """
    p, y = _pair(prob, truth_binary)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        if b == n_bins - 1:
            mask = (p >= lo) & (p <= hi)
        else:
            mask = (p >= lo) & (p < hi)
        n = int(mask.sum())
        if n == 0:
            rows.append((lo, hi, float("nan"), float("nan"), 0))
        else:
            rows.append((lo, hi, float(p[mask].mean()), float(y[mask].mean()), n))
    return pd.DataFrame(rows, columns=["bin_lower", "bin_upper", "mean_forecast",
                                        "observed_frequency", "n"])


def frequency_bias(forecast_binary, truth_binary) -> float:
    """forecast_wet_count / observed_wet_count. 1 = unbiased."""
    f, y = _pair(forecast_binary, truth_binary)
    f_wet = int(np.sum(f >= 0.5))
    y_wet = int(np.sum(y >= 0.5))
    if y_wet == 0:
        return float("nan")
    return f_wet / y_wet


def contingency(forecast_binary, truth_binary) -> dict:
    """2x2 confusion matrix + POD, FAR, CSI, frequency bias."""
    f, y = _pair(forecast_binary, truth_binary)
    fb = (f >= 0.5).astype(int)
    yb = (y >= 0.5).astype(int)
    hits   = int(np.sum((fb == 1) & (yb == 1)))
    misses = int(np.sum((fb == 0) & (yb == 1)))
    fa     = int(np.sum((fb == 1) & (yb == 0)))
    tn     = int(np.sum((fb == 0) & (yb == 0)))
    total = hits + misses + fa + tn
    pod = hits / (hits + misses) if (hits + misses) else float("nan")
    far = fa / (hits + fa) if (hits + fa) else float("nan")
    csi = hits / (hits + misses + fa) if (hits + misses + fa) else float("nan")
    fbias = (hits + fa) / (hits + misses) if (hits + misses) else float("nan")
    return {
        "hits": hits, "misses": misses, "false_alarms": fa, "true_negatives": tn,
        "total": total, "pod": pod, "far": far, "csi": csi, "frequency_bias": fbias,
    }


def crps_sample(forecast_samples, truth) -> float:
    """
    Proper CRPS for an ensemble/sampled forecast.

    Uses the fair estimator from Ferro 2014:
      CRPS = (1/N) sum_i [ mean_j |x_ij - y_i| - 0.5 * mean_{j,k} |x_ij - x_ik| ]

    `forecast_samples` shape: (n_rows, n_members). `truth` shape: (n_rows,).
    """
    x = np.asarray(forecast_samples, dtype=float)
    y = np.asarray(truth, dtype=float)
    if x.ndim != 2:
        raise ValueError(f"forecast_samples must be 2D (n_rows, n_members), got {x.shape}")
    if y.ndim != 1 or x.shape[0] != y.shape[0]:
        raise ValueError(f"truth shape {y.shape} does not match forecast rows {x.shape[0]}")

    mask = ~np.isnan(y) & ~np.any(np.isnan(x), axis=1)
    if not mask.any():
        return float("nan")
    x = x[mask]
    y = y[mask]

    # Term 1: mean_j |x_ij - y_i|
    term1 = np.mean(np.abs(x - y[:, None]), axis=1)
    # Term 2: 0.5 * mean_{j,k} |x_ij - x_ik|; use sort trick for O(m log m) per row
    xs = np.sort(x, axis=1)
    m = xs.shape[1]
    # sum over j<k of (xs[:,k] - xs[:,j]) = sum_k xs[:,k] * (2k - m + 1)
    weights = (2 * np.arange(m) - m + 1).astype(float)
    mad_sum = np.sum(xs * weights, axis=1)
    term2 = mad_sum / (m * m)  # = (2/m^2) * sum_{j<k}(xs_k - xs_j)

    return float(np.mean(term1 - term2))


if __name__ == "__main__":
    # Quick smoke test — values should match PrecipMetricsTests.cs.
    p = np.array([0.3] * 10)
    y = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
    assert abs(brier(p, y) - 0.21) < 1e-9, "brier mismatch"
    assert frequency_bias([1, 1, 1, 0], [1, 0, 0, 0]) == 3.0, "freq bias mismatch"
    print("metrics.py smoke test OK")
