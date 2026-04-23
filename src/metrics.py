from __future__ import annotations

import numpy as np

from src.features import extract_dominant_frequency

EPS = 1e-8


def positive_normalize(attribution: np.ndarray) -> np.ndarray:
    weights = np.abs(attribution).astype(np.float64) + EPS
    return weights / weights.sum()


def physical_consistency_score(attribution: np.ndarray, region_start: float, region_end: float) -> float:
    if np.isnan(region_start) or np.isnan(region_end):
        return np.nan
    start = int(region_start)
    end = int(region_end)
    normalized = positive_normalize(attribution)
    return float(normalized[start:end].sum())


def attribution_weighted_frequency(
    signal: np.ndarray,
    attribution: np.ndarray,
    sample_rate: float,
    exclude_band: tuple[float, float] | None = None,
) -> float:
    weights = positive_normalize(attribution)
    weighted_signal = (signal - np.mean(signal)) * weights
    return extract_dominant_frequency(weighted_signal, sample_rate=sample_rate, exclude_band=exclude_band)


def frequency_alignment_score(reference_frequency: float, attribution_frequency: float, scale_hz: float = 10.0) -> float:
    return float(np.exp(-abs(reference_frequency - attribution_frequency) / scale_hz))


def stability_score(reference_attribution: np.ndarray, perturbed_attribution: np.ndarray) -> float:
    reference = positive_normalize(reference_attribution)
    perturbed = positive_normalize(perturbed_attribution)
    total_variation = 0.5 * np.abs(reference - perturbed).sum()
    return float(max(0.0, 1.0 - total_variation))


def temporal_coherence(attribution: np.ndarray) -> float:
    normalized = positive_normalize(attribution)
    variation = np.abs(np.diff(normalized)).mean()
    return float(1.0 / (1.0 + 1000.0 * variation))
