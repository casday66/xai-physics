from __future__ import annotations

import numpy as np

from src.features import extract_dominant_frequency

EPS = 1e-8


def positive_normalize(attribution: np.ndarray) -> np.ndarray:
    weights = np.abs(attribution).astype(np.float64) + EPS
    return weights / weights.sum()


def physical_consistency_score(attribution: np.ndarray, anomaly_mask: np.ndarray) -> float:
    if anomaly_mask.sum() == 0:
        return np.nan
    normalized = positive_normalize(attribution)
    return float(normalized[anomaly_mask > 0].sum())


def temporal_smearing_score(attribution: np.ndarray, anomaly_mask: np.ndarray) -> float:
    if anomaly_mask.sum() == 0:
        return 0.0
    normalized = positive_normalize(attribution)
    return float(normalized[anomaly_mask == 0].sum())


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


def attribution_ablation_score(original_probability: float, ablated_probability: float) -> float:
    if original_probability <= EPS:
        return 0.0
    return float(np.clip((original_probability - ablated_probability) / original_probability, 0.0, 1.0))


def counterfactual_consistency(
    original_probability: float,
    counterfactual_probability: float,
    counterfactual_label: int,
    expected_label: int,
    attribution_region_drop: float,
) -> float:
    if original_probability <= EPS:
        probability_term = 0.0
    else:
        probability_term = float(np.clip((original_probability - counterfactual_probability) / original_probability, 0.0, 1.0))
    label_term = float(counterfactual_label == expected_label)
    return float(np.mean([label_term, probability_term, np.clip(attribution_region_drop, 0.0, 1.0)]))


def causal_score(ablation_score: float, counterfactual_score: float) -> float:
    return float(np.mean([ablation_score, counterfactual_score]))


def physics_violation_score(consistency: float, attribution_score: float, temporal_smearing: float) -> float:
    consistency_term = 0.0 if np.isnan(consistency) else 1.0 - consistency
    return float(np.mean([consistency_term, 1.0 - attribution_score, temporal_smearing]))


def stability_score(reference_attribution: np.ndarray, perturbed_attribution: np.ndarray) -> float:
    reference = positive_normalize(reference_attribution)
    perturbed = positive_normalize(perturbed_attribution)
    total_variation = 0.5 * np.abs(reference - perturbed).sum()
    return float(max(0.0, 1.0 - total_variation))


def temporal_coherence(attribution: np.ndarray) -> float:
    normalized = positive_normalize(attribution)
    variation = np.abs(np.diff(normalized)).mean()
    return float(1.0 / (1.0 + 1000.0 * variation))
