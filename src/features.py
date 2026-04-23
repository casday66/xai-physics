from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
EPS = 1e-8


def load_dataset(data_dir: Path = DATA_DIR) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    signals = np.load(data_dir / "signals.npy")
    labels = np.load(data_dir / "labels.npy")
    metadata = pd.read_csv(data_dir / "metadata.csv")
    return signals.astype(np.float32), labels.astype(np.int64), metadata


def load_counterfactuals(data_dir: Path = DATA_DIR) -> np.ndarray:
    return np.load(data_dir / "counterfactuals.npy").astype(np.float32)


def parse_regions(region_string: str | float) -> list[tuple[int, int]]:
    if isinstance(region_string, float) and np.isnan(region_string):
        return []
    if not isinstance(region_string, str) or region_string == "none":
        return []
    regions = []
    for token in region_string.split(";"):
        start, end = token.split(":")
        regions.append((int(start), int(end)))
    return regions


def build_anomaly_mask(region_string: str | float, signal_length: int) -> np.ndarray:
    mask = np.zeros(signal_length, dtype=np.float32)
    for start, end in parse_regions(region_string):
        mask[start:end] = 1.0
    return mask


def frequency_axis(signal_length: int, sample_rate: float) -> np.ndarray:
    return np.fft.rfftfreq(signal_length, d=1.0 / sample_rate)


def compute_fft_magnitude(signal: np.ndarray) -> np.ndarray:
    centered = signal - np.mean(signal)
    return np.abs(np.fft.rfft(centered))


def extract_dominant_frequency(
    signal: np.ndarray,
    sample_rate: float,
    min_frequency: float = 1.0,
    exclude_band: tuple[float, float] | None = None,
) -> float:
    spectrum = compute_fft_magnitude(signal)
    freqs = frequency_axis(signal.shape[-1], sample_rate)
    valid = freqs >= min_frequency
    if exclude_band is not None:
        low, high = exclude_band
        valid &= ~((freqs >= low) & (freqs <= high))
    if not np.any(valid):
        valid = freqs >= min_frequency
    return float(freqs[valid][np.argmax(spectrum[valid])])


def resize_vector(values: np.ndarray, target_length: int) -> np.ndarray:
    x_old = np.linspace(0.0, 1.0, num=values.shape[0], dtype=np.float32)
    x_new = np.linspace(0.0, 1.0, num=target_length, dtype=np.float32)
    return np.interp(x_new, x_old, values).astype(np.float32)


def normalize_signal(signal: np.ndarray) -> np.ndarray:
    signal = signal.astype(np.float32)
    return (signal - signal.mean()) / (signal.std() + EPS)


def prepare_model_inputs(signals: np.ndarray, sample_rate: float) -> dict[str, np.ndarray]:
    length = signals.shape[-1]
    raw_features = np.stack([normalize_signal(signal) for signal in signals]).astype(np.float32)
    fft_magnitudes = []
    dominant_freqs = []
    for signal in signals:
        fft_mag = np.log1p(compute_fft_magnitude(signal))
        fft_mag = fft_mag / (fft_mag.max() + EPS)
        fft_magnitudes.append(resize_vector(fft_mag, target_length=length))
        dominant_freqs.append(extract_dominant_frequency(signal, sample_rate))
    fft_features = np.stack(fft_magnitudes).astype(np.float32)
    dominant_freqs = np.array(dominant_freqs, dtype=np.float32)
    nyquist = sample_rate / 2.0
    dominant_freqs_normalized = (dominant_freqs / nyquist).reshape(-1, 1)
    series = np.stack([raw_features, fft_features], axis=1).astype(np.float32)
    return {
        "series": series,
        "raw": raw_features,
        "fft": fft_features,
        "dominant_frequency": dominant_freqs,
        "dominant_frequency_normalized": dominant_freqs_normalized.astype(np.float32),
    }
