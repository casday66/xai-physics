from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src import CLASS_NAMES, LABEL_TO_INDEX

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
SEED = 42
SIGNAL_LENGTH = 1000
SAMPLE_RATE = 200.0
SAMPLES_PER_CLASS = 180
MIXTURE_POOL = np.array([8.0, 11.0, 14.0, 18.0, 22.0, 28.0, 32.0, 36.0, 42.0, 48.0, 55.0], dtype=np.float32)


def time_axis(length: int = SIGNAL_LENGTH, sample_rate: float = SAMPLE_RATE) -> np.ndarray:
    return np.arange(length, dtype=np.float32) / sample_rate


def burst_mask(length: int, regions: list[tuple[int, int]]) -> np.ndarray:
    mask = np.zeros(length, dtype=np.float32)
    for start, end in regions:
        width = max(end - start, 2)
        mask[start:end] += np.hanning(width).astype(np.float32)
    return np.clip(mask, 0.0, 1.0)


def drifting_sinusoid(
    t: np.ndarray,
    base_frequency: float,
    amplitude: float,
    phase: float,
    drift_strength: float,
    drift_rate: float,
    modulation: np.ndarray | None = None,
) -> np.ndarray:
    instantaneous_frequency = base_frequency + drift_strength * np.sin(2.0 * np.pi * drift_rate * t + 0.5 * phase)
    phase_trajectory = phase + 2.0 * np.pi * np.cumsum(instantaneous_frequency) / SAMPLE_RATE
    wave = amplitude * np.sin(phase_trajectory)
    if modulation is not None:
        wave = wave * modulation
    return wave.astype(np.float32)


def format_regions(regions: list[tuple[int, int]]) -> str:
    if not regions:
        return "none"
    return ";".join(f"{start}:{end}" for start, end in regions)


def colored_noise(rng: np.random.Generator, beta: float, length: int) -> np.ndarray:
    white = rng.normal(size=length)
    spectrum = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(length, d=1.0 / SAMPLE_RATE)
    scaling = np.ones_like(freqs)
    scaling[1:] = 1.0 / np.power(freqs[1:], beta / 2.0)
    colored = np.fft.irfft(spectrum * scaling, n=length)
    colored = colored / (np.std(colored) + 1e-8)
    return colored.astype(np.float32)

def sample_regions(
    rng: np.random.Generator,
    min_count: int,
    max_count: int,
    min_width: int,
    max_width: int,
) -> list[tuple[int, int]]:
    count = int(rng.integers(min_count, max_count + 1))
    regions: list[tuple[int, int]] = []
    for _ in range(count):
        width = int(rng.integers(min_width, max_width + 1))
        start = int(rng.integers(60, SIGNAL_LENGTH - width - 40))
        regions.append((start, start + width))
    return sorted(regions)


def sample_mixture(rng: np.random.Generator, target_frequency: float) -> tuple[list[float], list[np.ndarray]]:
    count = int(rng.integers(2, 5))
    overlap = target_frequency + rng.normal(loc=0.0, scale=3.0, size=2)
    candidate_pool = np.concatenate([MIXTURE_POOL, overlap]).astype(np.float32)
    candidate_pool = candidate_pool[(candidate_pool > 6.0) & (candidate_pool < 60.0)]
    freqs = [float(freq) for freq in rng.choice(candidate_pool, size=count, replace=False)]
    waves = [
        drifting_sinusoid(
            time_axis(),
            base_frequency=freq,
            amplitude=float(rng.uniform(0.08, 0.24)),
            phase=float(rng.uniform(0.0, 2.0 * np.pi)),
            drift_strength=float(rng.uniform(0.1, 1.2)),
            drift_rate=float(rng.uniform(0.03, 0.18)),
        )
        for freq in freqs
    ]
    return freqs, waves


def simulate_sample(label_name: str, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    t = time_axis()
    noise_level = float(rng.uniform(0.06, 0.32))
    noise_beta = float(rng.uniform(0.4, 1.4))
    base_amplitude = float(rng.uniform(0.85, 1.15))
    base_phase = float(rng.uniform(0.0, 2.0 * np.pi))
    base_drift = float(rng.uniform(0.15, 0.8))
    base_signal = drifting_sinusoid(
        t,
        base_frequency=5.0,
        amplitude=base_amplitude,
        phase=base_phase,
        drift_strength=base_drift,
        drift_rate=float(rng.uniform(0.02, 0.08)),
    )
    primary_freq = 5.0
    components = [5.0]
    anomaly_regions: list[tuple[int, int]] = []
    anomaly_wave = np.zeros(SIGNAL_LENGTH, dtype=np.float32)
    anomaly_amplitude = 0.0
    anomaly_drift = 0.0

    if label_name == "fault_light":
        primary_freq = 20.0
        components.append(primary_freq)
        anomaly_regions = sample_regions(rng, min_count=2, max_count=3, min_width=140, max_width=260)
        local_mask = burst_mask(SIGNAL_LENGTH, anomaly_regions)
        modulation = np.clip(0.35 + 0.85 * local_mask, 0.0, 1.2)
        anomaly_amplitude = float(rng.uniform(0.28, 0.52))
        anomaly_drift = float(rng.uniform(0.8, 2.2))
        anomaly_wave = drifting_sinusoid(
            t,
            base_frequency=primary_freq,
            amplitude=anomaly_amplitude,
            phase=float(rng.uniform(0.0, 2.0 * np.pi)),
            drift_strength=anomaly_drift,
            drift_rate=float(rng.uniform(0.04, 0.14)),
            modulation=modulation,
        )
    elif label_name == "fault_heavy":
        primary_freq = 50.0
        components.append(primary_freq)
        anomaly_regions = sample_regions(rng, min_count=2, max_count=4, min_width=45, max_width=120)
        local_mask = burst_mask(SIGNAL_LENGTH, anomaly_regions)
        anomaly_amplitude = float(rng.uniform(1.0, 1.65))
        anomaly_drift = float(rng.uniform(1.8, 3.8))
        anomaly_wave = drifting_sinusoid(
            t,
            base_frequency=primary_freq,
            amplitude=anomaly_amplitude,
            phase=float(rng.uniform(0.0, 2.0 * np.pi)),
            drift_strength=anomaly_drift,
            drift_rate=float(rng.uniform(0.05, 0.18)),
            modulation=local_mask,
        )

    mixture_freqs, mixture_waves = sample_mixture(rng, target_frequency=primary_freq)
    signal = base_signal.copy()
    for wave in mixture_waves:
        signal += wave
    components.extend(mixture_freqs)
    noise = noise_level * colored_noise(rng, beta=noise_beta, length=SIGNAL_LENGTH)
    signal += noise
    counterfactual_signal = signal.copy()
    signal += anomaly_wave

    region_start = min(start for start, _ in anomaly_regions) if anomaly_regions else np.nan
    region_end = max(end for _, end in anomaly_regions) if anomaly_regions else np.nan
    metadata = {
        "label": label_name,
        "label_id": LABEL_TO_INDEX[label_name],
        "freq": primary_freq,
        "noise": noise_level,
        "noise_beta": noise_beta,
        "drift_strength": anomaly_drift if label_name != "normal" else base_drift,
        "anomaly_region": format_regions(anomaly_regions),
        "anomaly_regions": format_regions(anomaly_regions),
        "region_start": region_start,
        "region_end": region_end,
        "num_regions": len(anomaly_regions),
        "anomaly_amplitude": anomaly_amplitude,
        "components": ";".join(str(int(freq)) for freq in sorted(set(components))),
        "sample_rate": SAMPLE_RATE,
    }
    return signal.astype(np.float32), counterfactual_signal.astype(np.float32), metadata


def generate_dataset(samples_per_class: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    signals: list[np.ndarray] = []
    counterfactuals: list[np.ndarray] = []
    labels: list[int] = []
    metadata_rows: list[dict[str, object]] = []

    for label_name in CLASS_NAMES:
        for _ in range(samples_per_class):
            signal, counterfactual, metadata = simulate_sample(label_name, rng)
            metadata["sample_id"] = len(signals)
            signals.append(signal)
            counterfactuals.append(counterfactual)
            labels.append(int(metadata["label_id"]))
            metadata_rows.append(metadata)

    metadata = pd.DataFrame(metadata_rows).sort_values("sample_id").reset_index(drop=True)
    return (
        np.stack(signals),
        np.stack(counterfactuals),
        np.array(labels, dtype=np.int64),
        metadata,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate physics-based time-series signals.")
    parser.add_argument("--samples-per-class", type=int, default=SAMPLES_PER_CLASS)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    signals, counterfactuals, labels, metadata = generate_dataset(args.samples_per_class, args.seed)
    np.save(DATA_DIR / "signals.npy", signals)
    np.save(DATA_DIR / "counterfactuals.npy", counterfactuals)
    np.save(DATA_DIR / "labels.npy", labels)
    metadata.to_csv(DATA_DIR / "metadata.csv", index=False)


if __name__ == "__main__":
    main()
