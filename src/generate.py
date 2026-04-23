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
MIXTURE_POOL = np.array([8.0, 12.0, 15.0, 30.0, 35.0, 45.0], dtype=np.float32)


def time_axis(length: int = SIGNAL_LENGTH, sample_rate: float = SAMPLE_RATE) -> np.ndarray:
    return np.arange(length, dtype=np.float32) / sample_rate


def sinusoid(
    t: np.ndarray,
    frequency: float,
    amplitude: float,
    phase: float,
    region: tuple[int, int] | None = None,
) -> np.ndarray:
    wave = amplitude * np.sin(2.0 * np.pi * frequency * t + phase)
    if region is None:
        return wave
    start, end = region
    envelope = np.zeros_like(t)
    envelope[start:end] = np.hanning(max(end - start, 2))
    return wave * envelope


def format_region(region: tuple[int, int] | None) -> str:
    if region is None:
        return "none"
    return f"{region[0]}:{region[1]}"


def sample_mixture(rng: np.random.Generator, exclude: float) -> tuple[list[float], list[np.ndarray]]:
    count = int(rng.integers(1, 3))
    freqs = [float(freq) for freq in rng.choice(MIXTURE_POOL[MIXTURE_POOL != exclude], size=count, replace=False)]
    waves = [
        sinusoid(
            time_axis(),
            frequency=freq,
            amplitude=float(rng.uniform(0.08, 0.22)),
            phase=float(rng.uniform(0.0, 2.0 * np.pi)),
        )
        for freq in freqs
    ]
    return freqs, waves


def simulate_sample(label_name: str, rng: np.random.Generator) -> tuple[np.ndarray, dict[str, object]]:
    t = time_axis()
    noise_level = float(rng.uniform(0.05, 0.35))
    signal = sinusoid(
        t,
        frequency=5.0,
        amplitude=float(rng.uniform(0.9, 1.1)),
        phase=float(rng.uniform(0.0, 2.0 * np.pi)),
    )
    anomaly_region: tuple[int, int] | None = None
    primary_freq = 5.0
    components = [5.0]

    if label_name == "fault_light":
        primary_freq = 20.0
        anomaly_region = (0, SIGNAL_LENGTH)
        components.append(primary_freq)
        signal += sinusoid(
            t,
            frequency=primary_freq,
            amplitude=float(rng.uniform(0.28, 0.52)),
            phase=float(rng.uniform(0.0, 2.0 * np.pi)),
        )
    elif label_name == "fault_heavy":
        primary_freq = 50.0
        width = int(rng.integers(90, 180))
        start = int(rng.integers(120, SIGNAL_LENGTH - width - 40))
        anomaly_region = (start, start + width)
        components.append(primary_freq)
        signal += sinusoid(
            t,
            frequency=primary_freq,
            amplitude=float(rng.uniform(1.2, 1.8)),
            phase=float(rng.uniform(0.0, 2.0 * np.pi)),
            region=anomaly_region,
        )

    mixture_freqs, mixture_waves = sample_mixture(rng, exclude=primary_freq)
    for wave in mixture_waves:
        signal += wave
    components.extend(mixture_freqs)
    signal += rng.normal(loc=0.0, scale=noise_level, size=SIGNAL_LENGTH)

    region_start = anomaly_region[0] if anomaly_region else np.nan
    region_end = anomaly_region[1] if anomaly_region else np.nan
    metadata = {
        "label": label_name,
        "label_id": LABEL_TO_INDEX[label_name],
        "freq": primary_freq,
        "noise": noise_level,
        "anomaly_region": format_region(anomaly_region),
        "region_start": region_start,
        "region_end": region_end,
        "components": ";".join(str(int(freq)) for freq in sorted(set(components))),
        "sample_rate": SAMPLE_RATE,
    }
    return signal.astype(np.float32), metadata


def generate_dataset(samples_per_class: int, seed: int) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    signals: list[np.ndarray] = []
    labels: list[int] = []
    metadata_rows: list[dict[str, object]] = []

    for label_name in CLASS_NAMES:
        for _ in range(samples_per_class):
            signal, metadata = simulate_sample(label_name, rng)
            metadata["sample_id"] = len(signals)
            signals.append(signal)
            labels.append(int(metadata["label_id"]))
            metadata_rows.append(metadata)

    metadata = pd.DataFrame(metadata_rows).sort_values("sample_id").reset_index(drop=True)
    return np.stack(signals), np.array(labels, dtype=np.int64), metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate physics-based time-series signals.")
    parser.add_argument("--samples-per-class", type=int, default=SAMPLES_PER_CLASS)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    signals, labels, metadata = generate_dataset(args.samples_per_class, args.seed)
    np.save(DATA_DIR / "signals.npy", signals)
    np.save(DATA_DIR / "labels.npy", labels)
    metadata.to_csv(DATA_DIR / "metadata.csv", index=False)


if __name__ == "__main__":
    main()
