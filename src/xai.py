from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from captum.attr import IntegratedGradients

from src import CLASS_NAMES
from src.features import (
    compute_fft_magnitude,
    extract_dominant_frequency,
    frequency_axis,
    load_dataset,
    prepare_model_inputs,
)
from src.metrics import (
    attribution_weighted_frequency,
    frequency_alignment_score,
    physical_consistency_score,
    stability_score,
    temporal_coherence,
)
from src.model import PhysicsGuidedCNN, set_seed

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
PLOT_DIR = OUTPUT_DIR / "plots"
SEED = 42
DEVICE = "cpu"
SMOOTHING_WINDOW = 31


def smooth_attribution(attribution: np.ndarray, window: int = SMOOTHING_WINDOW) -> np.ndarray:
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(attribution, kernel, mode="same")


def normalized_trace(values: np.ndarray) -> np.ndarray:
    values = np.abs(values)
    return values / (values.max() + 1e-8)


def plot_overlay(sample_id: int, signal: np.ndarray, attribution: np.ndarray, metadata_row: pd.Series, sample_rate: float) -> None:
    t = np.arange(signal.shape[0], dtype=np.float32) / sample_rate
    span = np.max(np.abs(signal)) + 1e-8
    scaled_attr = normalized_trace(attribution) * span
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, signal, color="#1f2937", linewidth=1.1, label="signal")
    ax.fill_between(t, 0.0, scaled_attr, color="#ef4444", alpha=0.3, label="smoothed attribution")
    if pd.notna(metadata_row["region_start"]) and pd.notna(metadata_row["region_end"]):
        ax.axvspan(
            float(metadata_row["region_start"]) / sample_rate,
            float(metadata_row["region_end"]) / sample_rate,
            color="#fde68a",
            alpha=0.35,
            label="anomaly region",
        )
    ax.set_title(f"Signal and attribution overlay: sample {sample_id} ({metadata_row['label']})")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Amplitude")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / f"overlay_sample_{sample_id}.png", dpi=180)
    plt.close(fig)


def plot_frequency_alignment(sample_id: int, signal: np.ndarray, attribution: np.ndarray, metadata_row: pd.Series, sample_rate: float) -> None:
    freqs = frequency_axis(signal.shape[0], sample_rate)
    fft_signal = normalized_trace(compute_fft_magnitude(signal))
    weighted_signal = normalized_trace(compute_fft_magnitude((signal - signal.mean()) * np.abs(attribution)))
    exclude_band = (4.0, 6.0) if metadata_row["label"] != "normal" else None
    attr_freq = attribution_weighted_frequency(signal, attribution, sample_rate, exclude_band=exclude_band)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(freqs, fft_signal, color="#2563eb", linewidth=1.2, label="FFT magnitude")
    ax.plot(freqs, weighted_signal, color="#dc2626", linewidth=1.2, label="attribution-weighted FFT")
    ax.axvline(float(metadata_row["freq"]), color="#16a34a", linestyle="--", linewidth=1.0, label="true freq")
    ax.axvline(attr_freq, color="#7c3aed", linestyle=":", linewidth=1.0, label="attribution freq")
    ax.set_xlim(0.0, min(sample_rate / 2.0, 80.0))
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Normalized magnitude")
    ax.set_title(f"FFT vs attribution-weighted spectrum: sample {sample_id}")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / f"fft_alignment_sample_{sample_id}.png", dpi=180)
    plt.close(fig)


def select_examples(predictions: pd.DataFrame) -> list[int]:
    selected: list[int] = []
    for class_name in CLASS_NAMES:
        subset = predictions[predictions["label"] == class_name]
        correct = subset[subset["predicted_label_name"] == class_name]
        chosen = correct if not correct.empty else subset
        selected.extend(chosen["sample_id"].head(2).astype(int).tolist())
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Captum attributions and physics metrics.")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--ig-steps", type=int, default=48)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    checkpoint = torch.load(OUTPUT_DIR / "model.pt", map_location=DEVICE, weights_only=False)
    signals, labels, metadata = load_dataset()
    sample_rate = float(checkpoint["sample_rate"])
    features = prepare_model_inputs(signals, sample_rate=sample_rate)

    model = PhysicsGuidedCNN(num_classes=len(CLASS_NAMES)).to(DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    ig = IntegratedGradients(model)

    predictions = pd.read_csv(OUTPUT_DIR / "test_predictions.csv").sort_values("sample_id").reset_index(drop=True)
    metadata = metadata.set_index("sample_id")

    attributions = []
    smoothed = []
    perturbed_smoothed = []
    metric_rows = []

    for row in predictions.itertuples(index=False):
        sample_id = int(row.sample_id)
        signal = signals[sample_id]
        series = torch.from_numpy(features["series"][sample_id : sample_id + 1]).to(DEVICE)
        dominant = torch.from_numpy(features["dominant_frequency_normalized"][sample_id : sample_id + 1]).to(DEVICE)
        target = int(row.pred_label)
        attr_tensor = ig.attribute(
            inputs=series,
            baselines=torch.zeros_like(series),
            target=target,
            additional_forward_args=dominant,
            n_steps=args.ig_steps,
        )
        raw_attr = attr_tensor.detach().cpu().numpy()[0, 0]
        smooth_attr = smooth_attribution(raw_attr)

        perturb_rng = np.random.default_rng(args.seed + sample_id)
        perturb_scale = float(metadata.loc[sample_id, "noise"]) * 0.35 + 0.02
        perturbed_signal = signal + perturb_rng.normal(0.0, perturb_scale, size=signal.shape).astype(np.float32)
        perturbed_features = prepare_model_inputs(perturbed_signal[None, :], sample_rate=sample_rate)
        perturbed_series = torch.from_numpy(perturbed_features["series"]).to(DEVICE)
        perturbed_dom = torch.from_numpy(perturbed_features["dominant_frequency_normalized"]).to(DEVICE)
        perturbed_attr_tensor = ig.attribute(
            inputs=perturbed_series,
            baselines=torch.zeros_like(perturbed_series),
            target=target,
            additional_forward_args=perturbed_dom,
            n_steps=args.ig_steps,
        )
        perturbed_attr = smooth_attribution(perturbed_attr_tensor.detach().cpu().numpy()[0, 0])

        exclude_band = (4.0, 6.0) if metadata.loc[sample_id, "label"] != "normal" else None
        diagnostic_fft_frequency = extract_dominant_frequency(signal, sample_rate, exclude_band=exclude_band)
        attr_frequency = attribution_weighted_frequency(signal, smooth_attr, sample_rate, exclude_band=exclude_band)
        consistency = physical_consistency_score(
            smooth_attr,
            float(metadata.loc[sample_id, "region_start"]),
            float(metadata.loc[sample_id, "region_end"]),
        )
        alignment = frequency_alignment_score(diagnostic_fft_frequency, attr_frequency)
        attribution_score = frequency_alignment_score(float(metadata.loc[sample_id, "freq"]), attr_frequency)
        stability = stability_score(smooth_attr, perturbed_attr)
        coherence = temporal_coherence(smooth_attr)

        attributions.append(raw_attr.astype(np.float32))
        smoothed.append(smooth_attr.astype(np.float32))
        perturbed_smoothed.append(perturbed_attr.astype(np.float32))
        metric_rows.append(
            {
                "sample_id": sample_id,
                "class_name": metadata.loc[sample_id, "label"],
                "true_label": int(labels[sample_id]),
                "pred_label": target,
                "true_freq": float(metadata.loc[sample_id, "freq"]),
                "fft_dominant_freq": float(row.fft_dominant_freq),
                "diagnostic_fft_freq": diagnostic_fft_frequency,
                "attribution_dominant_freq": attr_frequency,
                "frequency_alignment_score": alignment,
                "attribution_score": attribution_score,
                "consistency_score": consistency,
                "stability_score": stability,
                "temporal_coherence": coherence,
            }
        )

    np.save(OUTPUT_DIR / "attributions.npy", np.stack(attributions))
    np.save(OUTPUT_DIR / "smoothed_attributions.npy", np.stack(smoothed))
    np.save(OUTPUT_DIR / "perturbed_attributions.npy", np.stack(perturbed_smoothed))
    metrics_frame = pd.DataFrame(metric_rows)
    metrics_frame.to_csv(OUTPUT_DIR / "xai_metrics.csv", index=False)

    selected_ids = select_examples(predictions)
    for sample_id in selected_ids:
        row = metadata.loc[sample_id]
        smooth_attr = np.stack(smoothed)[predictions.index[predictions["sample_id"] == sample_id][0]]
        plot_overlay(sample_id, signals[sample_id], smooth_attr, row, sample_rate)
        plot_frequency_alignment(sample_id, signals[sample_id], smooth_attr, row, sample_rate)

    pd.DataFrame({"sample_id": selected_ids}).to_csv(OUTPUT_DIR / "xai_selected_examples.csv", index=False)


if __name__ == "__main__":
    main()
