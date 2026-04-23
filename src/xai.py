from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from captum.attr import IntegratedGradients, NoiseTunnel, Saliency

from src import CLASS_NAMES
from src.features import (
    build_anomaly_mask,
    compute_fft_magnitude,
    extract_dominant_frequency,
    frequency_axis,
    load_counterfactuals,
    load_dataset,
    prepare_model_inputs,
)
from src.metrics import (
    attribution_ablation_score,
    attribution_weighted_frequency,
    causal_score,
    counterfactual_consistency,
    frequency_alignment_score,
    physical_consistency_score,
    physics_violation_score,
    stability_score,
    temporal_coherence,
    temporal_smearing_score,
)
from src.model import PhysicsGuidedCNN, set_seed

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
PLOT_DIR = OUTPUT_DIR / "plots"
SEED = 42
DEVICE = "cpu"
SMOOTHING_WINDOW = 21
METHOD_NAMES = ("integrated_gradients", "saliency", "smoothgrad")
METHOD_COLORS = {
    "integrated_gradients": "#2563eb",
    "saliency": "#dc2626",
    "smoothgrad": "#16a34a",
}


def smooth_attribution(attribution: np.ndarray, window: int = SMOOTHING_WINDOW) -> np.ndarray:
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(attribution, kernel, mode="same")


def normalized_trace(values: np.ndarray) -> np.ndarray:
    values = np.abs(values)
    return values / (values.max() + 1e-8)


def compute_method_attribution(
    method: str,
    explainers: dict[str, object],
    series: torch.Tensor,
    dominant: torch.Tensor,
    target: int,
    ig_steps: int,
    smoothgrad_samples: int,
    smoothgrad_stdev: float,
) -> np.ndarray:
    if method == "integrated_gradients":
        attr = explainers[method].attribute(
            inputs=series,
            baselines=torch.zeros_like(series),
            target=target,
            additional_forward_args=dominant,
            n_steps=ig_steps,
        )
    elif method == "saliency":
        attr = explainers[method].attribute(inputs=series, target=target, additional_forward_args=dominant)
    else:
        attr = explainers[method].attribute(
            inputs=series,
            target=target,
            additional_forward_args=dominant,
            nt_type="smoothgrad",
            nt_samples=smoothgrad_samples,
            stdevs=smoothgrad_stdev,
        )
    return attr.detach().cpu().numpy()[0, 0].astype(np.float32)


def predict_signal(model: torch.nn.Module, signal: np.ndarray, sample_rate: float) -> tuple[np.ndarray, int]:
    feature_bank = prepare_model_inputs(signal[None, :], sample_rate=sample_rate)
    series = torch.from_numpy(feature_bank["series"]).to(DEVICE)
    dominant = torch.from_numpy(feature_bank["dominant_frequency_normalized"]).to(DEVICE)
    with torch.no_grad():
        logits = model(series, dominant)
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]
    return probabilities.astype(np.float32), int(np.argmax(probabilities))


def topk_ablation(signal: np.ndarray, counterfactual_signal: np.ndarray, attribution: np.ndarray, top_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    top_k = max(1, int(round(signal.shape[0] * top_fraction)))
    top_indices = np.argsort(np.abs(attribution))[-top_k:]
    ablated = signal.copy()
    ablated[top_indices] = counterfactual_signal[top_indices]
    return ablated.astype(np.float32), np.sort(top_indices)


def plot_overlay(
    sample_id: int,
    signal: np.ndarray,
    method_attributions: dict[str, np.ndarray],
    anomaly_mask: np.ndarray,
    sample_rate: float,
) -> None:
    t = np.arange(signal.shape[0], dtype=np.float32) / sample_rate
    heatmap = np.stack([normalized_trace(method_attributions[method]) for method in METHOD_NAMES], axis=0)
    signal_span = np.max(np.abs(signal)) + 1e-8

    fig = plt.figure(figsize=(11, 6), constrained_layout=True)
    grid = fig.add_gridspec(2, 1, height_ratios=[1.0, 2.0], hspace=0.18)
    ax_top = fig.add_subplot(grid[0, 0])
    ax_bottom = fig.add_subplot(grid[1, 0], sharex=ax_top)

    image = ax_top.imshow(
        heatmap,
        aspect="auto",
        cmap="turbo",
        extent=[t[0], t[-1], 0, len(METHOD_NAMES)],
        origin="lower",
        vmin=0.0,
        vmax=1.0,
    )
    ax_top.set_yticks(np.arange(len(METHOD_NAMES)) + 0.5, ["IG", "Saliency", "SmoothGrad"])
    ax_top.set_ylabel("method")
    ax_top.set_title(f"Temporal saliency map and signal overlay: sample {sample_id}")

    if anomaly_mask.sum() > 0:
        in_region = np.where(anomaly_mask > 0)[0]
        splits = np.split(in_region, np.where(np.diff(in_region) != 1)[0] + 1)
        for segment in splits:
            ax_top.axvspan(t[segment[0]], t[segment[-1]], facecolor="none", edgecolor="black", linestyle="--", linewidth=1.1)
            ax_bottom.axvspan(t[segment[0]], t[segment[-1]], facecolor="#fde68a", alpha=0.18, edgecolor="none")

    colorbar = fig.colorbar(image, ax=ax_top, fraction=0.046, pad=0.02)
    colorbar.set_label("weight")

    ax_bottom.plot(t, signal, color="#334155", linewidth=1.1, label="signal")
    for method in METHOD_NAMES:
        scaled = normalized_trace(method_attributions[method]) * signal_span
        ax_bottom.fill_between(t, 0.0, scaled, color=METHOD_COLORS[method], alpha=0.20, label=method.replace("_", " "))
    ax_bottom.set_xlabel("Time [s]")
    ax_bottom.set_ylabel("Amplitude")
    ax_bottom.legend(loc="upper right", ncol=2)
    fig.savefig(PLOT_DIR / f"overlay_sample_{sample_id}.png", dpi=180)
    plt.close(fig)


def plot_frequency_alignment(
    sample_id: int,
    signal: np.ndarray,
    method_attributions: dict[str, np.ndarray],
    true_frequency: float,
    sample_rate: float,
) -> None:
    t = np.arange(signal.shape[0], dtype=np.float32) / sample_rate
    freqs = frequency_axis(signal.shape[0], sample_rate)
    fft_signal = normalized_trace(compute_fft_magnitude(signal))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    axes[0].plot(t, signal, color="#334155", linewidth=1.1)
    axes[0].set_title(f"Observed signal: sample {sample_id}")
    axes[0].set_xlabel("Time [s]")
    axes[0].set_ylabel("Amplitude")

    axes[1].plot(freqs, fft_signal, color="#0f172a", linewidth=1.2, label="FFT")
    for method in METHOD_NAMES:
        weighted_fft = normalized_trace(compute_fft_magnitude((signal - signal.mean()) * np.abs(method_attributions[method])))
        axes[1].plot(freqs, weighted_fft, color=METHOD_COLORS[method], linewidth=1.2, alpha=0.9, label=method.replace("_", " "))
    axes[1].axvline(true_frequency, color="#7c3aed", linestyle="--", linewidth=1.0, label="true freq")
    axes[1].set_xlim(0.0, min(sample_rate / 2.0, 80.0))
    axes[1].set_xlabel("Frequency [Hz]")
    axes[1].set_ylabel("Normalized magnitude")
    axes[1].set_title("FFT vs attribution-weighted spectra")
    axes[1].legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / f"fft_alignment_sample_{sample_id}.png", dpi=180)
    plt.close(fig)


def select_examples(metrics: pd.DataFrame) -> list[int]:
    chosen = []
    for class_name in CLASS_NAMES:
        subset = metrics[metrics["class_name"] == class_name]
        if class_name == "fault_heavy":
            ranked = subset.sort_values(["causal_score", "physics_violation_score"], ascending=[True, False])
        else:
            ranked = subset.sort_values(["causal_score", "attribution_score"], ascending=[False, False])
        chosen.extend(ranked["sample_id"].drop_duplicates().head(1).astype(int).tolist())
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate XAI maps and causal metrics.")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--ig-steps", type=int, default=32)
    parser.add_argument("--smoothgrad-samples", type=int, default=6)
    parser.add_argument("--smoothgrad-stdev", type=float, default=0.05)
    parser.add_argument("--ablation-fraction", type=float, default=0.10)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    for pattern in ("overlay_sample_*.png", "fft_alignment_sample_*.png"):
        for path in PLOT_DIR.glob(pattern):
            path.unlink()

    checkpoint = torch.load(OUTPUT_DIR / "model.pt", map_location=DEVICE, weights_only=False)
    signals, labels, metadata = load_dataset()
    counterfactuals = load_counterfactuals()
    sample_rate = float(checkpoint["sample_rate"])
    features = prepare_model_inputs(signals, sample_rate=sample_rate)

    model = PhysicsGuidedCNN(num_classes=len(CLASS_NAMES)).to(DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    explainers = {
        "integrated_gradients": IntegratedGradients(model),
        "saliency": Saliency(model),
        "smoothgrad": NoiseTunnel(Saliency(model)),
    }

    predictions = pd.read_csv(OUTPUT_DIR / "test_predictions.csv").sort_values("sample_id").reset_index(drop=True)
    metadata = metadata.set_index("sample_id")
    sample_positions = {int(sample_id): position for position, sample_id in enumerate(predictions["sample_id"].astype(int))}

    raw_maps = {method: [] for method in METHOD_NAMES}
    smooth_maps = {method: [] for method in METHOD_NAMES}
    perturbed_maps = {method: [] for method in METHOD_NAMES}
    counterfactual_maps = {method: [] for method in METHOD_NAMES}
    metric_rows = []

    for row in predictions.itertuples(index=False):
        sample_id = int(row.sample_id)
        signal = signals[sample_id]
        counterfactual_signal = counterfactuals[sample_id]
        series = torch.from_numpy(features["series"][sample_id : sample_id + 1]).to(DEVICE)
        dominant = torch.from_numpy(features["dominant_frequency_normalized"][sample_id : sample_id + 1]).to(DEVICE)
        target = int(row.pred_label)
        original_probability = float(getattr(row, f"prob_{CLASS_NAMES[target]}"))
        exclude_band = (4.0, 6.0) if metadata.loc[sample_id, "label"] != "normal" else None
        anomaly_mask = build_anomaly_mask(metadata.loc[sample_id, "anomaly_regions"], signal.shape[0])
        diagnostic_fft_frequency = extract_dominant_frequency(signal, sample_rate, exclude_band=exclude_band)
        counterfactual_probability, counterfactual_label = predict_signal(model, counterfactual_signal, sample_rate)

        perturb_rng = np.random.default_rng(args.seed + sample_id)
        perturb_scale = float(metadata.loc[sample_id, "noise"]) * 0.30 + 0.02
        perturbed_signal = signal + perturb_rng.normal(0.0, perturb_scale, size=signal.shape).astype(np.float32)
        perturbed_features = prepare_model_inputs(perturbed_signal[None, :], sample_rate=sample_rate)
        perturbed_series = torch.from_numpy(perturbed_features["series"]).to(DEVICE)
        perturbed_dominant = torch.from_numpy(perturbed_features["dominant_frequency_normalized"]).to(DEVICE)

        counterfactual_features = prepare_model_inputs(counterfactual_signal[None, :], sample_rate=sample_rate)
        counterfactual_series = torch.from_numpy(counterfactual_features["series"]).to(DEVICE)
        counterfactual_dominant = torch.from_numpy(counterfactual_features["dominant_frequency_normalized"]).to(DEVICE)

        for method in METHOD_NAMES:
            raw_attr = compute_method_attribution(
                method,
                explainers,
                series,
                dominant,
                target,
                args.ig_steps,
                args.smoothgrad_samples,
                args.smoothgrad_stdev,
            )
            smooth_attr = smooth_attribution(raw_attr)

            perturbed_attr = compute_method_attribution(
                method,
                explainers,
                perturbed_series,
                perturbed_dominant,
                target,
                args.ig_steps,
                args.smoothgrad_samples,
                args.smoothgrad_stdev,
            )
            smooth_perturbed = smooth_attribution(perturbed_attr)

            counterfactual_attr = compute_method_attribution(
                method,
                explainers,
                counterfactual_series,
                counterfactual_dominant,
                target,
                args.ig_steps,
                args.smoothgrad_samples,
                args.smoothgrad_stdev,
            )
            smooth_counterfactual = smooth_attribution(counterfactual_attr)

            attr_frequency = attribution_weighted_frequency(signal, smooth_attr, sample_rate, exclude_band=exclude_band)
            consistency = physical_consistency_score(smooth_attr, anomaly_mask)
            temporal_smearing = temporal_smearing_score(smooth_attr, anomaly_mask)
            alignment = frequency_alignment_score(diagnostic_fft_frequency, attr_frequency)
            attribution_score = frequency_alignment_score(float(metadata.loc[sample_id, "freq"]), attr_frequency)
            stability = stability_score(smooth_attr, smooth_perturbed)
            coherence = temporal_coherence(smooth_attr)

            ablated_signal, _ = topk_ablation(signal, counterfactual_signal, smooth_attr, top_fraction=args.ablation_fraction)
            ablated_probabilities, _ = predict_signal(model, ablated_signal, sample_rate)
            ablation_score = attribution_ablation_score(original_probability, float(ablated_probabilities[target]))

            if anomaly_mask.sum() > 0:
                cf_region_score = physical_consistency_score(smooth_counterfactual, anomaly_mask)
                region_drop = np.clip((consistency if not np.isnan(consistency) else 0.0) - (cf_region_score if not np.isnan(cf_region_score) else 0.0), 0.0, 1.0)
                cf_score = counterfactual_consistency(
                    original_probability,
                    float(counterfactual_probability[target]),
                    counterfactual_label,
                    expected_label=0,
                    attribution_region_drop=region_drop,
                )
            else:
                cf_score = float(counterfactual_label == target)

            causal = causal_score(ablation_score, cf_score)
            violation = physics_violation_score(consistency, attribution_score, temporal_smearing)

            raw_maps[method].append(raw_attr)
            smooth_maps[method].append(smooth_attr)
            perturbed_maps[method].append(smooth_perturbed)
            counterfactual_maps[method].append(smooth_counterfactual)

            metric_rows.append(
                {
                    "sample_id": sample_id,
                    "method": method,
                    "class_name": metadata.loc[sample_id, "label"],
                    "true_label": int(labels[sample_id]),
                    "pred_label": target,
                    "true_freq": float(metadata.loc[sample_id, "freq"]),
                    "fft_dominant_freq": float(row.fft_dominant_freq),
                    "diagnostic_fft_freq": diagnostic_fft_frequency,
                    "attribution_dominant_freq": attr_frequency,
                    "frequency_bias_hz": abs(attr_frequency - float(metadata.loc[sample_id, "freq"])),
                    "frequency_alignment_score": alignment,
                    "attribution_score": attribution_score,
                    "consistency_score": consistency,
                    "temporal_smearing": temporal_smearing,
                    "stability_score": stability,
                    "temporal_coherence": coherence,
                    "ablation_score": ablation_score,
                    "counterfactual_consistency": cf_score,
                    "causal_score": causal,
                    "physics_violation_score": violation,
                    "counterfactual_label": counterfactual_label,
                }
            )

    stacked_raw = {f"{method}_raw": np.stack(values) for method, values in raw_maps.items()}
    stacked_smooth = {f"{method}_smooth": np.stack(values) for method, values in smooth_maps.items()}
    stacked_perturbed = {f"{method}_perturbed": np.stack(values) for method, values in perturbed_maps.items()}
    stacked_counterfactual = {f"{method}_counterfactual": np.stack(values) for method, values in counterfactual_maps.items()}
    np.savez(OUTPUT_DIR / "attribution_maps.npz", **stacked_raw, **stacked_smooth, **stacked_perturbed, **stacked_counterfactual)
    np.save(OUTPUT_DIR / "attributions.npy", stacked_raw["integrated_gradients_raw"])
    np.save(OUTPUT_DIR / "smoothed_attributions.npy", stacked_smooth["integrated_gradients_smooth"])
    np.save(OUTPUT_DIR / "perturbed_attributions.npy", stacked_perturbed["integrated_gradients_perturbed"])

    metrics_frame = pd.DataFrame(metric_rows).sort_values(["method", "sample_id"]).reset_index(drop=True)
    metrics_frame.to_csv(OUTPUT_DIR / "xai_metrics.csv", index=False)

    selected_ids = select_examples(metrics_frame[metrics_frame["method"] == "integrated_gradients"])
    for sample_id in selected_ids:
        position = sample_positions[sample_id]
        method_attributions = {
            method: smooth_maps[method][position]
            for method in METHOD_NAMES
        }
        row = metadata.loc[sample_id]
        anomaly_mask = build_anomaly_mask(row["anomaly_regions"], signals[sample_id].shape[0])
        plot_overlay(sample_id, signals[sample_id], method_attributions, anomaly_mask, sample_rate)
        plot_frequency_alignment(sample_id, signals[sample_id], method_attributions, float(row["freq"]), sample_rate)

    pd.DataFrame({"sample_id": selected_ids}).to_csv(OUTPUT_DIR / "xai_selected_examples.csv", index=False)


if __name__ == "__main__":
    main()
