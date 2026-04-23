from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import CLASS_NAMES
from src.features import compute_fft_magnitude, frequency_axis, load_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
PLOT_DIR = OUTPUT_DIR / "plots"


def confusion_matrix(true_labels: np.ndarray, pred_labels: np.ndarray) -> np.ndarray:
    matrix = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=int)
    for true_label, pred_label in zip(true_labels, pred_labels):
        matrix[int(true_label), int(pred_label)] += 1
    return matrix


def plot_confusion(matrix: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(CLASS_NAMES)), CLASS_NAMES, rotation=20)
    ax.set_yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix")
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(col, row, str(matrix[row, col]), ha="center", va="center", color="#111827")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "confusion_matrix.png", dpi=180)
    plt.close(fig)


def normalized(values: np.ndarray) -> np.ndarray:
    values = np.abs(values)
    return values / (values.max() + 1e-8)


def plot_average_attributions(signals: np.ndarray, predictions: pd.DataFrame, metadata: pd.DataFrame, attributions: np.ndarray, sample_rate: float) -> None:
    t = np.arange(signals.shape[-1], dtype=np.float32) / sample_rate
    fig, axes = plt.subplots(len(CLASS_NAMES), 1, figsize=(11, 9), sharex=True)
    for axis, class_name in zip(axes, CLASS_NAMES):
        class_rows = predictions[predictions["label"] == class_name]
        sample_ids = class_rows["sample_id"].astype(int).to_numpy()
        sample_positions = class_rows.index.to_numpy()
        mean_signal = signals[sample_ids].mean(axis=0)
        mean_attr = normalized(attributions[sample_positions].mean(axis=0))
        axis.plot(t, mean_signal, color="#1f2937", linewidth=1.1, label="mean signal")
        axis.fill_between(t, 0.0, mean_attr * (np.max(np.abs(mean_signal)) + 1e-8), color="#ef4444", alpha=0.28, label="mean attribution")
        if class_name == "fault_heavy":
            heavy_regions = metadata.loc[sample_ids, ["region_start", "region_end"]].dropna()
            if not heavy_regions.empty:
                axis.axvspan(
                    heavy_regions["region_start"].median() / sample_rate,
                    heavy_regions["region_end"].median() / sample_rate,
                    color="#fde68a",
                    alpha=0.25,
                    label="median anomaly window",
                )
        axis.set_title(class_name)
        axis.legend(loc="upper right")
    axes[-1].set_xlabel("Time [s]")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "class_average_attribution_overlay.png", dpi=180)
    plt.close(fig)


def plot_average_frequency(signals: np.ndarray, predictions: pd.DataFrame, attributions: np.ndarray, sample_rate: float) -> None:
    freqs = frequency_axis(signals.shape[-1], sample_rate)
    fig, axes = plt.subplots(len(CLASS_NAMES), 1, figsize=(11, 9), sharex=True)
    for axis, class_name in zip(axes, CLASS_NAMES):
        class_rows = predictions[predictions["label"] == class_name]
        sample_ids = class_rows["sample_id"].astype(int).to_numpy()
        attr_positions = class_rows.index.to_numpy()
        fft_stack = [normalized(compute_fft_magnitude(signals[sample_id])) for sample_id in sample_ids]
        weighted_stack = [
            normalized(compute_fft_magnitude((signals[sample_id] - signals[sample_id].mean()) * np.abs(attributions[pos])))
            for sample_id, pos in zip(sample_ids, attr_positions)
        ]
        axis.plot(freqs, np.mean(fft_stack, axis=0), color="#2563eb", linewidth=1.2, label="mean FFT")
        axis.plot(freqs, np.mean(weighted_stack, axis=0), color="#dc2626", linewidth=1.2, label="mean attribution FFT")
        axis.set_xlim(0.0, 80.0)
        axis.set_title(class_name)
        axis.legend(loc="upper right")
    axes[-1].set_xlabel("Frequency [Hz]")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "class_average_fft_alignment.png", dpi=180)
    plt.close(fig)


def generate_insights(predictions: pd.DataFrame, metrics: pd.DataFrame, summary_table: pd.DataFrame) -> str:
    accuracy = float((predictions["true_label"] == predictions["pred_label"]).mean())
    overall_alignment = float(metrics["frequency_alignment_score"].mean())
    overall_physics_alignment = float(metrics["attribution_score"].mean())
    overall_stability = float(metrics["stability_score"].mean())
    overall_coherence = float(metrics["temporal_coherence"].mean())
    heavy_consistency = summary_table.loc[summary_table["class"] == "fault_heavy", "consistency_score"].iloc[0]
    light_alignment = summary_table.loc[summary_table["class"] == "fault_light", "attribution_score"].iloc[0]

    focus_text = (
        "The CNN concentrates attribution inside the annotated burst region for fault_heavy signals."
        if heavy_consistency >= 0.60
        else "The CNN only partially concentrates attribution inside the annotated burst region for fault_heavy signals."
    )
    physics_text = (
        "Attribution-weighted spectra remain close to the physical signal frequencies, suggesting the explanations track the intended dynamics."
        if overall_physics_alignment >= 0.70
        else "Attribution-weighted spectra drift away from the physical signal frequencies, suggesting the explanations rely on non-physical correlations."
    )
    light_text = (
        "For fault_light, the model remains frequency-aligned even though the anomaly is global in time."
        if light_alignment >= 0.70
        else "For fault_light, the model is less frequency-aligned and may be using incidental correlations."
    )
    return "\n".join(
        [
            f"Classification accuracy on the held-out set is {accuracy:.3f}.",
            f"{focus_text} The mean heavy-fault consistency score is {heavy_consistency:.3f}.",
            f"{physics_text} The diagnostic FFT alignment score is {overall_alignment:.3f}, and the true-physics attribution score is {overall_physics_alignment:.3f}.",
            f"Attribution stability under additional noise is {overall_stability:.3f}, and temporal coherence is {overall_coherence:.3f}.",
            light_text,
            "Overall, the explanations can be tested against physics: localized transient faults demand temporal overlap, while spectral faults demand frequency agreement.",
        ]
    )


def markdown_table(frame: pd.DataFrame) -> str:
    rounded = frame.copy()
    for column in rounded.columns:
        if pd.api.types.is_numeric_dtype(rounded[column]):
            rounded[column] = rounded[column].map(lambda value: "" if pd.isna(value) else f"{value:.4f}")
    header = "| " + " | ".join(rounded.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(rounded.columns)) + " |"
    rows = ["| " + " | ".join(str(value) for value in row) + " |" for row in rounded.itertuples(index=False, name=None)]
    return "\n".join([header, separator, *rows])


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    signals, _, metadata = load_dataset()
    metadata = metadata.set_index("sample_id")
    sample_rate = float(metadata["sample_rate"].iloc[0])
    predictions = pd.read_csv(OUTPUT_DIR / "test_predictions.csv").sort_values("sample_id").reset_index(drop=True)
    metrics = pd.read_csv(OUTPUT_DIR / "xai_metrics.csv").sort_values("sample_id").reset_index(drop=True)
    attributions = np.load(OUTPUT_DIR / "smoothed_attributions.npy")

    matrix = confusion_matrix(predictions["true_label"].to_numpy(), predictions["pred_label"].to_numpy())
    pd.DataFrame(matrix, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(OUTPUT_DIR / "confusion_matrix.csv")
    plot_confusion(matrix)
    plot_average_attributions(signals, predictions, metadata, attributions, sample_rate)
    plot_average_frequency(signals, predictions, attributions, sample_rate)

    summary_table = (
        metrics.merge(predictions[["sample_id", "predicted_label_name"]], on="sample_id", how="left")
        .groupby("class_name", as_index=False)
        .agg(
            true_freq=("true_freq", "mean"),
            predicted_freq=("attribution_dominant_freq", "mean"),
            attribution_score=("attribution_score", "mean"),
            consistency_score=("consistency_score", "mean"),
        )
        .rename(columns={"class_name": "class"})
    )
    summary_table.to_csv(OUTPUT_DIR / "class_frequency_table.csv", index=False)
    (OUTPUT_DIR / "class_frequency_table.md").write_text(markdown_table(summary_table) + "\n", encoding="utf-8")

    insight_text = generate_insights(predictions, metrics, summary_table)
    (OUTPUT_DIR / "insights.txt").write_text(insight_text + "\n", encoding="utf-8")

    summary = {
        "accuracy": float((predictions["true_label"] == predictions["pred_label"]).mean()),
        "frequency_alignment": float(metrics["frequency_alignment_score"].mean()),
        "attribution_score": float(metrics["attribution_score"].mean()),
        "stability": float(metrics["stability_score"].mean()),
        "temporal_coherence": float(metrics["temporal_coherence"].mean()),
        "class_table": summary_table.to_dict(orient="records"),
    }
    with open(OUTPUT_DIR / "analysis_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


if __name__ == "__main__":
    main()
