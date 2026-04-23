from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import CLASS_NAMES

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
PLOT_DIR = OUTPUT_DIR / "plots"
METHOD_COLORS = {
    "integrated_gradients": "#2563eb",
    "saliency": "#dc2626",
    "smoothgrad": "#16a34a",
}


def confusion_matrix(true_labels: np.ndarray, pred_labels: np.ndarray) -> np.ndarray:
    matrix = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=int)
    for true_label, pred_label in zip(true_labels, pred_labels):
        matrix[int(true_label), int(pred_label)] += 1
    return matrix


def markdown_table(frame: pd.DataFrame) -> str:
    rounded = frame.copy()
    for column in rounded.columns:
        if pd.api.types.is_numeric_dtype(rounded[column]):
            rounded[column] = rounded[column].map(lambda value: "" if pd.isna(value) else f"{value:.4f}")
    header = "| " + " | ".join(rounded.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(rounded.columns)) + " |"
    rows = ["| " + " | ".join(str(value) for value in row) + " |" for row in rounded.itertuples(index=False, name=None)]
    return "\n".join([header, separator, *rows])


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


def plot_method_comparison(method_summary: pd.DataFrame) -> None:
    metrics = [
        ("attribution_score", "physics alignment"),
        ("consistency_score", "consistency"),
        ("ablation_score", "ablation"),
        ("counterfactual_consistency", "counterfactual"),
        ("causal_score", "causal"),
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(16, 3.8))
    for ax, (column, title) in zip(axes, metrics):
        colors = [METHOD_COLORS[method] for method in method_summary["method"]]
        ax.bar(method_summary["method"], method_summary[column], color=colors, alpha=0.9)
        ax.set_title(title)
        ax.set_ylim(0.0, 1.05)
        ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "xai_method_comparison.png", dpi=180)
    plt.close(fig)


def plot_fault_heavy_failures(failure_frame: pd.DataFrame) -> None:
    metrics = [
        ("frequency_bias_hz", "frequency bias [Hz]"),
        ("temporal_smearing", "temporal smearing"),
        ("physics_violation_score", "physics violation"),
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(12, 3.8))
    for ax, (column, title) in zip(axes, metrics):
        colors = [METHOD_COLORS[method] for method in failure_frame["method"]]
        ax.bar(failure_frame["method"], failure_frame[column], color=colors, alpha=0.9)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "fault_heavy_failure_modes.png", dpi=180)
    plt.close(fig)


def generate_insights(
    predictions: pd.DataFrame,
    method_summary: pd.DataFrame,
    method_class_summary: pd.DataFrame,
) -> str:
    accuracy = float((predictions["true_label"] == predictions["pred_label"]).mean())
    best_method = method_summary.sort_values("causal_score", ascending=False).iloc[0]
    heavy = method_class_summary[method_class_summary["class_name"] == "fault_heavy"].sort_values("causal_score", ascending=False)
    heavy_best = heavy.iloc[0]
    heavy_worst = heavy.iloc[-1]
    light = method_class_summary[method_class_summary["class_name"] == "fault_light"].sort_values("attribution_score", ascending=False).iloc[0]

    failure_reason = []
    if heavy_best["frequency_bias_hz"] > 8.0:
        failure_reason.append("persistent frequency bias away from the injected burst frequency")
    if heavy_best["temporal_smearing"] > 0.65:
        failure_reason.append("temporal smearing outside the annotated burst windows")
    if heavy_best["consistency_score"] < 0.35:
        failure_reason.append("weak localization inside the anomaly regions")
    failure_clause = ", ".join(failure_reason) if failure_reason else "mixed spectral-temporal attribution errors"

    return "\n".join(
        [
            f"Held-out accuracy remains {accuracy:.3f}, so predictive performance is no longer the bottleneck.",
            f"Across XAI methods, {best_method['method']} has the highest mean causal score ({best_method['causal_score']:.3f}).",
            f"For fault_light, {light['method']} best preserves the physical 20 Hz signature with attribution score {light['attribution_score']:.3f}.",
            f"Fault_heavy remains the hard case. Even the strongest method ({heavy_best['method']}) shows {failure_clause}.",
            f"The weakest heavy-fault method is {heavy_worst['method']}, with physics violation score {heavy_worst['physics_violation_score']:.3f}.",
            "The upgraded benchmark therefore supports a stronger conclusion: correct classification can coexist with causally weak and physically inconsistent explanations, especially under non-stationary burst faults.",
        ]
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    for stale_name in ("class_average_attribution_overlay.png", "class_average_fft_alignment.png"):
        stale_path = PLOT_DIR / stale_name
        if stale_path.exists():
            stale_path.unlink()

    predictions = pd.read_csv(OUTPUT_DIR / "test_predictions.csv").sort_values("sample_id").reset_index(drop=True)
    metrics = pd.read_csv(OUTPUT_DIR / "xai_metrics.csv").sort_values(["method", "sample_id"]).reset_index(drop=True)

    matrix = confusion_matrix(predictions["true_label"].to_numpy(), predictions["pred_label"].to_numpy())
    pd.DataFrame(matrix, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(OUTPUT_DIR / "confusion_matrix.csv")
    plot_confusion(matrix)

    class_frequency_table = (
        metrics.groupby(["method", "class_name"], as_index=False)
        .agg(
            true_freq=("true_freq", "mean"),
            predicted_freq=("attribution_dominant_freq", "mean"),
            attribution_score=("attribution_score", "mean"),
            consistency_score=("consistency_score", "mean"),
        )
        .rename(columns={"class_name": "class"})
    )
    class_frequency_table.to_csv(OUTPUT_DIR / "class_frequency_table.csv", index=False)
    (OUTPUT_DIR / "class_frequency_table.md").write_text(markdown_table(class_frequency_table) + "\n", encoding="utf-8")

    method_summary = (
        metrics.groupby("method", as_index=False)
        .agg(
            attribution_score=("attribution_score", "mean"),
            consistency_score=("consistency_score", "mean"),
            ablation_score=("ablation_score", "mean"),
            counterfactual_consistency=("counterfactual_consistency", "mean"),
            causal_score=("causal_score", "mean"),
            physics_violation_score=("physics_violation_score", "mean"),
            frequency_bias_hz=("frequency_bias_hz", "mean"),
            temporal_smearing=("temporal_smearing", "mean"),
        )
        .sort_values("causal_score", ascending=False)
        .reset_index(drop=True)
    )
    method_summary.to_csv(OUTPUT_DIR / "xai_method_summary.csv", index=False)

    method_class_summary = (
        metrics.groupby(["method", "class_name"], as_index=False)
        .agg(
            attribution_score=("attribution_score", "mean"),
            consistency_score=("consistency_score", "mean"),
            ablation_score=("ablation_score", "mean"),
            counterfactual_consistency=("counterfactual_consistency", "mean"),
            causal_score=("causal_score", "mean"),
            physics_violation_score=("physics_violation_score", "mean"),
            frequency_bias_hz=("frequency_bias_hz", "mean"),
            temporal_smearing=("temporal_smearing", "mean"),
        )
        .sort_values(["class_name", "causal_score"], ascending=[True, False])
        .reset_index(drop=True)
    )
    method_class_summary.to_csv(OUTPUT_DIR / "xai_method_class_summary.csv", index=False)

    plot_method_comparison(method_summary)
    plot_fault_heavy_failures(method_class_summary[method_class_summary["class_name"] == "fault_heavy"])

    insight_text = generate_insights(predictions, method_summary, method_class_summary)
    (OUTPUT_DIR / "insights.txt").write_text(insight_text + "\n", encoding="utf-8")

    summary = {
        "accuracy": float((predictions["true_label"] == predictions["pred_label"]).mean()),
        "best_method": method_summary.iloc[0]["method"],
        "best_causal_score": float(method_summary.iloc[0]["causal_score"]),
        "method_summary": method_summary.to_dict(orient="records"),
        "method_class_summary": method_class_summary.to_dict(orient="records"),
    }
    with open(OUTPUT_DIR / "analysis_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


if __name__ == "__main__":
    main()
