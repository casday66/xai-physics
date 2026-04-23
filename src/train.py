from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src import CLASS_NAMES
from src.features import load_dataset, prepare_model_inputs
from src.model import PhysicsGuidedCNN, predict_probabilities, set_seed

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
SEED = 42
BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 1e-3
TEST_RATIO = 0.2
DEVICE = "cpu"


def split_indices(count: int, test_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = rng.permutation(count)
    test_count = int(round(count * test_ratio))
    test_idx = np.sort(indices[:test_count])
    train_idx = np.sort(indices[test_count:])
    return train_idx, test_idx


def make_loader(series: np.ndarray, dominant_frequency: np.ndarray, labels: np.ndarray, indices: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(series[indices]),
        torch.from_numpy(dominant_frequency[indices]),
        torch.from_numpy(labels[indices]),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def run_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer | None, criterion: nn.Module) -> tuple[float, float]:
    training = optimizer is not None
    model.train(mode=training)
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for series_batch, dom_batch, label_batch in loader:
        series_batch = series_batch.to(DEVICE)
        dom_batch = dom_batch.to(DEVICE)
        label_batch = label_batch.to(DEVICE)
        if training:
            optimizer.zero_grad(set_to_none=True)
        logits = model(series_batch, dom_batch)
        loss = criterion(logits, label_batch)
        if training:
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * label_batch.shape[0]
        total_correct += (logits.argmax(dim=1) == label_batch).sum().item()
        total_examples += label_batch.shape[0]

    return total_loss / total_examples, total_correct / total_examples


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the physics-guided CNN.")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    signals, labels, metadata = load_dataset()
    sample_rate = float(metadata["sample_rate"].iloc[0])
    feature_bank = prepare_model_inputs(signals, sample_rate=sample_rate)
    train_idx, test_idx = split_indices(len(signals), TEST_RATIO, seed=args.seed)
    np.savez(OUTPUT_DIR / "split_indices.npz", train_idx=train_idx, test_idx=test_idx)

    model = PhysicsGuidedCNN(num_classes=len(CLASS_NAMES)).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()

    train_loader = make_loader(
        feature_bank["series"],
        feature_bank["dominant_frequency_normalized"],
        labels,
        train_idx,
        args.batch_size,
        shuffle=True,
    )
    test_loader = make_loader(
        feature_bank["series"],
        feature_bank["dominant_frequency_normalized"],
        labels,
        test_idx,
        args.batch_size,
        shuffle=False,
    )

    history_rows = []
    best_accuracy = -np.inf
    best_state = None
    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = run_epoch(model, train_loader, optimizer, criterion)
        test_loss, test_accuracy = run_epoch(model, test_loader, None, criterion)
        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "test_loss": test_loss,
                "test_accuracy": test_accuracy,
            }
        )
        if test_accuracy >= best_accuracy:
            best_accuracy = test_accuracy
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    history = pd.DataFrame(history_rows)
    history.to_csv(OUTPUT_DIR / "train_history.csv", index=False)

    series_tensor = torch.from_numpy(feature_bank["series"][test_idx])
    dom_tensor = torch.from_numpy(feature_bank["dominant_frequency_normalized"][test_idx])
    probabilities = predict_probabilities(model, series_tensor, dom_tensor, device=DEVICE)
    predicted_labels = probabilities.argmax(axis=1)

    prediction_frame = metadata.iloc[test_idx][["sample_id", "label", "freq", "noise", "anomaly_region"]].copy()
    prediction_frame["true_label"] = labels[test_idx]
    prediction_frame["pred_label"] = predicted_labels
    prediction_frame["predicted_label_name"] = [CLASS_NAMES[idx] for idx in predicted_labels]
    prediction_frame["fft_dominant_freq"] = feature_bank["dominant_frequency"][test_idx]
    for class_index, class_name in enumerate(CLASS_NAMES):
        prediction_frame[f"prob_{class_name}"] = probabilities[:, class_index]
    prediction_frame.to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)
    np.save(OUTPUT_DIR / "test_probabilities.npy", probabilities.astype(np.float32))

    torch.save(
        {
            "state_dict": model.state_dict(),
            "class_names": CLASS_NAMES,
            "sample_rate": sample_rate,
            "signal_length": int(signals.shape[-1]),
            "seed": args.seed,
        },
        OUTPUT_DIR / "model.pt",
    )

    summary = {
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "test_accuracy": float((predicted_labels == labels[test_idx]).mean()),
        "train_examples": int(train_idx.shape[0]),
        "test_examples": int(test_idx.shape[0]),
    }
    with open(OUTPUT_DIR / "train_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


if __name__ == "__main__":
    main()
