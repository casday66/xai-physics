from __future__ import annotations

import os
import random

import numpy as np
import torch
from torch import nn


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True)
    except RuntimeError:
        pass


class PhysicsGuidedCNN(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(2, 16, kernel_size=9, padding=4),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(65, 32),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(32, num_classes),
        )

    def forward(self, series: torch.Tensor, dominant_frequency: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(series).squeeze(-1)
        if dominant_frequency.ndim == 1:
            dominant_frequency = dominant_frequency.unsqueeze(1)
        features = torch.cat([encoded, dominant_frequency], dim=1)
        return self.classifier(features)


@torch.no_grad()
def predict_probabilities(
    model: nn.Module,
    series: torch.Tensor,
    dominant_frequency: torch.Tensor,
    batch_size: int = 128,
    device: str = "cpu",
) -> np.ndarray:
    model.eval()
    probabilities = []
    for start in range(0, series.shape[0], batch_size):
        stop = start + batch_size
        logits = model(series[start:stop].to(device), dominant_frequency[start:stop].to(device))
        probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(probabilities, axis=0)
