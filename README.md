# Physics-Guided Explainable AI for Time-Series Signals

I evaluate whether explainable AI methods align with physical signal properties using novel metrics (consistency, frequency alignment, stability).

## Problem Definition

This project studies whether a neural network can both classify synthetic physical signals and produce explanations that agree with the known mechanics of the signal generator. The focus is not only predictive performance, but whether saliency concentrates on the physically meaningful portions of the waveform and whether attribution-weighted spectra preserve the expected frequency content.

## Mathematical Signal Model

Each signal has length \(T = 1000\) and sample rate \(f_s = 200\text{ Hz}\). The baseline process is

\[
x_{\text{base}}(t) = A_5 \sin(2\pi \cdot 5 t + \phi_5) + \sum_{k=1}^{K} A_k \sin(2\pi f_k t + \phi_k) + \epsilon(t),
\]

where \(f_k\) are nuisance mixture frequencies, \(K \in \{1,2\}\), and \(\epsilon(t) \sim \mathcal{N}(0, \sigma^2)\) with variable \(\sigma\).

Class-specific perturbations are:

\[
x_{\text{normal}}(t) = x_{\text{base}}(t),
\]

\[
x_{\text{fault\_light}}(t) = x_{\text{base}}(t) + A_{20}\sin(2\pi \cdot 20 t + \phi_{20}),
\]

\[
x_{\text{fault\_heavy}}(t) = x_{\text{base}}(t) + w(t; \tau_0, \tau_1) A_{50}\sin(2\pi \cdot 50 t + \phi_{50}),
\]

where \(w(t; \tau_0, \tau_1)\) is a Hann-windowed burst active only on the annotated anomaly interval.

## AI Method

The classifier is a reproducible PyTorch 1D CNN operating on two channels:

1. Raw normalized waveform.
2. Resized log-FFT magnitude.

The dominant FFT frequency is injected as a scalar auxiliary feature before the classifier head. Training uses a deterministic 80/20 split, cross-entropy loss, and Adam.

## XAI Method

Integrated Gradients from Captum is used to explain the predicted class with respect to the two-channel input. For physical evaluation, the raw-signal attribution is retained and then smoothed temporally with a moving-average kernel. Both raw and smoothed attribution maps are saved.

## Evaluation Metrics

1. **Physical Consistency Score**
   Fraction of positive attribution mass overlapping the true anomaly region:
   \[
   \mathrm{PCS} = \sum_{t \in \Omega_{\text{anomaly}}} \tilde{a}(t),
   \]
   where \(\tilde{a}(t)\) is normalized positive attribution.

2. **Frequency Alignment Score**
   Agreement between the dominant FFT frequency \(f_{\text{FFT}}\) and the dominant frequency of the attribution-weighted signal \(f_{\text{attr}}\):
   \[
   \mathrm{FAS} = \exp\left(-\frac{|f_{\text{FFT}} - f_{\text{attr}}|}{10}\right).
   \]

3. **Stability**
   Attribution robustness under injected noise, measured as one minus the total variation distance between normalized attribution maps.

4. **Temporal Coherence**
   Smoothness of attribution over time, computed from mean adjacent variation after normalization.

## Project Layout

```text
xai-physics/
├── data/
├── outputs/
├── src/
│   ├── analysis.py
│   ├── features.py
│   ├── generate.py
│   ├── metrics.py
│   ├── model.py
│   ├── train.py
│   └── xai.py
├── README.md
└── requirements.txt
```

## Reproducible Run

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m src.generate
.venv/bin/python -m src.train
.venv/bin/python -m src.xai
.venv/bin/python -m src.analysis
```

## Outputs

Running the pipeline writes:

- `data/signals.npy`, `data/labels.npy`, `data/metadata.csv`
- `outputs/model.pt`, `outputs/train_history.csv`, `outputs/test_predictions.csv`
- `outputs/attributions.npy`, `outputs/smoothed_attributions.npy`, `outputs/xai_metrics.csv`
- `outputs/confusion_matrix.csv`, `outputs/class_frequency_table.csv`, `outputs/insights.txt`
- `outputs/plots/*.png` for signal overlays, FFT alignment, and confusion matrices

## Results

The current run (`seed = 42`, `180` samples per class) produced:

- Held-out classification accuracy: `1.000`
- Diagnostic frequency alignment: `0.960`
- True-physics attribution score: `0.836`
- Attribution stability under noise: `0.973`
- Temporal coherence: `0.933`

Per-class attribution summary:

| class | true_freq | predicted_freq | attribution_score | consistency_score |
| --- | --- | --- | --- | --- |
| fault_heavy | 50.00 | 30.71 | 0.3198 | 0.0831 |
| fault_light | 20.00 | 20.00 | 1.0000 | 1.0000 |
| normal | 5.00 | 5.00 | 1.0000 | n/a |

Interpretation:

- The CNN classifies all three classes correctly.
- For `fault_light`, attribution-weighted spectra recover the injected `20 Hz` component exactly.
- For `fault_heavy`, the explanation does **not** localize strongly to the burst window (`PCS = 0.0831`) and the attribution-weighted spectrum drifts toward `30.71 Hz` rather than the true `50 Hz` burst.

The generated quantitative summary is stored in `outputs/analysis_summary.json`, the per-class table in `outputs/class_frequency_table.csv`, and the auto-written interpretation in `outputs/insights.txt`.

## Discussion

The central research question is whether the CNN learns physics-informed behavior or exploits superficial correlations. The answer is evaluated from two complementary perspectives:

1. **Temporal localization**: transient burst faults should attract attribution inside the annotated burst interval.
2. **Spectral fidelity**: global faults should preserve their dominant physical frequencies in the attribution-weighted spectrum.

When both hold simultaneously under perturbation, the explanation is more plausibly causal with respect to the underlying signal mechanism rather than merely correlated with the class label.

In this run, the answer is mixed:

- The model clearly learns the spectral physics for `fault_light`.
- The model is stable and its attributions are smooth.
- The model does **not** fully learn the transient burst physics for `fault_heavy`, despite perfect classification.

This is the core scientific finding of the project: high predictive accuracy does not guarantee physically faithful explanations. The heavy-fault case suggests that the CNN can separate classes while still relying on correlated spectral side-effects or contextual patterns rather than the true anomaly window itself. The proposed metrics make that gap measurable.
