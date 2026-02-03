# Tail aware Volatility 
# Volatility Tail Model

A research-oriented deep learning framework for **volatility forecasting with tail-risk sensitivity**, combining engineered financial features, fixed linear feature modulation (ψ), and sequence models (CNN/RNN/Transformer-style blocks).

This repository is designed for **exploration, robustness, and extensibility**, not for claiming benchmark performance.

---

## Overview

Financial volatility exhibits:
- heavy tails
- clustering
- sudden spikes
- asymmetric underestimation risk

This project addresses these properties by:
- rich feature engineering from OHLCV data
- a **fixed, feature-wise linear modulation (ψ)** learned on the training split
- sequence models that emphasize **tail behavior**
- asymmetric loss terms that penalize volatility underestimation

---

## Key Ideas

### 1. Feature Engineering
Deterministic features computed from raw OHLCV data:
- returns and lagged volatility
- price structure (range, wicks, gaps)
- momentum and trend indicators
- mean-reversion signals
- volume and liquidity measures
- market stress indicators

All feature engineering is **stateless and deterministic**.

---

### 2. Fixed ψ Feature Modulation

A linear model is fitted **only on the training split** to obtain a coefficient vector ψ.

ψ is used as a **feature-wise diagonal modulation**:

X_tilde[t, i] = X[t, i] * psi[i] + b


The modulated features are concatenated with the original features to preserve memory and structure.

Important:
- ψ is **not refit** on validation or test data
- ψ is treated as a **fixed preprocessing transform**
- this is **not** used as a forecast, but as a static linear memory/gating mechanism

---

### 3. Time-Series Windowing

Sequence construction is handled in the dataset layer:
- sliding windows over time
- configurable sequence length
- proper target alignment (`t → t+1` volatility)

This keeps preprocessing reusable and model-agnostic.

---

### 4. Models

The framework supports multiple architectures:
- CNN-based feature extractors
- RNN (GRU) for temporal dynamics
- attention / transformer-style blocks
- DenseNet-1D / ResNet-1D variants

Models operate on `(batch, time, features)` sequences.

---

### 5. Tail Penalty Coefficient (k)

The parameter k controls how strongly the model penalizes underprediction during high-volatility (tail) periods.

Standard MSE loss tends to smooth predictions and underestimate rare volatility spikes. To address this, an additional asymmetric penalty is applied whenever the predicted log-volatility is lower than the true log-volatility.

How it works

During training:

The model predicts log-volatility

Mean Squared Error (MSE) is used as the base loss

An extra penalty is added only when volatility is underpredicted

A relative error term stabilizes scaling across regimes

The final loss is a weighted sum of:

MSE loss on log-volatility

Tail penalty (activated only when prediction < target)

Relative error in real volatility space

The coefficient k determines how much weight is given to the tail penalty.

Effect of different k values

Small k (0 – 0.5)
Smooth predictions, strong regime tracking, but tail spikes are often underestimated.

Moderate k (1 – 3) (recommended)
Better responsiveness to volatility spikes while maintaining stable training.

Large k (> 5)
Aggressive tail sensitivity, which may lead to overprediction and unstable training dynamics.

Practical considerations

Volatility tails are asset-specific; a single global k may behave differently across stocks.

Increasing k improves crisis-period accuracy but can degrade performance during low-volatility regimes.

In this repository, k is treated as a research hyperparameter, not a fixed constant.

Recommended setting

For most equity datasets:
k in the range [1,3]


## Repository Structure

tail-aware-volatility-model/
│
├── train.py              # training loop (mostly your current code)
├── model.py              # model class only
├── dataset.py            # Dataset + DataLoader             
├── requirements.txt
└── README.md
