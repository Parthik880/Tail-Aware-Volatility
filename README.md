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

\[
\tilde{X}_{t,i} = X_{t,i} \cdot \psi_i + b
\]

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

### 5. Tail-Aware Training Objective

The training loss combines:
- MSE on log-volatility
- asymmetric tail penalty (underestimation-aware)
- optional relative error in real-volatility space

This biases the model toward **safer tail behavior**, which is often more important than mean accuracy in risk-sensitive settings.

---

## Repository Structure

