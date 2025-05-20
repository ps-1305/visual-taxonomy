---
layout: default
title: 3 ▸ Training
nav_order: 3
---

## 3. Training Strategy

| component | choice |
| -------------- | ------ |
| Loss | **Weighted Focal Loss** (γ = 2) |
| Optimiser | AdamW (LR = 1e-3, wd = 0.01) |
| Scheduler | CosineAnnealingLR (T<sub>max</sub>=5) |
| Batch size | 32 |
| Epochs | 5 (+ early stop patience = 1) |
| Gradient clip | 1.0 |

### 3.1 Why Focal Loss?
Dataset has extreme label imbalance (rare colours, patterns).
Formula:

\[
\text{FL} = \frac{1}{N}\sum (1-p_t)^{\gamma}\log(p_t)\,,
\]
`p_t` = softmax probability of ground-truth class.

Class weights (`alpha`) come from `attr_distributions` (√ inverse frequency, then normalised).

### 3.2 Per-sample, per-attribute weighting
Inside the batch loop we multiply each attribute loss by `attr_weights[attr][idx]` → a light form of **cost-sensitive learning**.

### 3.3 Regularisation
* Heavy data augmentation.
* Dropout inside heads (0.3–0.4).
* GroupNorm (8 groups) instead of BatchNorm – robust to small batch sizes.
