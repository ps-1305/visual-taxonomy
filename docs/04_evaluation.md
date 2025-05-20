---
layout: default
title: 4 ▸ Evaluation
nav_order: 4
---

## 4. Validation & Metrics

### 4.1 Hierarchical F1
For every `(Category, Attribute)` pair we compute both **micro** and **macro** F1, then harmonic-mean them:

\[
F_{harm} = \frac{2 \, F_{micro} \, F_{macro}}{F_{micro}+F_{macro}}
\]

Category score = mean over its attribute scores.
Final leaderboard score = mean over categories.

### 4.2 Early stopping
If validation score does not improve for *one* epoch → stop (small dataset).
The best checkpoint is stored as `best_model.pth` with full optimiser/scheduler state.
