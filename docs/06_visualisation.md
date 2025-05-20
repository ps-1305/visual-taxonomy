---
layout: default
title: 6 ▸ Result Visualisation
nav_order: 6
---

## 6. Visualising Performance

During post-training analysis we reuse the saved model *on training data* to spot over-/under-fitting:

```
final_score, category_scores, preds, gts = evaluate(model, train_loader)
```

### 6.1 Category-level bar chart
![Category F1](assets/img/category_f1.png)

### 6.2 Attribute-level charts
Loop through `category_scores` and draw per-attribute macro-F1 bars (see notebook snippet in repo).
These plots reveal that *attr\_4* in *Sarees* and *attr\_5* in *Women Tshirts* are still bottlenecks.
