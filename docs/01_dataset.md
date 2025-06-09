---
layout: default
title: 1 ▸ Dataset
nav_order: 1
---

## 1. Dataset & Label Engineering

### 1.1 CSV Schema

| column | description                |
| ----------- | -------------------------- |
| id | unique product identifier  |
| Category | top-level garment class    |
| attr\_1 … 10 | category-specific features |

### 1.2 Category → attribute mapping
Different categories own different attribute vocabularies.  
During `FashionDataset.setup_mappings()` we:

* build a `category_map` (`{'Sarees': 0, 'Women Tops & Tunics': 1, …}`),
* create `attr_maps[category][attr_i] = {value → index}` **only for values that appear**,
* store frequency in `attr_distributions` for class-balanced loss weights.

### 1.3 Dynamic padding
If a row is shorter than the maximum attribute length of its category we keep `NaN`; the training loop ignores labels `-1`, while the **post-processing script** fills `dummy_value` to match the competition submission format.

### 1.4 Data augmentation
`augment=True` enables a `transforms.Compose` with  
`RandomHorizontalFlip → RandomRotation → ColorJitter → RandomAffine`  
while the *validation* pipeline is plain `Resize → ToTensor → Normalize`.
CS
