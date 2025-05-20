---
layout: default
title: 2 ▸ Model Architecture
nav_order: 2
---

## 2. Model

### 2.1 Backbone
* **ResNet-50** pretrained on ImageNet.  
* First two residual stages are frozen; `layer3` & `layer4` are fine-tuned.
* A **Squeeze-and-Excitation block (SEBlock)** refines channel attention.

### 2.2 Multi-head design
Every category owns an individual *attribute head dictionary*:

self.attribute_heads['<cat-id>'][attr_name] = nn.Sequential(...)

If `is_complex_attribute()` returns *True* (empirically harder attributes), the sub-head is deeper (512->256->out); else a lighter 256-dim head is used.

### 2.3 Forward pass
1. Extract convolutional features.  
2. Pass through `SEBlock` and global-average-pool.  
3. For each sample in the batch select **only** heads that belong to its category.  
   This prevents impossible predictions (e.g. predicting *Sleeve Length* for *Sarees*).

Output:
```python
{
batch_idx0: {'attr_1': logits, 'attr_3': logits, ...},
batch_idx5: {...}, ...
}
```
