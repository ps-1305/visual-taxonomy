---
layout: default
title: 5 ▸ Inference
nav_order: 5
---

## 5. Batch Inference

model = load_trained_model('best_model.pth', dataset, device)
predictions_df = predict_test_data(model,
                                   test_csv='test.csv',
                                   test_img_dir='test_images',
                                   batch_size=32)


Highlights:

* **Category-conditioned heads** again ensure we don’t predict invalid attributes.
* Probabilities via `softmax`; highest prob = prediction, stored with its confidence.
* The helper `process_csv()` appends `len` column and fills trailing `dummy_value` to comply with competition submission specs.
