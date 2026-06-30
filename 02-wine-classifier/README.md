# 02 — Wine Quality Classifier

Three-class classification model identifying wine type
(**class 0, 1, or 2**) from 13 chemical measurements.

**Val F1: 1.00 · Accuracy: 100% · Best Epoch: 35**

---

## Results

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Class 0 | 1.00 | 1.00 | 1.00 |
| Class 1 | 1.00 | 1.00 | 1.00 |
| Class 2 | 1.00 | 1.00 | 1.00 |
| **Weighted Avg** | **1.00** | **1.00** | **1.00** |

---

## Architecture

```
Input (13 features)
    ↓
Linear(13 → 64) → BatchNorm → ReLU → Dropout(0.3)
    ↓
Linear(64 → 32) → BatchNorm → ReLU → Dropout(0.3)
    ↓
Linear(32 → 3)
    ↓
Output (Class 0 / Class 1 / Class 2)
```

| Hyperparameter | Value |
|---|---|
| Optimiser | AdamW |
| Learning rate | 0.001 |
| Weight decay | 0.01 |
| Batch size | 32 |
| Dropout | 0.3 |
| Patience | 15 |
| Best epoch | 35 |

---

## Dataset

**Source:** sklearn wine dataset (built-in)

| Property | Value |
|---|---|
| Total samples | 178 |
| Features | 13 |
| Classes | 3 |
| Train split | 80% (142 samples) |
| Val split | 20% (36 samples) |

---

## Key Learnings

- Confirmed the training loop pattern works across different class counts
- Noticed validation F1 starts higher than train F1 in early epochs
  — because dropout is ON during training and OFF during validation
- Early stopping correctly saved the best checkpoint at epoch 35

---

## How to Run

```bash
pip install torch scikit-learn numpy
python wine_classifier.py
```