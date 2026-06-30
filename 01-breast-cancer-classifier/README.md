# 01 — Breast Cancer Classifier

Binary classification model predicting whether a breast tumour is
**malignant** or **benign** from 30 clinical measurements.

**Val F1: 0.98 · Accuracy: 98% · Best Epoch: 4**

---

## Results

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Malignant | 0.98 | 0.98 | 0.98 |
| Benign | 0.99 | 0.99 | 0.99 |
| **Weighted Avg** | **0.98** | **0.98** | **0.98** |

---

## Architecture

```
Input (30 features)
    ↓
Linear(30 → 64) → BatchNorm → ReLU → Dropout(0.3)
    ↓
Linear(64 → 32) → BatchNorm → ReLU → Dropout(0.3)
    ↓
Linear(32 → 2)
    ↓
Output (Malignant / Benign)
```

| Hyperparameter | Value |
|---|---|
| Optimiser | AdamW |
| Learning rate | 0.001 |
| Weight decay | 0.01 |
| Batch size | 32 |
| Dropout | 0.3 |
| Patience | 15 |
| Best epoch | 4 |

---

## Dataset

**Source:** sklearn breast cancer dataset (built-in)

| Property | Value |
|---|---|
| Total samples | 569 |
| Features | 30 |
| Classes | 2 (Malignant: 212, Benign: 357) |
| Train split | 80% (455 samples) |
| Val split | 20% (114 samples) |

---

## Key Learnings

- Built the complete PyTorch training loop from scratch for the first time
- Learned the correct order: zero_grad → forward → loss → backward → step
- Understood why features must be scaled before training
- Implemented early stopping and model checkpointing
- Model converged quickly — best weights found at epoch 4

---

## How to Run

```bash
pip install torch scikit-learn numpy
python breast_cancer.py
```