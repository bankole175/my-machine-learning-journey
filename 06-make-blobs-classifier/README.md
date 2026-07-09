# 06 — Make Blobs Classifier

Four-class classification model identifying which cluster
a generated data point belongs to from 2 features.

**Val F1: 1.00 · Accuracy: 100% · Best Epoch: 1 · First Run ✅**

---

## Results

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Class 0 | 1.00 | 1.00 | 1.00 | 50 |
| Class 1 | 1.00 | 1.00 | 1.00 | 50 |
| Class 2 | 1.00 | 1.00 | 1.00 | 50 |
| Class 3 | 1.00 | 1.00 | 1.00 | 50 |
| **Weighted Avg** | **1.00** | **1.00** | **1.00** | **200** |

> Perfect classification on all four clusters on the first run.
> Val F1 reached 1.000 at epoch 1 and never dropped.

---

## Architecture

```
Input (2 features — x and y coordinates in 2D space)
    ↓
Linear(2 → 64) → BatchNorm → ReLU → Dropout(0.2)
    ↓
Linear(64 → 32) → BatchNorm → ReLU → Dropout(0.2)
    ↓
Linear(32 → 4)
    ↓
Output (Class 0 / Class 1 / Class 2 / Class 3)
```

| Hyperparameter | Value |
|---|---|
| Optimiser | AdamW |
| Learning rate | 0.001 |
| Weight decay | 0.01 |
| Batch size | 32 |
| Dropout | 0.2 |
| Patience | 20 |
| Scheduler | None |
| Early stopping epoch | 21 |

---

## Dataset

**Source:** sklearn make_blobs (generated)

```python
from sklearn.datasets import make_blobs

X, y = make_blobs(
    n_samples    = 1000,
    n_features   = 2,
    centers      = 4,
    random_state = 42
)
```

| Property | Value |
|---|---|
| Total samples | 1,000 |
| Features | 2 (x and y coordinates) |
| Classes | 4 (Cluster 0, 1, 2, 3) |
| Train split | 80% (800 samples) |
| Val split | 20% (200 samples) |
| Samples per class (val) | 50 each — perfectly balanced |

> make_blobs generates synthetic data with clean cluster structure.
> No noise, no overlap, no missing values.
> Perfect separation makes this the easiest classification task in the series.

---

## Training Output

```
Epoch   1 | Train Loss: 1.0360 | Train F1: 0.652 | Val F1: 1.000 | improved
Epoch   5 | Train Loss: 0.1518 | Train F1: 0.999 | Val F1: 1.000 | no imp 4/20
Epoch  10 | Train Loss: 0.0666 | Train F1: 0.985 | Val F1: 1.000 | no imp 9/20
Epoch  21 | Early stopping
```

Val F1 hit 1.000 on the very first epoch and never dropped.
The cluster boundaries are so clean that the model found them immediately.

---

## Why This Dataset Is Different

| Dataset | Type | Noise | First Run F1 | Runs Needed |
|---|---|---|---|---|
| Breast Cancer | Real medical | Some | 0.98 | 1 |
| Wine | Real chemical | Some | 1.00 | 1 |
| Digits | Real images | Moderate | 0.44 | 3 |
| Iris | Real botanical | Moderate | 0.90 | 3 |
| Diabetes | Real medical | High | -3.99 | 3 |
| Make Blobs | Generated | None | 1.00 | 1 |

Generated data with no noise is always easiest to classify.
Real-world data always has noise — which is why it is harder
and why the previous projects needed multiple debugging runs.

---

## Key Learnings

- **Generated vs real data** — clean generated data is much easier than real data
- **Hyperparameter decisions were correct first time** — applied lessons from all previous projects
- **Val F1 > Train F1 early on** — dropout cripples training mode but full model is strong
- **Early stopping working correctly** — fired at epoch 21 and stopped immediately
- **No scheduler needed** — clean data converges fast without one
- **Scale y? No** — classification labels are never scaled

---

## How to Run

```bash
pip install torch scikit-learn numpy
python make_blobs_classifier.py
```