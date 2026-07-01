# 04 — Iris Flower Classifier

Three-class classification model identifying iris flower species
(**Setosa, Versicolour, or Virginica**) from 4 petal and sepal measurements.

**Val F1: 0.97 · Accuracy: 97% · Best Epoch: 35**

---

## Results

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Setosa | 1.00 | 1.00 | 1.00 |
| Versicolour | 1.00 | 0.90 | 0.95 |
| Virginica | 0.91 | 1.00 | 0.95 |
| **Weighted Avg** | **0.97** | **0.97** | **0.97** |

> Setosa is perfectly separable from the other two species.
> Versicolour and Virginica overlap slightly — a known property of this dataset.

---

## Architecture

```
Input (4 features: sepal length, sepal width, petal length, petal width)
    ↓
Linear(4 → 32) → BatchNorm → ReLU → Dropout(0.1)
    ↓
Linear(32 → 16) → BatchNorm → ReLU → Dropout(0.1)
    ↓
Linear(16 → 3)
    ↓
Output (Setosa / Versicolour / Virginica)
```

| Hyperparameter | Value |
|---|---|
| Optimiser | AdamW |
| Learning rate | 0.001 |
| Weight decay | 0.01 |
| Batch size | 32 |
| Dropout | 0.1 |
| Patience | 20 |
| Scheduler | None |
| Best epoch | 35 |

---

## Dataset

**Source:** sklearn iris dataset (built-in)

| Property | Value |
|---|---|
| Total samples | 150 |
| Features | 4 |
| Classes | 3 (Setosa: 50, Versicolour: 50, Virginica: 50) |
| Train split | 80% (120 samples) |
| Val split | 20% (30 samples) |

---

## Training Journey

This was the smallest dataset trained on so far — 120 training samples.
Required three runs to converge correctly.

| Run | Val F1 | Problem | Diagnosis | Fix |
|---|---|---|---|---|
| 1 | 0.90 | Early stopping broken, overfitting | break inside batch loop, not epoch loop | Moved break to epoch level |
| 2 | 0.73 | Not learning enough | Model too small, patience too low | hidden_size 16→32, patience 10→20 |
| 3 | 0.97 | Converged ✅ | — | — |

### Run 1 — Early Stopping Bug (Val F1: 0.90)

```
Epoch 15 | no imp 15/15
Early stopping at epoch 16
Early stopping at epoch 17  ← still running — bug
Early stopping at epoch 18  ← still running — bug
```

The `break` statement was inside the batch loop instead of the epoch loop.
It was breaking out of each batch iteration but not stopping training.
The model kept running for 100 epochs despite triggering early stopping at epoch 16.

**Fix:** Moved early stopping check and `break` to the epoch level — after the validation loop.

### Run 2 — Underfitting (Val F1: 0.73)

```
Epoch  1 | Train F1: 0.289 | Val F1: 0.139
Epoch 23 | Early stopping — still improving
Final Val F1: 0.73
```

Only 291 parameters — too few for a 3-class problem even with 120 samples.
Patience of 10 fired at epoch 23 while the model was still learning.

**Fix:** hidden_size 16 → 32 (835 parameters), patience 10 → 20

### Run 3 — Converged (Val F1: 0.97)

```
Epoch  5  | Val F1: 0.707
Epoch 30  | Val F1: 0.898
Epoch 35  | Val F1: 0.967  ← best checkpoint saved
Epoch 54  | Early stopping
```

Smooth convergence. The jump from 0.898 to 0.967 between epochs 30–35
is the model finding the decision boundary between Versicolour and Virginica.

---

## Key Learnings

- **Early stopping placement** — `break` must be at the epoch level, not inside the batch loop
- **Tiny datasets need smaller models** — 120 samples cannot support a large hidden layer
- **Patience calibration** — patience=10 was too low, model was still learning when it stopped
- **Class difficulty is real** — Setosa is perfectly separable (F1: 1.00), Versicolour and Virginica overlap (F1: 0.95)
- **No scheduler needed** — tiny dataset trains fast, scheduler adds unnecessary complexity

---

## Debugging Pattern (From This Project)

```
Symptom                                   → Diagnosis
──────────────────────────────────────────────────────────
"Early stopping" prints but keeps running → break in wrong loop
Both losses high, stopped too soon        → model too small or patience too low
Val F1 jumps sharply at one epoch         → model found a key decision boundary
```

---

## How to Run

```bash
pip install torch scikit-learn numpy
python iris_classifier.py
```