# 03 — Handwritten Digit Recogniser

Ten-class classification model recognising handwritten digits
(**0 through 9**) from 8×8 pixel images flattened to 64 features.

**Val F1: 0.99 · Accuracy: 99% · Best Epoch: 20**

---

## Results

| Class | Precision | Recall | F1 |
|---|---|---|---|
| 0 | 1.00 | 1.00 | 1.00 |
| 1 | 0.95 | 0.97 | 0.96 |
| 2 | 0.97 | 1.00 | 0.99 |
| 3 | 0.97 | 1.00 | 0.99 |
| 4 | 1.00 | 1.00 | 1.00 |
| 5 | 1.00 | 1.00 | 1.00 |
| 6 | 1.00 | 1.00 | 1.00 |
| 7 | 1.00 | 1.00 | 1.00 |
| 8 | 0.97 | 0.91 | 0.94 |
| 9 | 1.00 | 0.97 | 0.99 |
| **Weighted Avg** | **0.99** | **0.99** | **0.99** |

---

## Architecture

```
Input (64 features — 8×8 pixel grid)
    ↓
Linear(64 → 128) → BatchNorm → ReLU → Dropout(0.1)
    ↓
Linear(128 → 64) → BatchNorm → ReLU → Dropout(0.1)
    ↓
Linear(64 → 10)
    ↓
Output (Digits 0–9)
```

| Hyperparameter | Value |
|---|---|
| Optimiser | AdamW |
| Learning rate | 0.001 |
| Weight decay | 0.01 |
| Batch size | 32 |
| Dropout | 0.1 |
| Patience | 30 |
| Scheduler | None |
| Best epoch | 20 |

---

## Dataset

**Source:** sklearn digits dataset (built-in)

| Property | Value |
|---|---|
| Total samples | 1,797 |
| Features | 64 (8×8 pixel images) |
| Classes | 10 (digits 0–9) |
| Train split | 80% (1,437 samples) |
| Val split | 20% (360 samples) |

---

## Training Journey

This project required three runs to converge — each one teaching
a different lesson about hyperparameter tuning and model diagnosis.

| Run | Val F1 | Problem | Diagnosis | Fix |
|---|---|---|---|---|
| 1 | 0.44 | Not learning | Underfitting | Bigger model, more patience, slower scheduler |
| 2 | 0.85 | Plateau | Dropout too strong | Dropout 0.3 → 0.1, removed scheduler |
| 3 | 0.99 | Converged ✅ | — | — |

### Run 1 — Underfitting (Val F1: 0.44)

```
Epoch  1 | Train F1: 0.189 | Val F1: 0.406
Epoch 22 | Early stopping
```

Both train and val F1 were low — classic underfitting.
The model was too small, patience too low, and the scheduler
was decaying the learning rate before the model had learned anything.

**Fixes:** hidden_size 64 → 128, patience 15 → 30, step_size 10 → 20

### Run 2 — Plateau (Val F1: 0.85)

```
Epoch  1 | Train F1: 0.470 | Val F1: 0.827
Epoch 44 | Train F1: 0.770 | Val F1: 0.852
```

Val F1 much higher than Train F1 — dropout too aggressive.
Dropout(0.3) was randomly zeroing too many neurons during training,
preventing the model from fitting the training data properly.
The scheduler was still decaying lr too early.

**Fixes:** Dropout 0.3 → 0.1, removed scheduler entirely

### Run 3 — Converged (Val F1: 0.99)

```
Epoch  1 | Train F1: 0.658 | Val F1: 0.907
Epoch 10 | Train F1: 0.986 | Val F1: 0.986
Epoch 20 | Train F1: 0.993 | Val F1: 0.992
```

Train and val close together. Both high. Smooth convergence.
Best checkpoint saved at epoch 20.

---

## Key Learnings

- **Underfitting vs Overfitting** — experienced both in real training runs
- **Reading the train/val gap** — large gap means something is wrong
- **Dropout calibration** — 0.3 is too aggressive for larger datasets
- **Scheduler timing** — premature lr decay prevents convergence
- **Debugging workflow** — read output → diagnose → one fix at a time

---

## Debugging Cheatsheet (From This Project)

```
Both train and val F1 low    → Underfitting → bigger model, more patience
Val F1 >> Train F1           → Dropout too high → reduce dropout
Model plateauing early       → LR decaying too fast → remove or slow scheduler
Both high and close together → Well fitted ✅
```

---

## How to Run

```bash
pip install torch scikit-learn numpy
python digits_classifier.py
```