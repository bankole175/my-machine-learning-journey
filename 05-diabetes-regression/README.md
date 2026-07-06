# 05 — Diabetes Regression Predictor

Regression model predicting disease progression in diabetes patients
one year after baseline — from 10 clinical measurements.

**Final R²: 0.529 · RMSE: 49.9 · Best Epoch: 40**

> R² of 0.529 means the model explains 52.9% of the variance
> in disease progression. A linear regression baseline scores ~0.48.
> This neural network outperforms the classical baseline.

---

## Results

| Metric | Value |
|---|---|
| R² Score | **0.529** |
| RMSE | **49.9** |
| Linear Regression Baseline R² | ~0.480 |
| Beats Baseline | ✅ Yes |

### Sample Predictions

| Actual | Predicted | Error |
|---|---|---|
| 219.0 | 131.8 | 87.2 |
| 70.0 | 168.5 | 98.5 |
| 202.0 | 178.9 | 23.1 |
| 230.0 | 296.6 | 66.6 |
| 111.0 | 107.8 | 3.2 |
| 84.0 | 123.0 | 39.0 |
| 242.0 | 285.9 | 43.9 |
| 272.0 | 201.5 | 70.5 |
| 94.0 | 81.9 | 12.1 |
| 96.0 | 77.5 | 18.5 |

> The model captures the general trend well (errors of 3–25 on easier cases)
> but struggles with outlier patients (errors of 87–99).
> This reflects genuine noise in disease progression data.

---

## Architecture

```
Input (10 clinical measurements)
    ↓
Linear(10 → 64) → BatchNorm → ReLU → Dropout(0.1)
    ↓
Linear(64 → 32) → BatchNorm → ReLU → Dropout(0.1)
    ↓
Linear(32 → 1)
    ↓
Output (disease progression score — one continuous number)
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
| Best epoch | 40 |

---

## Dataset

**Source:** sklearn diabetes dataset (built-in)

| Property | Value |
|---|---|
| Total samples | 442 |
| Features | 10 (age, sex, BMI, blood pressure, 6 serum measurements) |
| Target | Disease progression score (continuous, range 25–346) |
| Train split | 80% (353 samples) |
| Val split | 20% (89 samples) |

---

## Key Difference From Classification Projects

This is the first **regression** project in the series.
Several things change compared to classification:

| | Classification | Regression |
|---|---|---|
| Loss function | CrossEntropyLoss | MSELoss |
| Output size | num_classes | 1 |
| Label dtype | torch.long | torch.float32 |
| Label scaling | Not needed | Required ✅ |
| Predictions | argmax(dim=1) | squeeze() |
| Metric | F1 score | R² and RMSE |
| Final report | classification_report | actual vs predicted table |

---

## Training Journey

Required three runs to diagnose and fix the label scaling issue.

| Run | Val R² | Problem | Diagnosis | Fix |
|---|---|---|---|---|
| 1 | -3.996 | Model outputting near-zero values | Labels not scaled | Added StandardScaler for y |
| 2 | 0.481 | Predictions showing scaled values | Forgot inverse_transform in display | Applied scaler_y.inverse_transform |
| 3 | 0.529 | Converged ✅ | — | — |

### Run 1 — Label Scaling Missing (Val R²: -3.996)

```
Epoch  1 | Val R²: -3.996
Epoch 10 | Val R²: -3.869
Sample predictions:
  Actual: 219.0    Predicted: 0.6    Error: 218.4
```

The model was outputting values between 0 and 13 while actual values
were between 70 and 346. The gap was too large for the model to bridge.

Root cause: features were scaled (mean=0, std=1) but labels were not.
The model learned in normalised feature space but had to output
unnormalised label values — impossible with small initial weights.

**Fix:** Added StandardScaler for y labels.
```python
scaler_y = StandardScaler()
y_train  = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
y_val    = scaler_y.transform(y_val.reshape(-1, 1)).flatten()
```

### Run 2 — Display Bug (Val R²: 0.481)

```
Sample predictions:
  Actual: 0.8    Predicted: -0.0    Error: 0.9
```

R² was correct (0.481) but the sample table was showing scaled values.
The inverse transform was applied but the display loop used the wrong variables.

**Fix:** Changed `final_labels[:10]` to `final_labels_real[:10]` in the display loop.

### Run 3 — Converged (Val R²: 0.529)

```
Epoch  5  | Val R²: 0.504
Epoch 35  | Val R²: 0.534
Epoch 40  | Val R²: 0.536  ← best checkpoint
Epoch 58  | Early stopping

Final R²:   0.529
Final RMSE: 49.9
```

---

## Key Learnings

- **Regression vs Classification** — fundamentally different output and loss
- **Scale your labels** — the most important regression-specific lesson
- **Inverse transform for display** — model works in scaled space, humans need real values
- **R² interpretation** — 0.529 means 52.9% of variance explained
- **squeeze() for output** — removes the extra dimension from single-output regression
- **Edge case handling** — squeeze() on batch_size=1 removes all dims, need unsqueeze(0)
- **Natural ceiling** — disease progression has genuine noise, R² above 0.55 is very hard

---

## Regression Debugging Cheatsheet

```
Symptom                          Diagnosis                Fix
─────────────────────────────────────────────────────────────────
R² deeply negative (-3 to -10)  Labels not scaled        Add StandardScaler for y
Predictions near zero            Same as above            Same fix
Sample table shows 0.8, -1.1    Forgot inverse_transform Use final_labels_real
Loss stuck at 28,000+            Labels not scaled        Add StandardScaler for y
R² plateaus around 0.5           Natural data ceiling     Stop training, move on
```

---

## How to Run

```bash
pip install torch scikit-learn numpy
python diabetes_regression.py
```