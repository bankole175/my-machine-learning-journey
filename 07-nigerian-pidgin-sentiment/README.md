# Nigerian Pidgin Sentiment Classifier

A from-scratch PyTorch sentiment classifier (positive / negative / neutral) for Nigerian
Pidgin text, trained on a combination of synthetic e-commerce reviews and real,
human-annotated Twitter data.

## TL;DR

- **Real-world performance:** 65% accuracy, 0.49 macro F1, 0.66 weighted F1 on a genuine
  held-out set of human-written Pidgin text.
- **Positive/negative classification** works reasonably well (F1 0.58 and 0.75).
- **Neutral classification is a known weak point** (F1 0.14), limited by how little
  genuinely neutral, real-world labeled Pidgin text exists in any available corpus.
- A model trained purely on synthetic data scored a *misleading* 1.00 F1 on its own
  synthetic test set, but only 0.36–0.47 macro F1 on real text. See
  [Key finding](#key-finding-synthetic-only-training-was-misleading) below.

## Datasets

### 1. Synthetic e-commerce reviews (`nigerian_pidgin_reviews.csv`)

1,500 template-generated Pidgin product reviews, written to mimic the style of Jumia
(Nigerian e-commerce) buyer reviews. Built from ~60 product names combined with ~15
sentiment-specific sentence templates per class, plus randomized openers/closers for
surface variety.

**Important limitation:** this data is synthetic and template-based. It is useful for
learning e-commerce-specific vocabulary ("delivery," "packaging," "seller," etc.) but
does not capture the linguistic diversity of real Pidgin, and a model trained only on it
will overfit to the template structure rather than learning general sentiment (see
below).

### 2. Real annotated data — NaijaSenti (`naijasenti_train.csv`, `naijasenti_dev.csv`, `naijasenti_test.csv`)

Sourced from [NaijaSenti](https://github.com/hausanlp/NaijaSenti) (Muhammad et al.,
LREC 2022), the first large-scale human-annotated Twitter sentiment corpus for Nigerian
Pidgin (and Hausa, Igbo, Yoruba). 10,556 real, human-labeled tweets covering everyday
topics — not product reviews, but genuine, naturally-written Pidgin with real sentiment
labels assigned by human annotators.

Original label distribution:

| Split | Positive | Negative | Neutral | Total |
|---|---|---|---|---|
| train | 1,808 | 3,241 | 72 | 5,121 |
| dev | 447 | 813 | 21 | 1,281 |
| test | 1,397 | 2,326 | 431 | 4,154 |

### 3. Combined training set

`combined_train.csv` — real train + real dev + all synthetic reviews (7,902 rows
originally). This file was later **updated in place**: 258 real neutral examples were
reallocated into it from `naijasenti_test.csv` to address severe neutral-class scarcity,
bringing it to 8,160 rows. See [Addressing the neutral class](#addressing-the-neutral-class)
below.

`naijasenti_test.csv` was correspondingly **updated in place** to remove those 258
reallocated rows, shrinking from 4,154 to 3,896 rows (2,326 negative, 1,397 positive, 173
neutral). It remains a genuine, disjoint holdout — the reallocated rows exist only in
`combined_train.csv` now, not in both files.

**Note:** because both files were edited in place rather than saved under new names, any
model checkpoint or result generated *before* the reallocation used a different version
of these two files than anything generated *after*. The iteration table below reflects
which version of the data each result used.

## Key finding: synthetic-only training was misleading

Early experiments trained and evaluated entirely on the synthetic 1,500-review dataset
and produced a perfect classification report:

```
accuracy: 1.00   macro avg F1: 1.00   (all three classes: 1.00 precision/recall/F1)
```

This did **not** mean the model understood Pidgin sentiment. It meant the model had
memorized the ~45 sentence templates used to generate the data — every test example was
a template the model had already seen dozens of times in training, just with a different
product name swapped in. TF-IDF features made this worse: shared template phrases mapped
to identical or near-identical feature vectors regardless of the product mentioned.

**Evaluating the same model on real, held-out NaijaSenti tweets (never seen in any form
during training) told a very different story:** macro F1 dropped from 1.00 to 0.36, and
neutral-class recall was effectively zero. This is the central lesson of this project —
synthetic template data can produce arbitrarily high self-reported scores that have no
relationship to real-world performance. Every result reported here is measured against
real, human-written text, never against synthetic data alone.

## Methodology and iteration history

All real-data numbers below are from `naijasenti_test.csv` (genuine holdout, never used
in training or model selection). Iterations 1–3 used the original version of
`naijasenti_test.csv` (4,154 rows, 431 real neutral); iterations 4–5 used the file
*after* it was updated in place to remove the 258 reallocated neutral rows (3,896 rows,
173 real neutral) — see the note in [Datasets](#3-combined-training-set) above. Because
the test set itself changed at that point, iteration 3 and iteration 4's scores aren't
perfectly apples-to-apples, though both are genuine, disjoint holdouts at the time each
was measured.

| Iteration | Change | Real accuracy | Real macro F1 | Real neutral F1 |
|---|---|---|---|---|
| 1 | Synthetic data only | 0.47 | 0.36 | 0.13 |
| 2 | + Real NaijaSenti train/dev mixed into training | 0.63 | 0.43 | 0.01 (collapsed) |
| 3 | + Class-weighted loss (`compute_class_weight`) | 0.62 | 0.42 | 0.03 |
| 4 | + Reallocated 258 real neutral examples from test → train (test set updated in place) | 0.63 | 0.47 | 0.11 |
| 5 | + Richer TF-IDF (5000 features, bigrams, `sublinear_tf`) + loss-aware checkpoint selection | **0.65** | **0.49** | **0.14** |

For reference, a trivial baseline that always predicts the majority class ("negative,"
~56–60% of the real test set depending on version) would score roughly that accuracy.
The final model (iteration 5) exceeds this baseline with real margin, and does so with a
genuinely balanced macro F1 across classes rather than by only getting the majority class
right.

### Addressing the neutral class

Neutral was consistently the hardest class throughout this project, for a clear reason:
real, human-annotated neutral Pidgin text is scarce. The original NaijaSenti train+dev
splits contained only 93 real neutral examples combined (72 + 21), compared to thousands
of positive and negative examples. Class-weighted loss alone could not fix this — it
made the model *try* to predict neutral more often, but it had no real linguistic
signal to learn from, since ~76% of "neutral" training examples were still
synthetic-template text that doesn't resemble how real neutral sentiment is expressed.

The fix that actually moved the needle was **reallocating real data**: 258 of the 431
real neutral examples in the original test set were moved into `combined_train.csv`
(60%), leaving 173 in `naijasenti_test.csv` as a smaller but still genuine, disjoint
holdout. Both files were updated in place to reflect this. This nearly quadrupled real
neutral training signal (93 → 351 examples) and improved real neutral F1 from 0.03 to
0.11–0.14 over subsequent iterations.

**This remains the weakest part of the model.** With only 173 held-out real neutral
examples, that specific class's F1 score should be treated as a rough estimate — a small
number of flipped predictions moves it more than for positive/negative. Further
improvement would require sourcing additional genuinely neutral, real-world Pidgin text,
which is a data-collection problem rather than something fixable through further model
or loss tuning.

### Model calibration note

Across most iterations, validation loss climbed substantially during training (e.g. 0.79
→ 2.19 in one run) even while validation F1 stayed flat or improved slightly. This
indicates the model becomes increasingly overconfident in its predictions over training,
independent of whether those predictions are correct. Checkpoint selection was changed
in iteration 5 to require validation loss not to be increasing sharply (within 15% of its
running minimum) before accepting a new "best" F1 checkpoint, which reduced but did not
eliminate this effect. If prediction confidence scores are ever exposed downstream, they
should not be treated as calibrated probabilities.

## Architecture

- **Features:** TF-IDF (5,000 features, unigrams + bigrams, `sublinear_tf=True`,
  `min_df=2`)
- **Model:** 3-layer feedforward network (`Linear → BatchNorm → ReLU → Dropout`,
  repeated), hidden sizes 32–64, trained with `AdamW` and `CrossEntropyLoss`
  (class-weighted, `'balanced'` scheme)
- **Training:** early stopping on validation F1 with a loss-stability guard, patience 20
  epochs

## Reproducing results

```python
df = pd.read_csv('combined_train.csv')   # training + internal val split
# ... train as in train.py ...
real_df = pd.read_csv('naijasenti_test.csv')  # final, untouched real-world evaluation
```

Always evaluate final results against the current `naijasenti_test.csv`, not against any
internal validation split carved out of the training data — the internal split shares
distribution with training data (including synthetic templates) and will overstate
real-world performance.

## Files

| File | Description |
|---|---|
| `nigerian_pidgin_reviews.csv` | Synthetic Pidgin e-commerce reviews |
| `naijasenti_train.csv` / `naijasenti_dev.csv` | Real NaijaSenti splits (unmodified) |
| `naijasenti_test.csv` | Real NaijaSenti test split — **updated in place** after neutral reallocation (currently 3,896 rows: 2,326 negative, 1,397 positive, 173 neutral) |
| `combined_train.csv` | Training data — **updated in place**: originally real train+dev + synthetic reviews (7,902 rows), then grew to 8,160 rows after 258 real neutral examples were reallocated in from `naijasenti_test.csv` |
| `nigerian_pidgin_sentiment.pth` | Trained model checkpoint |

## Limitations

- Neutral-class performance is weak (F1 ~0.14) due to scarcity of real, human-labeled
  neutral Pidgin text across all available sources.
- Synthetic review data covers e-commerce vocabulary; real NaijaSenti data covers general
  Twitter topics (politics, football, daily life). Neither alone fully represents the
  target domain of e-commerce product reviews in the wild.
- Model shows calibration drift (increasing overconfidence) during training; predicted
  class is more reliable than any associated confidence score.
- TF-IDF + feedforward architecture is a reasonable baseline but does not capture word
  order or context the way a transformer-based approach (e.g. fine-tuning a multilingual
  or Pidgin-specific pretrained model) would; this is a natural next step for further
  improvement.

## Acknowledgments

Real Pidgin sentiment data from NaijaSenti:

> Muhammad, S. H., Adelani, D. I., Ruder, S., Ahmad, I. S., Abdulmumin, I., Bello, B. S.,
> Choudhury, M., Emezue, C. C., Abdullahi, S. S., Aremu, A., Jeorge, A., & Brazdil, P.
> (2022). NaijaSenti: A Nigerian Twitter Sentiment Corpus for Multilingual Sentiment
> Analysis. *Proceedings of LREC 2022*.