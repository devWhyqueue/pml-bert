# Sequence length experiments

Other hyperparameters are fixed as follows:

- `learning_rate=1e-4`
- `batch_size=512`
- `preprocessing=False`
- `weight_decay=0.0`
- `epochs=1`
- `pos_proportion=0.1`
- `optimizer=AdamW`
- `loss=BCEWithLogitsLoss`

## Config(sequence_len=64) took 4 min

#### Training Dataset

- **Loss:** 0.2563
- **Classification Report:**
    - Class 0:
        - Precision: 0.97
        - Recall: 0.98
        - F1-Score: 0.97
        - Support: 957,942
    - Class 1:
        - Precision: 0.78
        - Recall: 0.72
        - F1-Score: 0.75
        - Support: 106,443
    - Overall:
        - Accuracy: 0.95
        - Macro Avg - Precision: 0.87, Recall: 0.85, F1-Score: 0.86
        - Weighted Avg - Precision: 0.95, Recall: 0.95, F1-Score: 0.95
- **ROC-AUC Score:** 0.9628

---

#### Validation Dataset

- **Loss:** 0.2391
- **Classification Report:**
    - Class 0:
        - Precision: 0.98
        - Recall: 0.98
        - F1-Score: 0.98
        - Support: 91,672
    - Class 1:
        - Precision: 0.64
        - Recall: 0.70
        - F1-Score: 0.66
        - Support: 5,649
    - Overall:
        - Accuracy: 0.96
        - Macro Avg - Precision: 0.81, Recall: 0.84, F1-Score: 0.82
        - Weighted Avg - Precision: 0.96, Recall: 0.96, F1-Score: 0.96
- **ROC-AUC Score:** 0.9566

---

#### Testing Dataset

- **Loss:** 0.2416
- **Classification Report:**
    - Class 0:
        - Precision: 0.98
        - Recall: 0.98
        - F1-Score: 0.98
        - Support: 91,533
    - Class 1:
        - Precision: 0.64
        - Recall: 0.71
        - F1-Score: 0.68
        - Support: 5,788
    - Overall:
        - Accuracy: 0.96
        - Macro Avg - Precision: 0.81, Recall: 0.84, F1-Score: 0.83
        - Weighted Avg - Precision: 0.96, Recall: 0.96, F1-Score: 0.96
- **ROC-AUC Score:** 0.9572

## Config(sequence_len=128) took 8 min

#### Training Dataset

- **Loss:** 0.2457
- **Classification Report:**
    - Class 0:
        - Precision: 0.98
        - Recall: 0.97
        - F1-Score: 0.97
        - Support: 957,942
    - Class 1:
        - Precision: 0.76
        - Recall: 0.81
        - F1-Score: 0.78
        - Support: 106,443
    - Overall:
        - Accuracy: 0.95
        - Macro Avg - Precision: 0.87, Recall: 0.89, F1-Score: 0.88
        - Weighted Avg - Precision: 0.96, Recall: 0.95, F1-Score: 0.96
- **ROC-AUC Score:** 0.9783

---

#### Validation Dataset

- **Loss:** 0.2308
- **Classification Report:**
    - Class 0:
        - Precision: 0.99
        - Recall: 0.97
        - F1-Score: 0.98
        - Support: 91,672
    - Class 1:
        - Precision: 0.61
        - Recall: 0.79
        - F1-Score: 0.69
        - Support: 5,649
    - Overall:
        - Accuracy: 0.96
        - Macro Avg - Precision: 0.80, Recall: 0.88, F1-Score: 0.84
        - Weighted Avg - Precision: 0.97, Recall: 0.96, F1-Score: 0.96
- **ROC-AUC Score:** 0.9745

---

#### Testing Dataset

- **Loss:** 0.2337
- **Classification Report:**
    - Class 0:
        - Precision: 0.99
        - Recall: 0.97
        - F1-Score: 0.98
        - Support: 91,533
    - Class 1:
        - Precision: 0.62
        - Recall: 0.80
        - F1-Score: 0.70
        - Support: 5,788
    - Overall:
        - Accuracy: 0.96
        - Macro Avg - Precision: 0.81, Recall: 0.88, F1-Score: 0.84
        - Weighted Avg - Precision: 0.97, Recall: 0.96, F1-Score: 0.96
- **ROC-AUC Score:** 0.9747

## Config(sequence_len=256) took 15 min

#### Training Dataset

- **Loss:** 0.2425
- **Classification Report:**
    - Class 0:
        - Precision: 0.98
        - Recall: 0.97
        - F1-Score: 0.98
        - Support: 957,942
    - Class 1:
        - Precision: 0.75
        - Recall: 0.83
        - F1-Score: 0.79
        - Support: 106,443
    - Overall:
        - Accuracy: 0.96
        - Macro Avg - Precision: 0.87, Recall: 0.90, F1-Score: 0.88
        - Weighted Avg - Precision: 0.96, Recall: 0.96, F1-Score: 0.96
- **ROC-AUC Score:** 0.9831

---

#### Validation Dataset

- **Loss:** 0.2284
- **Classification Report:**
    - Class 0:
        - Precision: 0.99
        - Recall: 0.97
        - F1-Score: 0.98
        - Support: 91,672
    - Class 1:
        - Precision: 0.61
        - Recall: 0.82
        - F1-Score: 0.70
        - Support: 5,649
    - Overall:
        - Accuracy: 0.96
        - Macro Avg - Precision: 0.80, Recall: 0.89, F1-Score: 0.84
        - Weighted Avg - Precision: 0.97, Recall: 0.96, F1-Score: 0.96
- **ROC-AUC Score:** 0.9797

---

#### Testing Dataset

- **Loss:** 0.2309
- **Classification Report:**
    - Class 0:
        - Precision: 0.99
        - Recall: 0.97
        - F1-Score: 0.98
        - Support: 91,533
    - Class 1:
        - Precision: 0.62
        - Recall: 0.82
        - F1-Score: 0.70
        - Support: 5,788
    - Overall:
        - Accuracy: 0.96
        - Macro Avg - Precision: 0.80, Recall: 0.89, F1-Score: 0.84
        - Weighted Avg - Precision: 0.97, Recall: 0.96, F1-Score: 0.96
- **ROC-AUC Score:** 0.9800

## Config(sequence_len=512, batch_size=128) took 30 min

#### Training Dataset

- **Loss:** 0.2342
- **Classification Report:**
    - Class 0:
        - Precision: 0.97
        - Recall: 0.99
        - F1-Score: 0.98
        - Support: 957,942
    - Class 1:
        - Precision: 0.85
        - Recall: 0.73
        - F1-Score: 0.78
        - Support: 106,443
    - Overall:
        - Accuracy: 0.96
        - Macro Avg - Precision: 0.91, Recall: 0.86, F1-Score: 0.88
        - Weighted Avg - Precision: 0.96, Recall: 0.96, F1-Score: 0.96
- **ROC-AUC Score:** 0.9846

---

#### Validation Dataset

- **Loss:** 0.2237
- **Classification Report:**
    - Class 0:
        - Precision: 0.98
        - Recall: 0.98
        - F1-Score: 0.98
        - Support: 91,672
    - Class 1:
        - Precision: 0.71
        - Recall: 0.71
        - F1-Score: 0.71
        - Support: 5,649
    - Overall:
        - Accuracy: 0.97
        - Macro Avg - Precision: 0.85, Recall: 0.85, F1-Score: 0.85
        - Weighted Avg - Precision: 0.97, Recall: 0.97, F1-Score: 0.97
- **ROC-AUC Score:** 0.9788

---

#### Testing Dataset

- **Loss:** 0.2264
- **Classification Report:**
    - Class 0:
        - Precision: 0.98
        - Recall: 0.98
        - F1-Score: 0.98
        - Support: 91,533
    - Class 1:
        - Precision: 0.72
        - Recall: 0.72
        - F1-Score: 0.72
        - Support: 5,788
    - Overall:
        - Accuracy: 0.97
        - Macro Avg - Precision: 0.85, Recall: 0.85, F1-Score: 0.85
        - Weighted Avg - Precision: 0.97, Recall: 0.97, F1-Score: 0.97
- **ROC-AUC Score:** 0.9795

## Conclusion

BERT's default sequence length is 512.
We can reduce to 128 without significant performance loss, achieving a speedup of 4x and enabling larger batch sizes.
