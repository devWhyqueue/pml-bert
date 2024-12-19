# Balancing experiments

Other hyperparameters are fixed as follows:

- `learning_rate=1e-4`
- `batch_size=512`
- `preprocessing=False`
- `sequence_len=256`
- `weight_decay=0.0`
- `num_epochs=1`
- `optimizer=AdamW`
- `loss=BCEWithLogitsLoss`

## Config(pos_proportion=as_is) (~0.06) took 40 min

#### Training Dataset

- **Loss:** 0.2229
- **Classification Report:**
    - Class 0:
        - Precision: 0.99
        - Recall: 0.98
        - F1-Score: 0.98
        - Support: 1,698,440
    - Class 1:
        - Precision: 0.69
        - Recall: 0.78
        - F1-Score: 0.73
        - Support: 106,440
    - Overall:
        - Accuracy: 0.97
        - Macro Avg - Precision: 0.84, Recall: 0.88, F1-Score: 0.86
        - Weighted Avg - Precision: 0.97, Recall: 0.97, F1-Score: 0.97
- **ROC-AUC Score:** 0.9828

#### Validation Dataset

- **Loss:** 0.2234
- **Classification Report:**
    - Class 0:
        - Precision: 0.99
        - Recall: 0.98
        - F1-Score: 0.98
        - Support: 91,671
    - Class 1:
        - Precision: 0.67
        - Recall: 0.77
        - F1-Score: 0.72
        - Support: 5,649
    - Overall:
        - Accuracy: 0.97
        - Macro Avg - Precision: 0.83, Recall: 0.87, F1-Score: 0.85
        - Weighted Avg - Precision: 0.97, Recall: 0.97, F1-Score: 0.97
- **ROC-AUC Score:** 0.9803

#### Testing Dataset

- **Loss:** 0.2259
- **Classification Report:**
    - Class 0:
        - Precision: 0.99
        - Recall: 0.98
        - F1-Score: 0.98
        - Support: 91,532
    - Class 1:
        - Precision: 0.69
        - Recall: 0.77
        - F1-Score: 0.73
        - Support: 5,788
    - Overall:
        - Accuracy: 0.97
        - Macro Avg - Precision: 0.84, Recall: 0.88, F1-Score: 0.85
        - Weighted Avg - Precision: 0.97, Recall: 0.97, F1-Score: 0.97
- **ROC-AUC Score:** 0.9810

## Config(pos_proportion=0.1) took 25 min

#### Training Dataset

- **Loss:** 0.2377
- **Classification Report:**
    - Class 0:
        - Precision: 0.98
        - Recall: 0.98
        - F1-Score: 0.98
        - Support: 957,942
    - Class 1:
        - Precision: 0.79
        - Recall: 0.80
        - F1-Score: 0.79
        - Support: 106,438
    - Overall:
        - Accuracy: 0.96
        - Macro Avg - Precision: 0.88, Recall: 0.89, F1-Score: 0.89
        - Weighted Avg - Precision: 0.96, Recall: 0.96, F1-Score: 0.96
- **ROC-AUC Score:** 0.9837

---

#### Validation Dataset

- **Loss:** 0.2248
- **Classification Report:**
    - Class 0:
        - Precision: 0.99
        - Recall: 0.97
        - F1-Score: 0.98
        - Support: 91,671
    - Class 1:
        - Precision: 0.65
        - Recall: 0.78
        - F1-Score: 0.71
        - Support: 5,649
    - Overall:
        - Accuracy: 0.96
        - Macro Avg - Precision: 0.82, Recall: 0.88, F1-Score: 0.85
        - Weighted Avg - Precision: 0.97, Recall: 0.96, F1-Score: 0.96
- **ROC-AUC Score:** 0.9792

---

#### Testing Dataset

- **Loss:** 0.2275
- **Classification Report:**
    - Class 0:
        - Precision: 0.99
        - Recall: 0.97
        - F1-Score: 0.98
        - Support: 91,532
    - Class 1:
        - Precision: 0.66
        - Recall: 0.78
        - F1-Score: 0.72
        - Support: 5,788
    - Overall:
        - Accuracy: 0.96
        - Macro Avg - Precision: 0.82, Recall: 0.88, F1-Score: 0.85
        - Weighted Avg - Precision: 0.97, Recall: 0.96, F1-Score: 0.96
- **ROC-AUC Score:** 0.9798

## Config(pos_proportion=0.25)

#### Training Dataset

- **Loss:** 0.3120
- **Classification Report:**
    - Class 0:
        - Precision: 0.97
        - Recall: 0.94
        - F1-Score: 0.96
        - Support: 319,314
    - Class 1:
        - Precision: 0.83
        - Recall: 0.93
        - F1-Score: 0.88
        - Support: 106,438
    - Overall:
        - Accuracy: 0.93
        - Macro Avg - Precision: 0.90, Recall: 0.93, F1-Score: 0.92
        - Weighted Avg - Precision: 0.94, Recall: 0.93, F1-Score: 0.94
- **ROC-AUC Score:** 0.9822

---

#### Validation Dataset

- **Loss:** 0.2413
- **Classification Report:**
    - Class 0:
        - Precision: 0.99
        - Recall: 0.93
        - F1-Score: 0.96
        - Support: 91,671
    - Class 1:
        - Precision: 0.45
        - Recall: 0.91
        - F1-Score: 0.61
        - Support: 5,649
    - Overall:
        - Accuracy: 0.93
        - Macro Avg - Precision: 0.72, Recall: 0.92, F1-Score: 0.78
        - Weighted Avg - Precision: 0.96, Recall: 0.93, F1-Score: 0.94
- **ROC-AUC Score:** 0.9781

---

#### Testing Dataset

- **Loss:** 0.2440
- **Classification Report:**
    - Class 0:
        - Precision: 0.99
        - Recall: 0.93
        - F1-Score: 0.96
        - Support: 91,532
    - Class 1:
        - Precision: 0.47
        - Recall: 0.91
        - F1-Score: 0.62
        - Support: 5,788
    - Overall:
        - Accuracy: 0.93
        - Macro Avg - Precision: 0.73, Recall: 0.92, F1-Score: 0.79
        - Weighted Avg - Precision: 0.96, Recall: 0.93, F1-Score: 0.94
- **ROC-AUC Score:** 0.9784

## Conclusion

Increasing the positive proportion has a negative impact on the model's performance.
However, for `pos_proportion=0.1` the difference is negligible, but we achieve a speedup of 38% (due to fewer data to
process).
Therefore, we will use `pos_proportion=0.1` for the rest of the experiments.
