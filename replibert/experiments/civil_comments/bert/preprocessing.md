# Preprocessing experiments

Other hyperparameters are fixed as follows:

- `learning_rate=1e-4`
- `batch_size=512`
- `pos_proportion=as_is`
- `sequence_len=256`
- `weight_decay=0.0`
- `num_epochs=1`
- `optimizer=AdamW`
- `loss=BCEWithLogitsLoss`

## Config(preprocessing=False)

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

## Config(preprocessing=True)

#### Training Dataset

- **Loss:** 0.2264
- **Classification Report:**
    - Class 0:
        - Precision: 0.98
        - Recall: 0.99
        - F1-Score: 0.98
        - Support: 1,698,438
    - Class 1:
        - Precision: 0.74
        - Recall: 0.67
        - F1-Score: 0.70
        - Support: 106,438
    - Overall:
        - Accuracy: 0.97
        - Macro Avg - Precision: 0.86, Recall: 0.83, F1-Score: 0.84
        - Weighted Avg - Precision: 0.97, Recall: 0.97, F1-Score: 0.97
- **ROC-AUC Score:** 0.9777

---

#### Validation Dataset

- **Loss:** 0.2275
- **Classification Report:**
    - Class 0:
        - Precision: 0.98
        - Recall: 0.98
        - F1-Score: 0.98
        - Support: 91,671
    - Class 1:
        - Precision: 0.72
        - Recall: 0.66
        - F1-Score: 0.69
        - Support: 5,649
    - Overall:
        - Accuracy: 0.97
        - Macro Avg - Precision: 0.85, Recall: 0.82, F1-Score: 0.84
        - Weighted Avg - Precision: 0.96, Recall: 0.97, F1-Score: 0.96
- **ROC-AUC Score:** 0.9738

---

#### Testing Dataset

- **Loss:** 0.2304
- **Classification Report:**
    - Class 0:
        - Precision: 0.98
        - Recall: 0.98
        - F1-Score: 0.98
        - Support: 91,532
    - Class 1:
        - Precision: 0.73
        - Recall: 0.66
        - F1-Score: 0.69
        - Support: 5,788
    - Overall:
        - Accuracy: 0.97
        - Macro Avg - Precision: 0.85, Recall: 0.82, F1-Score: 0.84
        - Weighted Avg - Precision: 0.96, Recall: 0.97, F1-Score: 0.96
- **ROC-AUC Score:** 0.9748

## Conclusion

Preprocessing has no positive impact on model performance.
It even shows a slight negative impact on the model's class 1 F1-score (0.72 -> 0.69).
