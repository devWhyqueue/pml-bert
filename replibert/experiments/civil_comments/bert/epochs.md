# Epoch experiments

Other hyperparameters are fixed as follows:

- `learning_rate=1e-4`
- `batch_size=512`
- `preprocessing=False`
- `sequence_len=256`
- `weight_decay=0.0`
- `pos_proportion=0.1`
- `optimizer=AdamW`
- `loss=BCEWithLogitsLoss`

## Config(num_epochs=1)

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

## Config(num_epochs=2)

#### Training Dataset

- **Loss:** 0.2303
- **Classification Report:**
    - Class 0:
        - Precision: 0.98
        - Recall: 0.97
        - F1-Score: 0.98
        - Support: 957,942
    - Class 1:
        - Precision: 0.76
        - Recall: 0.86
        - F1-Score: 0.81
        - Support: 106,443
    - Overall:
        - Accuracy: 0.96
        - Macro Avg - Precision: 0.87, Recall: 0.92, F1-Score: 0.89
        - Weighted Avg - Precision: 0.96, Recall: 0.96, F1-Score: 0.96
- **ROC-AUC Score:** 0.9871

---

#### Validation Dataset

- **Loss:** 0.2269
- **Classification Report:**
    - Class 0:
        - Precision: 0.99
        - Recall: 0.97
        - F1-Score: 0.98
        - Support: 91,672
    - Class 1:
        - Precision: 0.60
        - Recall: 0.82
        - F1-Score: 0.70
        - Support: 5,649
    - Overall:
        - Accuracy: 0.96
        - Macro Avg - Precision: 0.80, Recall: 0.90, F1-Score: 0.84
        - Weighted Avg - Precision: 0.97, Recall: 0.96, F1-Score: 0.96
- **ROC-AUC Score:** 0.9794

---

#### Testing Dataset

- **Loss:** 0.2298
- **Classification Report:**
    - Class 0:
        - Precision: 0.99
        - Recall: 0.97
        - F1-Score: 0.98
        - Support: 91,533
    - Class 1:
        - Precision: 0.61
        - Recall: 0.82
        - F1-Score: 0.70
        - Support: 5,788
    - Overall:
        - Accuracy: 0.96
        - Macro Avg - Precision: 0.80, Recall: 0.89, F1-Score: 0.84
        - Weighted Avg - Precision: 0.97, Recall: 0.96, F1-Score: 0.96
- **ROC-AUC Score:** 0.9794

## Config(num_epochs=3)

#### Training Dataset

- **Loss:** 0.2182
- **Classification Report:**
    - Class 0:
        - Precision: 0.99
        - Recall: 0.97
        - F1-Score: 0.98
        - Support: 957,942
    - Class 1:
        - Precision: 0.77
        - Recall: 0.91
        - F1-Score: 0.83
        - Support: 106,443
    - Overall:
        - Accuracy: 0.96
        - Macro Avg - Precision: 0.88, Recall: 0.94, F1-Score: 0.91
        - Weighted Avg - Precision: 0.97, Recall: 0.96, F1-Score: 0.96
- **ROC-AUC Score:** 0.9904

---

#### Validation Dataset

- **Loss:** 0.2303
- **Classification Report:**
    - Class 0:
        - Precision: 0.99
        - Recall: 0.96
        - F1-Score: 0.98
        - Support: 91,672
    - Class 1:
        - Precision: 0.59
        - Recall: 0.83
        - F1-Score: 0.69
        - Support: 5,649
    - Overall:
        - Accuracy: 0.96
        - Macro Avg - Precision: 0.79, Recall: 0.90, F1-Score: 0.83
        - Weighted Avg - Precision: 0.97, Recall: 0.96, F1-Score: 0.96
- **ROC-AUC Score:** 0.9784

---

#### Testing Dataset

- **Loss:** 0.2333
- **Classification Report:**
    - Class 0:
        - Precision: 0.99
        - Recall: 0.96
        - F1-Score: 0.98
        - Support: 91,533
    - Class 1:
        - Precision: 0.59
        - Recall: 0.83
        - F1-Score: 0.69
        - Support: 5,788
    - Overall:
        - Accuracy: 0.96
        - Macro Avg - Precision: 0.79, Recall: 0.90, F1-Score: 0.83
        - Weighted Avg - Precision: 0.97, Recall: 0.96, F1-Score: 0.96
- **ROC-AUC Score:** 0.9783

## Config(epochs=5, weight_decay=1.0)

#### Training Dataset

- **Loss:** 0.2097
- **Classification Report:**
    - Class 0:
        - Precision: 0.99
        - Recall: 0.97
        - F1-Score: 0.98
        - Support: 957,942
    - Class 1:
        - Precision: 0.77
        - Recall: 0.94
        - F1-Score: 0.84
        - Support: 106,443
    - Overall:
        - Accuracy: 0.97
        - Macro Avg - Precision: 0.88, Recall: 0.95, F1-Score: 0.91
        - Weighted Avg - Precision: 0.97, Recall: 0.97, F1-Score: 0.97
- **ROC-AUC Score:** 0.9924

---

#### Validation Dataset

- **Loss:** 0.2361
- **Classification Report:**
    - Class 0:
        - Precision: 0.99
        - Recall: 0.96
        - F1-Score: 0.98
        - Support: 91,672
    - Class 1:
        - Precision: 0.57
        - Recall: 0.83
        - F1-Score: 0.68
        - Support: 5,649
    - Overall:
        - Accuracy: 0.95
        - Macro Avg - Precision: 0.78, Recall: 0.89, F1-Score: 0.83
        - Weighted Avg - Precision: 0.96, Recall: 0.95, F1-Score: 0.96
- **ROC-AUC Score:** 0.9768

---

#### Testing Dataset

- **Loss:** 0.2395
- **Classification Report:**
    - Class 0:
        - Precision: 0.99
        - Recall: 0.96
        - F1-Score: 0.98
        - Support: 91,533
    - Class 1:
        - Precision: 0.59
        - Recall: 0.83
        - F1-Score: 0.69
        - Support: 5,788
    - Overall:
        - Accuracy: 0.95
        - Macro Avg - Precision: 0.79, Recall: 0.90, F1-Score: 0.83
        - Weighted Avg - Precision: 0.96, Recall: 0.95, F1-Score: 0.96
- **ROC-AUC Score:** 0.9766

## Conclusion

More epochs do not lead to better model performance.
Even with weight decay, the model performance does not improve.
