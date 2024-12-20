# Regularization experiments

Other hyperparameters are fixed as follows:

- `learning_rate=1e-4`
- `batch_size=512`
- `preprocessing=False`
- `sequence_len=256`
- `epochs=2`
- `pos_proportion=0.1`
- `optimizer=AdamW`
- `loss=BCEWithLogitsLoss`

## Config(weight_decay=0.0)

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

## Config(weight_decay=0.1)

#### Training Dataset

- **Loss:** 0.2351
- **Classification Report:**
    - Class 0:
        - Precision: 0.98
        - Recall: 0.97
        - F1-Score: 0.98
        - Support: 957,942
    - Class 1:
        - Precision: 0.76
        - Recall: 0.85
        - F1-Score: 0.80
        - Support: 106,443
    - Overall:
        - Accuracy: 0.96
        - Macro Avg - Precision: 0.87, Recall: 0.91, F1-Score: 0.89
        - Weighted Avg - Precision: 0.96, Recall: 0.96, F1-Score: 0.96
- **ROC-AUC Score:** 0.9853

---

#### Validation Dataset

- **Loss:** 0.2268
- **Classification Report:**
    - Class 0:
        - Precision: 0.99
        - Recall: 0.97
        - F1-Score: 0.98
        - Support: 91,672
    - Class 1:
        - Precision: 0.60
        - Recall: 0.83
        - F1-Score: 0.70
        - Support: 5,649
    - Overall:
        - Accuracy: 0.96
        - Macro Avg - Precision: 0.80, Recall: 0.90, F1-Score: 0.84
        - Weighted Avg - Precision: 0.97, Recall: 0.96, F1-Score: 0.96
- **ROC-AUC Score:** 0.9797

---

#### Testing Dataset

- **Loss:** 0.2297
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
        - Macro Avg - Precision: 0.80, Recall: 0.90, F1-Score: 0.84
        - Weighted Avg - Precision: 0.97, Recall: 0.96, F1-Score: 0.96
- **ROC-AUC Score:** 0.9796

## Config(weight_decay=1.0)

#### Training Dataset

- **Loss:** 0.2335
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
        - Macro Avg - Precision: 0.87, Recall: 0.91, F1-Score: 0.89
        - Weighted Avg - Precision: 0.96, Recall: 0.96, F1-Score: 0.96
- **ROC-AUC Score:** 0.9864

---

#### Validation Dataset

- **Loss:** 0.2265
- **Classification Report:**
    - Class 0:
        - Precision: 0.99
        - Recall: 0.97
        - F1-Score: 0.98
        - Support: 91,672
    - Class 1:
        - Precision: 0.60
        - Recall: 0.83
        - F1-Score: 0.70
        - Support: 5,649
    - Overall:
        - Accuracy: 0.96
        - Macro Avg - Precision: 0.80, Recall: 0.90, F1-Score: 0.84
        - Weighted Avg - Precision: 0.97, Recall: 0.96, F1-Score: 0.96
- **ROC-AUC Score:** 0.9799

---

#### Testing Dataset

- **Loss:** 0.2295
- **Classification Report:**
    - Class 0:
        - Precision: 0.99
        - Recall: 0.97
        - F1-Score: 0.98
        - Support: 91,533
    - Class 1:
        - Precision: 0.61
        - Recall: 0.83
        - F1-Score: 0.70
        - Support: 5,788
    - Overall:
        - Accuracy: 0.96
        - Macro Avg - Precision: 0.80, Recall: 0.90, F1-Score: 0.84
        - Weighted Avg - Precision: 0.97, Recall: 0.96, F1-Score: 0.96
- **ROC-AUC Score:** 0.9799

## Config(weight_decay=10.0)

#### Training Dataset

- **Loss:** 0.2458
- **Classification Report:**
    - Class 0:
        - Precision: 0.98
        - Recall: 0.97
        - F1-Score: 0.97
        - Support: 957,942
    - Class 1:
        - Precision: 0.77
        - Recall: 0.78
        - F1-Score: 0.77
        - Support: 106,443
    - Overall:
        - Accuracy: 0.95
        - Macro Avg - Precision: 0.87, Recall: 0.88, F1-Score: 0.87
        - Weighted Avg - Precision: 0.95, Recall: 0.95, F1-Score: 0.95
- **ROC-AUC Score:** 0.9795

---

#### Validation Dataset

- **Loss:** 0.2298
- **Classification Report:**
    - Class 0:
        - Precision: 0.99
        - Recall: 0.97
        - F1-Score: 0.98
        - Support: 91,672
    - Class 1:
        - Precision: 0.63
        - Recall: 0.78
        - F1-Score: 0.70
        - Support: 5,649
    - Overall:
        - Accuracy: 0.96
        - Macro Avg - Precision: 0.81, Recall: 0.88, F1-Score: 0.84
        - Weighted Avg - Precision: 0.97, Recall: 0.96, F1-Score: 0.96
- **ROC-AUC Score:** 0.9764

---

#### Testing Dataset

- **Loss:** 0.2326
- **Classification Report:**
    - Class 0:
        - Precision: 0.99
        - Recall: 0.97
        - F1-Score: 0.98
        - Support: 91,533
    - Class 1:
        - Precision: 0.64
        - Recall: 0.77
        - F1-Score: 0.70
        - Support: 5,788
    - Overall:
        - Accuracy: 0.96
        - Macro Avg - Precision: 0.81, Recall: 0.87, F1-Score: 0.84
        - Weighted Avg - Precision: 0.97, Recall: 0.96, F1-Score: 0.96
- **ROC-AUC Score:** 0.9768

## Config(weight_decay=100.0)

#### Training Dataset

- **Loss:** 0.6076
- **Classification Report:**
    - Class 0:
        - Precision: 0.90
        - Recall: 1.00
        - F1-Score: 0.95
        - Support: 957,942
    - Class 1:
        - Precision: 0.00
        - Recall: 0.00
        - F1-Score: 0.00
        - Support: 106,443
    - Overall:
        - Accuracy: 0.90
        - Macro Avg - Precision: 0.45, Recall: 0.50, F1-Score: 0.47
        - Weighted Avg - Precision: 0.81, Recall: 0.90, F1-Score: 0.85
- **ROC-AUC Score:** 0.5153

---

#### Validation Dataset

- **Loss:** 0.6005
- **Classification Report:**
    - Class 0:
        - Precision: 0.94
        - Recall: 1.00
        - F1-Score: 0.97
        - Support: 91,672
    - Class 1:
        - Precision: 0.00
        - Recall: 0.00
        - F1-Score: 0.00
        - Support: 5,649
    - Overall:
        - Accuracy: 0.94
        - Macro Avg - Precision: 0.47, Recall: 0.50, F1-Score: 0.49
        - Weighted Avg - Precision: 0.89, Recall: 0.94, F1-Score: 0.91
- **ROC-AUC Score:** 0.5174

---

#### Testing Dataset

- **Loss:** 0.6009
- **Classification Report:**
    - Class 0:
        - Precision: 0.94
        - Recall: 1.00
        - F1-Score: 0.97
        - Support: 91,533
    - Class 1:
        - Precision: 0.00
        - Recall: 0.00
        - F1-Score: 0.00
        - Support: 5,788
    - Overall:
        - Accuracy: 0.94
        - Macro Avg - Precision: 0.47, Recall: 0.50, F1-Score: 0.48
        - Weighted Avg - Precision: 0.88, Recall: 0.94, F1-Score: 0.91
- **ROC-AUC Score:** 0.5153

## Conclusion

Weight decay has no significant effect on the model performance.
Only unreasonably high values of weight decay lead to a loss of learning capacity.
