# Learning rates experiments

Other hyperparameters are fixed as follows:

- `batch_size=512`
- `prepocessing=False`
- `pos_proportion=as_is`
- `sequence_len=256`
- `weight_decay=0.0`
- `num_epochs=1`
- `optimizer=AdamW`
- `loss=BCEWithLogitsLoss`

## Config(lr=1e-7)

#### Training Dataset

- **Loss:** 0.4391
- **Classification Report:**
    - Class 0:
        - Precision: 0.94
        - Recall: 1.00
        - F1-Score: 0.97
        - Support: 1,698,440
    - Class 1:
        - Precision: 0.05
        - Recall: 0.00
        - F1-Score: 0.01
        - Support: 106,440
    - Overall:
        - Accuracy: 0.94
        - Macro Avg - Precision: 0.50, Recall: 0.50, F1-Score: 0.49
        - Weighted Avg - Precision: 0.89, Recall: 0.94, F1-Score: 0.91
- **ROC-AUC Score:** 0.5066

#### Validation Dataset

- **Loss:** 0.4368
- **Classification Report:**
    - Class 0:
        - Precision: 0.94
        - Recall: 1.00
        - F1-Score: 0.97
        - Support: 91,671
    - Class 1:
        - Precision: 0.06
        - Recall: 0.00
        - F1-Score: 0.01
        - Support: 5,649
    - Overall:
        - Accuracy: 0.94
        - Macro Avg - Precision: 0.50, Recall: 0.50, F1-Score: 0.49
        - Weighted Avg - Precision: 0.89, Recall: 0.94, F1-Score: 0.91
- **ROC-AUC Score:** 0.5066

#### Testing Dataset

- **Loss:** 0.4389
- **Classification Report:**
    - Class 0:
        - Precision: 0.94
        - Recall: 1.00
        - F1-Score: 0.97
        - Support: 91,532
    - Class 1:
        - Precision: 0.04
        - Recall: 0.00
        - F1-Score: 0.01
        - Support: 5,788
    - Overall:
        - Accuracy: 0.94
        - Macro Avg - Precision: 0.49, Recall: 0.50, F1-Score: 0.49
        - Weighted Avg - Precision: 0.89, Recall: 0.94, F1-Score: 0.91
- **ROC-AUC Score:** 0.5104

## Config(lr=1e-6)

#### Training Dataset

- **Loss:** 0.2312
- **Classification Report:**
    - Class 0:
        - Precision: 0.98
        - Recall: 0.98
        - F1-Score: 0.98
        - Support: 1,698,440
    - Class 1:
        - Precision: 0.69
        - Recall: 0.70
        - F1-Score: 0.70
        - Support: 106,440
    - Overall:
        - Accuracy: 0.96
        - Macro Avg - Precision: 0.84, Recall: 0.84, F1-Score: 0.84
        - Weighted Avg - Precision: 0.96, Recall: 0.96, F1-Score: 0.96
- **ROC-AUC Score:** 0.9737

#### Validation Dataset

- **Loss:** 0.2276
- **Classification Report:**
    - Class 0:
        - Precision: 0.98
        - Recall: 0.98
        - F1-Score: 0.98
        - Support: 91,671
    - Class 1:
        - Precision: 0.69
        - Recall: 0.71
        - F1-Score: 0.70
        - Support: 5,649
    - Overall:
        - Accuracy: 0.96
        - Macro Avg - Precision: 0.84, Recall: 0.84, F1-Score: 0.84
        - Weighted Avg - Precision: 0.97, Recall: 0.96, F1-Score: 0.96
- **ROC-AUC Score:** 0.9747

#### Testing Dataset

- **Loss:** 0.2308
- **Classification Report:**
    - Class 0:
        - Precision: 0.98
        - Recall: 0.98
        - F1-Score: 0.98
        - Support: 91,532
    - Class 1:
        - Precision: 0.70
        - Recall: 0.70
        - F1-Score: 0.70
        - Support: 5,788
    - Overall:
        - Accuracy: 0.96
        - Macro Avg - Precision: 0.84, Recall: 0.84, F1-Score: 0.84
        - Weighted Avg - Precision: 0.96, Recall: 0.96, F1-Score: 0.96
- **ROC-AUC Score:** 0.9746

## Config(lr=1e-5)

#### Training Dataset

- **Loss:** 0.2243
- **Classification Report:**
    - Class 0:
        - Precision: 0.99
        - Recall: 0.98
        - F1-Score: 0.98
        - Support: 1,698,440
    - Class 1:
        - Precision: 0.67
        - Recall: 0.79
        - F1-Score: 0.73
        - Support: 106,440
    - Overall:
        - Accuracy: 0.96
        - Macro Avg - Precision: 0.83, Recall: 0.88, F1-Score: 0.85
        - Weighted Avg - Precision: 0.97, Recall: 0.96, F1-Score: 0.97
- **ROC-AUC Score:** 0.9821

#### Validation Dataset

- **Loss:** 0.2237
- **Classification Report:**
    - Class 0:
        - Precision: 0.99
        - Recall: 0.98
        - F1-Score: 0.98
        - Support: 91,671
    - Class 1:
        - Precision: 0.66
        - Recall: 0.78
        - F1-Score: 0.71
        - Support: 5,649
    - Overall:
        - Accuracy: 0.96
        - Macro Avg - Precision: 0.82, Recall: 0.88, F1-Score: 0.85
        - Weighted Avg - Precision: 0.97, Recall: 0.96, F1-Score: 0.96
- **ROC-AUC Score:** 0.9803

#### Testing Dataset

- **Loss:** 0.2262
- **Classification Report:**
    - Class 0:
        - Precision: 0.99
        - Recall: 0.98
        - F1-Score: 0.98
        - Support: 91,532
    - Class 1:
        - Precision: 0.67
        - Recall: 0.78
        - F1-Score: 0.72
        - Support: 5,788
    - Overall:
        - Accuracy: 0.96
        - Macro Avg - Precision: 0.83, Recall: 0.88, F1-Score: 0.85
        - Weighted Avg - Precision: 0.97, Recall: 0.96, F1-Score: 0.97
- **ROC-AUC Score:** 0.9808

## Config(lr=1e-4)

#### Training Dataset

- **Loss:** 0.2220
- **Classification Report:**
    - Class 0:
        - Precision: 0.98
        - Recall: 0.99
        - F1-Score: 0.98
        - Support: 1,698,440
    - Class 1:
        - Precision: 0.77
        - Recall: 0.69
        - F1-Score: 0.73
        - Support: 106,440
    - Overall:
        - Accuracy: 0.97
        - Macro Avg - Precision: 0.87, Recall: 0.84, F1-Score: 0.85
        - Weighted Avg - Precision: 0.97, Recall: 0.97, F1-Score: 0.97
- **ROC-AUC Score:** 0.9822

#### Validation Dataset

- **Loss:** 0.2238
- **Classification Report:**
    - Class 0:
        - Precision: 0.98
        - Recall: 0.99
        - F1-Score: 0.98
        - Support: 91,671
    - Class 1:
        - Precision: 0.75
        - Recall: 0.67
        - F1-Score: 0.71
        - Support: 5,649
    - Overall:
        - Accuracy: 0.97
        - Macro Avg - Precision: 0.87, Recall: 0.83, F1-Score: 0.85
        - Weighted Avg - Precision: 0.97, Recall: 0.97, F1-Score: 0.97
- **ROC-AUC Score:** 0.9786

#### Testing Dataset

- **Loss:** 0.2266
- **Classification Report:**
    - Class 0:
        - Precision: 0.98
        - Recall: 0.99
        - F1-Score: 0.98
        - Support: 91,532
    - Class 1:
        - Precision: 0.76
        - Recall: 0.67
        - F1-Score: 0.71
        - Support: 5,788
    - Overall:
        - Accuracy: 0.97
        - Macro Avg - Precision: 0.87, Recall: 0.83, F1-Score: 0.85
        - Weighted Avg - Precision: 0.97, Recall: 0.97, F1-Score: 0.97
- **ROC-AUC Score:** 0.9791

## Config(lr=1e-3)

#### Training Dataset

- **Loss:** 0.3359
- **Classification Report:**
    - Class 0:
        - Precision: 0.94
        - Recall: 1.00
        - F1-Score: 0.97
        - Support: 1,698,440
    - Class 1:
        - Precision: 0.00
        - Recall: 0.00
        - F1-Score: 0.00
        - Support: 106,440
    - Overall:
        - Accuracy: 0.94
        - Macro Avg - Precision: 0.47, Recall: 0.50, F1-Score: 0.48
        - Weighted Avg - Precision: 0.89, Recall: 0.94, F1-Score: 0.91
- **ROC-AUC Score:** 0.4987

#### Validation Dataset

- **Loss:** 0.3316
- **Classification Report:**
    - Class 0:
        - Precision: 0.94
        - Recall: 1.00
        - F1-Score: 0.97
        - Support: 91,671
    - Class 1:
        - Precision: 0.00
        - Recall: 0.00
        - F1-Score: 0.00
        - Support: 5,649
    - Overall:
        - Accuracy: 0.94
        - Macro Avg - Precision: 0.47, Recall: 0.50, F1-Score: 0.49
        - Weighted Avg - Precision: 0.89, Recall: 0.94, F1-Score: 0.91
- **ROC-AUC Score:** 0.4981

#### Testing Dataset

- **Loss:** 0.3353
- **Classification Report:**
    - Class 0:
        - Precision: 0.94
        - Recall: 1.00
        - F1-Score: 0.97
        - Support: 91,532
    - Class 1:
        - Precision: 0.00
        - Recall: 0.00
        - F1-Score: 0.00
        - Support: 5,788
    - Overall:
        - Accuracy: 0.94
        - Macro Avg - Precision: 0.47, Recall: 0.50, F1-Score: 0.48
        - Weighted Avg - Precision: 0.88, Recall: 0.94, F1-Score: 0.91
- **ROC-AUC Score:** 0.4970

---

## Conclusion

We tried $lr\_list=[1^{-7}, 1^{-6}, 1^{-5}, 1^{-4},1^{-3}]$.
Except the first and last, the rest looks promising.
We can start with the highest learning rate and then gradually decrease for multi-epoch training.
