# Batch size experiments

Other hyperparameters are fixed as follows:

- `learning_rate=1e-4`
- `prepocessing=False`
- `pos_proportion=0.1`
- `sequence_len=128`
- `weight_decay=0.0`
- `num_epochs=1`
- `optimizer=AdamW`
- `loss=BCEWithLogitsLoss`

## Config(batch_size=64) took 8 min 24 sec

#### Training Dataset

- **Loss:** 0.2393
- **Classification Report:**
    - Class 0:
        - Precision: 0.98
        - Recall: 0.97
        - F1-Score: 0.98
        - Support: 957,942
    - Class 1:
        - Precision: 0.77
        - Recall: 0.81
        - F1-Score: 0.79
        - Support: 106,442
    - Overall:
        - Accuracy: 0.96
        - Macro Avg - Precision: 0.87, Recall: 0.89, F1-Score: 0.88
        - Weighted Avg - Precision: 0.96, Recall: 0.96, F1-Score: 0.96
- **ROC-AUC Score:** 0.9802

---

#### Validation Dataset

- **Loss:** 0.2303
- **Classification Report:**
    - Class 0:
        - Precision: 0.99
        - Recall: 0.97
        - F1-Score: 0.98
        - Support: 91,671
    - Class 1:
        - Precision: 0.62
        - Recall: 0.78
        - F1-Score: 0.69
        - Support: 5,649
    - Overall:
        - Accuracy: 0.96
        - Macro Avg - Precision: 0.80, Recall: 0.87, F1-Score: 0.83
        - Weighted Avg - Precision: 0.96, Recall: 0.96, F1-Score: 0.96
- **ROC-AUC Score:** 0.9732

---

#### Testing Dataset

- **Loss:** 0.2337
- **Classification Report:**
    - Class 0:
        - Precision: 0.99
        - Recall: 0.97
        - F1-Score: 0.98
        - Support: 91,532
    - Class 1:
        - Precision: 0.62
        - Recall: 0.78
        - F1-Score: 0.70
        - Support: 5,788
    - Overall:
        - Accuracy: 0.96
        - Macro Avg - Precision: 0.81, Recall: 0.88, F1-Score: 0.84
        - Weighted Avg - Precision: 0.96, Recall: 0.96, F1-Score: 0.96
- **ROC-AUC Score:** 0.9733

## Config(batch_size=256) took 7 min

#### Training Dataset

- **Loss:** 0.2421
- **Classification Report:**
    - Class 0:
        - Precision: 0.98
        - Recall: 0.97
        - F1-Score: 0.98
        - Support: 957,942
    - Class 1:
        - Precision: 0.77
        - Recall: 0.80
        - F1-Score: 0.79
        - Support: 106,442
    - Overall:
        - Accuracy: 0.96
        - Macro Avg - Precision: 0.87, Recall: 0.89, F1-Score: 0.88
        - Weighted Avg - Precision: 0.96, Recall: 0.96, F1-Score: 0.96
- **ROC-AUC Score:** 0.9795

---

#### Validation Dataset

- **Loss:** 0.2291
- **Classification Report:**
    - Class 0:
        - Precision: 0.99
        - Recall: 0.97
        - F1-Score: 0.98
        - Support: 91,671
    - Class 1:
        - Precision: 0.63
        - Recall: 0.78
        - F1-Score: 0.70
        - Support: 5,649
    - Overall:
        - Accuracy: 0.96
        - Macro Avg - Precision: 0.81, Recall: 0.88, F1-Score: 0.84
        - Weighted Avg - Precision: 0.97, Recall: 0.96, F1-Score: 0.96
- **ROC-AUC Score:** 0.9746

---

#### Testing Dataset

- **Loss:** 0.2323
- **Classification Report:**
    - Class 0:
        - Precision: 0.99
        - Recall: 0.97
        - F1-Score: 0.98
        - Support: 91,532
    - Class 1:
        - Precision: 0.64
        - Recall: 0.79
        - F1-Score: 0.70
        - Support: 5,788
    - Overall:
        - Accuracy: 0.96
        - Macro Avg - Precision: 0.81, Recall: 0.88, F1-Score: 0.84
        - Weighted Avg - Precision: 0.97, Recall: 0.96, F1-Score: 0.96
- **ROC-AUC Score:** 0.9748

## Config(batch_size=1024) took 7 min

#### Training Dataset

- **Loss:** 0.2479
- **Classification Report:**
    - Class 0:
        - Precision: 0.97
        - Recall: 0.98
        - F1-Score: 0.97
        - Support: 957,942
    - Class 1:
        - Precision: 0.79
        - Recall: 0.75
        - F1-Score: 0.77
        - Support: 106,442
    - Overall:
        - Accuracy: 0.95
        - Macro Avg - Precision: 0.88, Recall: 0.86, F1-Score: 0.87
        - Weighted Avg - Precision: 0.95, Recall: 0.95, F1-Score: 0.95
- **ROC-AUC Score:** 0.9751

---

#### Validation Dataset

- **Loss:** 0.2284
- **Classification Report:**
    - Class 0:
        - Precision: 0.98
      - Recall: 0.98
        - F1-Score: 0.98
        - Support: 91,671
    - Class 1:
        - Precision: 0.65
        - Recall: 0.75
        - F1-Score: 0.70
        - Support: 5,649
    - Overall:
        - Accuracy: 0.96
        - Macro Avg - Precision: 0.82, Recall: 0.86, F1-Score: 0.84
        - Weighted Avg - Precision: 0.97, Recall: 0.96, F1-Score: 0.96
- **ROC-AUC Score:** 0.9726

---

#### Testing Dataset

- **Loss:** 0.2317
- **Classification Report:**
    - Class 0:
        - Precision: 0.98
      - Recall: 0.98
        - F1-Score: 0.98
        - Support: 91,532
    - Class 1:
        - Precision: 0.67
        - Recall: 0.75
        - F1-Score: 0.70
        - Support: 5,788
    - Overall:
        - Accuracy: 0.96
        - Macro Avg - Precision: 0.82, Recall: 0.86, F1-Score: 0.84
        - Weighted Avg - Precision: 0.97, Recall: 0.96, F1-Score: 0.96
- **ROC-AUC Score:** 0.9731

---

## Conclusion

Fine-tuning with larger batch sizes yields similar results but with a slight increase in speed (17% faster).
Batch size 1024, however, is the maximum batch size that fits into the GPU memory.
