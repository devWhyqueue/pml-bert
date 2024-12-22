# Learning rates experiments

Other hyperparameters are fixed as follows:

- `batch_size=128`
- `prepocessing=False`
- `pos_proportion=0.1`
- `sequence_len=512`
- `weight_decay=0.0`
- `num_epochs=1`
- `optimizer=AdamW`
- `loss=BCEWithLogitsLoss`

## Config(lr=1e-3)


#### Training Dataset
- **Loss:** 0.3259
- **Classification Report:**
  - Class 0:
    - Precision: 0.90
    - Recall: 1.00
    - F1-Score: 0.95
    - Support: 110,115
  - Class 1:
    - Precision: 0.00
    - Recall: 0.00
    - F1-Score: 0.00
    - Support: 12,237
  - Overall:
    - Accuracy: 0.90
    - Macro Avg - Precision: 0.45, Recall: 0.50, F1-Score: 0.47
    - Weighted Avg - Precision: 0.81, Recall: 0.90, F1-Score: 0.85
- **ROC-AUC Score:** 0.5719

#### Validation Dataset
- **Loss:** 0.3172
- **Classification Report:**
  - Class 0:
    - Precision: 0.90
    - Recall: 1.00
    - F1-Score: 0.95
    - Support: 28,861
  - Class 1:
    - Precision: 0.00
    - Recall: 0.00
    - F1-Score: 0.00
    - Support: 3,059
  - Overall:
    - Accuracy: 0.90
    - Macro Avg - Precision: 0.45, Recall: 0.50, F1-Score: 0.47
    - Weighted Avg - Precision: 0.82, Recall: 0.90, F1-Score: 0.86
- **ROC-AUC Score:** 0.5677

#### Testing Dataset
- **Loss:** 0.3159
- **Classification Report:**
  - Class 0:
    - Precision: 0.90
    - Recall: 1.00
    - F1-Score: 0.95
    - Support: 57,894
  - Class 1:
    - Precision: 0.00
    - Recall: 0.00
    - F1-Score: 0.00
    - Support: 6,090
  - Overall:
    - Accuracy: 0.90
    - Macro Avg - Precision: 0.45, Recall: 0.50, F1-Score: 0.48
    - Weighted Avg - Precision: 0.82, Recall: 0.90, F1-Score: 0.86
- **ROC-AUC Score:** 0.6286

## Config(lr=1e-4)


### Training Dataset

- **Loss:** 0.0535
- **Classification Report:**
    - Class 0:
        - Precision: 0.99
        - Recall: 0.99
        - F1-Score: 0.99
        - Support: 110,115
    - Class 1:
        - Precision: 0.90
        - Recall: 0.92
        - F1-Score: 0.91
        - Support: 12,237
    - Overall:
        - Accuracy: 0.98
        - Macro Avg - Precision: 0.94, Recall: 0.96, F1-Score: 0.95
        - Weighted Avg - Precision: 0.98, Recall: 0.98, F1-Score: 0.98
- **ROC-AUC Score:** 0.9958

### Validation Dataset

- **Loss:** 0.0887
- **Classification Report:**
    - Class 0:
        - Precision: 0.98
        - Recall: 0.98
        - F1-Score: 0.98
        - Support: 28,861
    - Class 1:
        - Precision: 0.80
        - Recall: 0.85
        - F1-Score: 0.82
        - Support: 3,059
    - Overall:
        - Accuracy: 0.96
        - Macro Avg - Precision: 0.89, Recall: 0.91, F1-Score: 0.90
        - Weighted Avg - Precision: 0.97, Recall: 0.96, F1-Score: 0.97
- **ROC-AUC Score:** 0.9870

### Testing Dataset

- **Loss:** 0.2543
- **Classification Report:**
    - Class 0:
        - Precision: 0.99
        - Recall: 0.90
        - F1-Score: 0.94
        - Support: 57,894
    - Class 1:
        - Precision: 0.50
        - Recall: 0.94
        - F1-Score: 0.65
        - Support: 6,090
    - Overall:
        - Accuracy: 0.90
        - Macro Avg - Precision: 0.74, Recall: 0.92, F1-Score: 0.80
        - Weighted Avg - Precision: 0.95, Recall: 0.90, F1-Score: 0.92
- **ROC-AUC Score:** 0.9737
```

## Config(lr=1e-5)


#### Training Dataset
- **Loss:** 0.0786
- **Classification Report:**
  - Class 0:
    - Precision: 0.98
    - Recall: 0.99
    - F1-Score: 0.98
    - Support: 110,115
  - Class 1:
    - Precision: 0.86
    - Recall: 0.82
    - F1-Score: 0.84
    - Support: 12,237
  - Overall:
    - Accuracy: 0.97
    - Macro Avg - Precision: 0.92, Recall: 0.90, F1-Score: 0.91
    - Weighted Avg - Precision: 0.97, Recall: 0.97, F1-Score: 0.97
- **ROC-AUC Score:** 0.9879

#### Validation Dataset
- **Loss:** 0.0909
- **Classification Report:**
  - Class 0:
    - Precision: 0.98
    - Recall: 0.98
    - F1-Score: 0.98
    - Support: 28,861
  - Class 1:
    - Precision: 0.84
    - Recall: 0.78
    - F1-Score: 0.81
    - Support: 3,059
  - Overall:
    - Accuracy: 0.97
    - Macro Avg - Precision: 0.91, Recall: 0.88, F1-Score: 0.90
    - Weighted Avg - Precision: 0.96, Recall: 0.97, F1-Score: 0.96
- **ROC-AUC Score:** 0.9830

#### Testing Dataset
- **Loss:** 0.1696
- **Classification Report:**
  - Class 0:
    - Precision: 0.99
    - Recall: 0.93
    - F1-Score: 0.96
    - Support: 57,894
  - Class 1:
    - Precision: 0.56
    - Recall: 0.87
    - F1-Score: 0.68
    - Support: 6,090
  - Overall:
    - Accuracy: 0.92
    - Macro Avg - Precision: 0.77, Recall: 0.90, F1-Score: 0.82
    - Weighted Avg - Precision: 0.95, Recall: 0.92, F1-Score: 0.93
- **ROC-AUC Score:** 0.9697

```

## Config(lr=1e-6)


#### Training Dataset
- **Loss:** 0.2754
- **Classification Report:**
  - Class 0:
    - Precision: 0.90
    - Recall: 1.00
    - F1-Score: 0.95
    - Support: 110,115
  - Class 1:
    - Precision: 1.00
    - Recall: 0.00
    - F1-Score: 0.00
    - Support: 12,237
  - Overall:
    - Accuracy: 0.90
    - Macro Avg - Precision: 0.95, Recall: 0.50, F1-Score: 0.47
    - Weighted Avg - Precision: 0.91, Recall: 0.90, F1-Score: 0.85
- **ROC-AUC Score:** 0.8154

#### Validation Dataset
- **Loss:** 0.2687
- **Classification Report:**
  - Class 0:
    - Precision: 0.90
    - Recall: 1.00
    - F1-Score: 0.95
    - Support: 28,861
  - Class 1:
    - Precision: 0.00
    - Recall: 0.00
    - F1-Score: 0.00
    - Support: 3,059
  - Overall:
    - Accuracy: 0.90
    - Macro Avg - Precision: 0.45, Recall: 0.50, F1-Score: 0.47
    - Weighted Avg - Precision: 0.82, Recall: 0.90, F1-Score: 0.86
- **ROC-AUC Score:** 0.8125

#### Testing Dataset
- **Loss:** 0.2676
- **Classification Report:**
  - Class 0:
    - Precision: 0.90
    - Recall: 1.00
    - F1-Score: 0.95
    - Support: 57,894
  - Class 1:
    - Precision: 1.00
    - Recall: 0.00
    - F1-Score: 0.00
    - Support: 6,090
  - Overall:
    - Accuracy: 0.90
    - Macro Avg - Precision: 0.95, Recall: 0.50, F1-Score: 0.48
    - Weighted Avg - Precision: 0.91, Recall: 0.90, F1-Score: 0.86

```
