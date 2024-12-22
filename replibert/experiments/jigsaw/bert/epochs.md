### Config(epochs=5, optim=Adam, lr=1e-4, weight_decay=0, max_pos_embeds=512, batch_size=128)

```markdown
### Evaluation Results

#### Training Dataset
- **Loss:** 0.0034
- **Classification Report:**
  - Class 0:
    - Precision: 1.00
    - Recall: 1.00
    - F1-Score: 1.00
    - Support: 110,115
  - Class 1:
    - Precision: 0.99
    - Recall: 1.00
    - F1-Score: 0.99
    - Support: 12,237
  - Overall:
    - Accuracy: 1.00
    - Macro Avg - Precision: 1.00, Recall: 1.00, F1-Score: 1.00
    - Weighted Avg - Precision: 1.00, Recall: 1.00, F1-Score: 1.00
- **ROC-AUC Score:** 1.0000

#### Validation Dataset
- **Loss:** 0.1916
- **Classification Report:**
  - Class 0:
    - Precision: 0.98
    - Recall: 0.97
    - F1-Score: 0.98
    - Support: 28,861
  - Class 1:
    - Precision: 0.77
    - Recall: 0.84
    - F1-Score: 0.80
    - Support: 3,059
  - Overall:
    - Accuracy: 0.96
    - Macro Avg - Precision: 0.88, Recall: 0.91, F1-Score: 0.89
    - Weighted Avg - Precision: 0.96, Recall: 0.96, F1-Score: 0.96
- **ROC-AUC Score:** 0.9807

#### Testing Dataset
- **Loss:** 0.5214
- **Classification Report:**
  - Class 0:
    - Precision: 0.99
    - Recall: 0.90
    - F1-Score: 0.94
    - Support: 57,894
  - Class 1:
    - Precision: 0.49
    - Recall: 0.91
    - F1-Score: 0.64
    - Support: 6,090
  - Overall:
    - Accuracy: 0.90
    - Macro Avg - Precision: 0.74, Recall: 0.91, F1-Score: 0.79
    - Weighted Avg - Precision: 0.94, Recall: 0.90, F1-Score: 0.91
- **ROC-AUC Score:** 0.9671

```

### Config(epochs=5, optim=Adam, lr=1e-6, weight_decay=0, max_pos_embeds=512, batch_size=128, preprocessing=False)
### Evaluation Results

#### Training Dataset
- **Loss:** 0.0886
- **Classification Report:**
  - Class 0:
    - Precision: 0.98
    - Recall: 0.98
    - F1-Score: 0.98
    - Support: 110,115
  - Class 1:
    - Precision: 0.84
    - Recall: 0.82
    - F1-Score: 0.83
    - Support: 12,237
  - Overall:
    - Accuracy: 0.97
    - Macro Avg - Precision: 0.91, Recall: 0.90, F1-Score: 0.90
    - Weighted Avg - Precision: 0.97, Recall: 0.97, F1-Score: 0.97
- **ROC-AUC Score:** 0.9851

#### Validation Dataset
- **Loss:** 0.0986
- **Classification Report:**
  - Class 0:
    - Precision: 0.98
    - Recall: 0.98
    - F1-Score: 0.98
    - Support: 28,861
  - Class 1:
    - Precision: 0.82
    - Recall: 0.79
    - F1-Score: 0.80
    - Support: 3,059
  - Overall:
    - Accuracy: 0.96
    - Macro Avg - Precision: 0.90, Recall: 0.88, F1-Score: 0.89
    - Weighted Avg - Precision: 0.96, Recall: 0.96, F1-Score: 0.96
- **ROC-AUC Score:** 0.9805

#### Testing Dataset
- **Loss:** 0.1791
- **Classification Report:**
  - Class 0:
    - Precision: 0.98
    - Recall: 0.93
    - F1-Score: 0.95
    - Support: 57,894
  - Class 1:
    - Precision: 0.55
    - Recall: 0.84
    - F1-Score: 0.66
    - Support: 6,090
  - Overall:
    - Accuracy: 0.92
    - Macro Avg - Precision: 0.76, Recall: 0.88, F1-Score: 0.81
    - Weighted Avg - Precision: 0.94, Recall: 0.92, F1-Score: 0.93
- **ROC-AUC Score:** 0.9647

### Config(epochs=22, optim=Adam, lr=1e-6, weight_decay=0, max_pos_embeds=512, batch_size=128)

Best results of training for 100 epochs where found after epoch 22:

### Evaluation Results

#### Training Dataset
- **Loss:** 0.0519
- **Classification Report:**
  - Class 0:
    - Precision: 0.99
    - Recall: 0.99
    - F1-Score: 0.99
    - Support: 110,115
  - Class 1:
    - Precision: 0.90
    - Recall: 0.91
    - F1-Score: 0.91
    - Support: 12,237
  - Overall:
    - Accuracy: 0.98
    - Macro Avg - Precision: 0.94, Recall: 0.95, F1-Score: 0.95
    - Weighted Avg - Precision: 0.98, Recall: 0.98, F1-Score: 0.98
- **ROC-AUC Score:** 0.9947

#### Validation Dataset
- **Loss:** 0.0965
- **Classification Report:**
  - Class 0:
    - Precision: 0.98
    - Recall: 0.98
    - F1-Score: 0.98
    - Support: 28,861
  - Class 1:
    - Precision: 0.80
    - Recall: 0.84
    - F1-Score: 0.82
    - Support: 3,059
  - Overall:
    - Accuracy: 0.96
    - Macro Avg - Precision: 0.89, Recall: 0.91, F1-Score: 0.90
    - Weighted Avg - Precision: 0.97, Recall: 0.96, F1-Score: 0.96
- **ROC-AUC Score:** 0.9853

#### Testing Dataset
- **Loss:** 0.2500
- **Classification Report:**
  - Class 0:
    - Precision: 0.99
    - Recall: 0.90
    - F1-Score: 0.95
    - Support: 57,894
  - Class 1:
    - Precision: 0.50
    - Recall: 0.92
    - F1-Score: 0.65
    - Support: 6,090
  - Overall:
    - Accuracy: 0.91
    - Macro Avg - Precision: 0.75, Recall: 0.91, F1-Score: 0.80
    - Weighted Avg - Precision: 0.94, Recall: 0.91, F1-Score: 0.92
- **ROC-AUC Score:** 0.9711
