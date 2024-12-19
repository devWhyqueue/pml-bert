### Config(epochs=1, optim=Adam, lr=1e-4, weight_decay=0, max_pos_embeds=512, batch_size=128, preprocessing=True)

```markdown
### Evaluation Results

#### Training Dataset
- **Loss:** 0.0691
- **Classification Report:**
  - Class 0:
    - Precision: 0.99
    - Recall: 0.99
    - F1-Score: 0.99
    - Support: 110,115
  - Class 1:
    - Precision: 0.87
    - Recall: 0.87
    - F1-Score: 0.87
    - Support: 12,237
  - Overall:
    - Accuracy: 0.97
    - Macro Avg - Precision: 0.93, Recall: 0.93, F1-Score: 0.93
    - Weighted Avg - Precision: 0.97, Recall: 0.97, F1-Score: 0.97
- **ROC-AUC Score:** 0.9909

#### Validation Dataset
- **Loss:** 0.1029
- **Classification Report:**
  - Class 0:
    - Precision: 0.98
    - Recall: 0.98
    - F1-Score: 0.98
    - Support: 28,861
  - Class 1:
    - Precision: 0.79
    - Recall: 0.80
    - F1-Score: 0.79
    - Support: 3,059
  - Overall:
    - Accuracy: 0.96
    - Macro Avg - Precision: 0.88, Recall: 0.89, F1-Score: 0.89
    - Weighted Avg - Precision: 0.96, Recall: 0.96, F1-Score: 0.96
- **ROC-AUC Score:** 0.9806

#### Testing Dataset
- **Loss:** 0.2473
- **Classification Report:**
  - Class 0:
    - Precision: 0.99
    - Recall: 0.91
    - F1-Score: 0.95
    - Support: 57,894
  - Class 1:
    - Precision: 0.51
    - Recall: 0.92
    - F1-Score: 0.66
    - Support: 6,090
  - Overall:
    - Accuracy: 0.91
    - Macro Avg - Precision: 0.75, Recall: 0.91, F1-Score: 0.80
    - Weighted Avg - Precision: 0.94, Recall: 0.91, F1-Score: 0.92
- **ROC-AUC Score:** 0.9700

```