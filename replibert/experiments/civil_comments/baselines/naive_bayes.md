# Naive Bayes Results Summary

## Configuration 1: preprocessing=True, pos_proportion=None

- **Train Set**:
    - ROC AUC: 0.895
    - Accuracy: 0.94
    - Precision/Recall/F1/Support:
        - Class 0: 0.94 / 1.00 / 0.97 / 957,942
        - Class 1: 0.97 / 0.06 / 0.11 / 106,438
    - Macro avg: 0.96 / 0.53 / 0.54
    - Weighted avg: 0.95 / 0.94 / 0.92
- **Validation Set**:
    - ROC AUC: 0.896
    - Accuracy: 0.95
    - Precision/Recall/F1/Support:
        - Class 0: 0.95 / 1.00 / 0.97 / 91,671
        - Class 1: 0.98 / 0.06 / 0.11 / 5,649
    - Macro avg: 0.96 / 0.53 / 0.54
    - Weighted avg: 0.95 / 0.95 / 0.92
- **Test Set**:
    - ROC AUC: 0.894
    - Accuracy: 0.94
    - Precision/Recall/F1/Support:
        - Class 0: 0.94 / 1.00 / 0.97 / 91,532
        - Class 1: 0.97 / 0.05 / 0.10 / 5,788
    - Macro avg: 0.96 / 0.53 / 0.54
    - Weighted avg: 0.95 / 0.94 / 0.92

---

## Configuration 2: preprocessing=True, pos_proportion=0.1

- **Train Set**:
    - ROC AUC: 0.895
    - Accuracy: 0.91
    - Precision/Recall/F1/Support:
        - Class 0: 0.94 / 1.00 / 0.97 / 957,942
        - Class 1: 0.96 / 0.12 / 0.21 / 106,438
    - Macro avg: 0.93 / 0.56 / 0.58
    - Weighted avg: 0.92 / 0.91 / 0.88
- **Validation Set**:
    - ROC AUC: 0.896
    - Accuracy: 0.95
    - Precision/Recall/F1/Support:
        - Class 0: 0.95 / 1.00 / 0.97 / 91,671
        - Class 1: 0.93 / 0.12 / 0.21 / 5,649
    - Macro avg: 0.94 / 0.56 / 0.59
    - Weighted avg: 0.95 / 0.95 / 0.93
- **Test Set**:
    - ROC AUC: 0.893
    - Accuracy: 0.95
    - Precision/Recall/F1/Support:
        - Class 0: 0.95 / 1.00 / 0.97 / 91,532
        - Class 1: 0.91 / 0.12 / 0.21 / 5,788
    - Macro avg: 0.93 / 0.56 / 0.59
    - Weighted avg: 0.95 / 0.95 / 0.93

---

## Configuration 3: preprocessing=True, pos_proportion=0.25

- **Train Set**:
    - ROC AUC: 0.897
    - Accuracy: 0.84
    - Precision/Recall/F1/Support:
        - Class 0: 0.84 / 0.98 / 0.90 / 319,314
        - Class 1: 0.90 / 0.42 / 0.58 / 106,438
    - Macro avg: 0.87 / 0.70 / 0.74
    - Weighted avg: 0.85 / 0.84 / 0.82
- **Validation Set**:
    - ROC AUC: 0.898
    - Accuracy: 0.95
    - Precision/Recall/F1/Support:
        - Class 0: 0.97 / 0.99 / 0.98 / 91,532
        - Class 1: 0.64 / 0.43 / 0.51 / 5,788
    - Macro avg: 0.80 / 0.71 / 0.74
    - Weighted avg: 0.95 / 0.95 / 0.95
- **Test Set**:
    - ROC AUC: 0.894
    - Accuracy: 0.95
    - Precision/Recall/F1/Support:
        - Class 0: 0.96 / 0.99 / 0.97 / 91,532
        - Class 1: 0.64 / 0.42 / 0.51 / 5,788
    - Macro avg: 0.80 / 0.70 / 0.74
    - Weighted avg: 0.94 / 0.95 / 0.95

---

## Configuration 4: preprocessing=True, pos_proportion=0.5

- **Train Set**:
    - ROC AUC: 0.896
    - Accuracy: 0.81
    - Precision/Recall/F1/Support:
        - Class 0: 0.83 / 0.78 / 0.81 / 106,438
        - Class 1: 0.80 / 0.84 / 0.82 / 106,438
    - Macro avg: 0.81 / 0.81 / 0.81
    - Weighted avg: 0.81 / 0.81 / 0.81
- **Validation Set**:
    - ROC AUC: 0.895
    - Accuracy: 0.79
    - Precision/Recall/F1/Support:
        - Class 0: 0.99 / 0.78 / 0.87 / 91,532
        - Class 1: 0.19 / 0.84 / 0.31 / 5,788
    - Macro avg: 0.59 / 0.81 / 0.59
    - Weighted avg: 0.94 / 0.79 / 0.84
- **Test Set**:
    - ROC AUC: 0.891
    - Accuracy: 0.78
    - Precision/Recall/F1/Support:
        - Class 0: 0.99 / 0.78 / 0.87 / 91,532
        - Class 1: 0.19 / 0.84 / 0.32 / 5,788
    - Macro avg: 0.59 / 0.81 / 0.59
    - Weighted avg: 0.94 / 0.78 / 0.84

---

## Configuration 5: preprocessing=False, pos_proportion=None

- **Train Set**:
    - ROC AUC: 0.895
    - Accuracy: 0.94
    - Precision/Recall/F1/Support:
        - Class 0: 0.94 / 1.00 / 0.97 / 1,698,436
        - Class 1: 0.97 / 0.06 / 0.11 / 106,438
    - Macro avg: 0.96 / 0.53 / 0.54
    - Weighted avg: 0.95 / 0.94 / 0.92
- **Validation Set**:
    - ROC AUC: 0.895
    - Accuracy: 0.95
    - Precision/Recall/F1/Support:
        - Class 0: 0.95 / 1.00 / 0.97 / 91,671
        - Class 1: 0.98 / 0.06 / 0.11 / 5,649
    - Macro avg: 0.96 / 0.53 / 0.54
    - Weighted avg: 0.95 / 0.95 / 0.92
- **Test Set**:
    - ROC AUC: 0.892
    - Accuracy: 0.94
    - Precision/Recall/F1/Support:
        - Class 0: 0.94 / 1.00 / 0.97 / 91,532
        - Class 1: 0.97 / 0.05 / 0.10 / 5,788
    - Macro avg: 0.96 / 0.53 / 0.53
    - Weighted avg: 0.95 / 0.94 / 0.92

---

## Configuration 6: preprocessing=False, pos_proportion=0.1

- **Train Set**:
    - ROC AUC: 0.896
    - Accuracy: 0.91
    - Precision/Recall/F1/Support:
        - Class 0: 0.91 / 1.00 / 0.95 / 957,942
        - Class 1: 0.96 / 0.12 / 0.22 / 106,438
    - Macro avg: 0.93 / 0.56 / 0.58
    - Weighted avg: 0.92 / 0.91 / 0.88
- **Validation Set**:
    - ROC AUC: 0.895
    - Accuracy: 0.95
    - Precision/Recall/F1/Support:
        - Class 0: 0.95 / 1.00 / 0.97 / 91,671
        - Class 1: 0.93 / 0.12 / 0.21 / 5,649
    - Macro avg: 0.94 / 0.56 / 0.59
    - Weighted avg: 0.95 / 0.95 / 0.93
- **Test Set**:
    - ROC AUC: 0.893
    - Accuracy: 0.95
    - Precision/Recall/F1/Support:
        - Class 0: 0.95 / 1.00 / 0.97 / 91,532
        - Class 1: 0.91 / 0.12 / 0.21 / 5,788
    - Macro avg: 0.93 / 0.56 / 0.59
    - Weighted avg: 0.95 / 0.95 / 0.93

---

## Configuration 7: preprocessing=False, pos_proportion=0.25

- **Train Set**:
    - ROC AUC: 0.899
    - Accuracy: 0.85
    - Precision/Recall/F1/Support:
        - Class 0: 0.84 / 0.99 / 0.91 / 319,314
        - Class 1: 0.90 / 0.43 / 0.58 / 106,438
    - Macro avg: 0.87 / 0.71 / 0.74
    - Weighted avg: 0.85 / 0.85 / 0.82
- **Validation Set**:
    - ROC AUC: 0.898
    - Accuracy: 0.95
    - Precision/Recall/F1/Support:
        - Class 0: 0.97 / 0.99 / 0.98 / 91,532
        - Class 1: 0.64 / 0.43 / 0.51 / 5,788
    - Macro avg: 0.80 / 0.71 / 0.74
    - Weighted avg: 0.95 / 0.95 / 0.96
- **Test Set**:
    - ROC AUC: 0.894
    - Accuracy: 0.95
    - Precision/Recall/F1/Support:
        - Class 0: 0.96 / 0.99 / 0.97 / 91,532
        - Class 1: 0.64 / 0.42 / 0.51 / 5,788
    - Macro avg: 0.80 / 0.70 / 0.74
    - Weighted avg: 0.95 / 0.95 / 0.95

---

## Configuration 8: preprocessing=False, pos_proportion=0.5

- **Train Set**:
    - ROC AUC: 0.897
    - Accuracy: 0.81
    - Precision/Recall/F1/Support:
        - Class 0: 0.83 / 0.78 / 0.81 / 106,438
        - Class 1: 0.80 / 0.84 / 0.82 / 106,438
    - Macro avg: 0.82 / 0.81 / 0.81
    - Weighted avg: 0.82 / 0.81 / 0.81
- **Validation Set**:
    - ROC AUC: 0.894
    - Accuracy: 0.79
    - Precision/Recall/F1/Support:
        - Class 0: 0.99 / 0.79 / 0.88 / 91,532
        - Class 1: 0.19 / 0.84 / 0.32 / 5,788
    - Macro avg: 0.59 / 0.81 / 0.60
    - Weighted avg: 0.94 / 0.79 / 0.84
- **Test Set**:
    - ROC AUC: 0.891
    - Accuracy: 0.79
    - Precision/Recall/F1/Support:
        - Class 0: 0.99 / 0.78 / 0.87 / 91,532
        - Class 1: 0.20 / 0.84 / 0.32 / 5,788
    - Macro avg: 0.59 / 0.81 / 0.60
    - Weighted avg: 0.94 / 0.79 / 0.84
