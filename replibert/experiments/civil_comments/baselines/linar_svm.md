# LinearSVC Results Summary

## Configuration 1: preprocessing=True, pos_proportion=None

- **Train Set**:
    - ROC AUC: 0.934
    - Accuracy: 0.96
    - Precision/Recall/F1/Support:
        - Class 0: 0.97 / 0.99 / 0.98 / 1,698,436
        - Class 1: 0.78 / 0.44 / 0.56 / 106,438
    - Macro avg: 0.87 / 0.72 / 0.77
    - Weighted avg: 0.95 / 0.96 / 0.95
- **Validation Set**:
    - ROC AUC: 0.932
    - Accuracy: 0.96
    - Precision/Recall/F1/Support:
        - Class 0: 0.97 / 0.99 / 0.98 / 91,671
        - Class 1: 0.76 / 0.45 / 0.56 / 5,649
    - Macro avg: 0.87 / 0.72 / 0.77
    - Weighted avg: 0.96 / 0.96 / 0.95
- **Test Set**:
    - ROC AUC: 0.931
    - Accuracy: 0.96
    - Precision/Recall/F1/Support:
        - Class 0: 0.97 / 0.99 / 0.98 / 91,532
        - Class 1: 0.76 / 0.44 / 0.55 / 5,788
    - Macro avg: 0.86 / 0.71 / 0.77
    - Weighted avg: 0.95 / 0.96 / 0.95

---

## Configuration 2: preprocessing=True, pos_proportion=0.1

- **Train Set**:
    - ROC AUC: 0.938
    - Accuracy: 0.94
    - Precision/Recall/F1/Support:
        - Class 0: 0.95 / 0.99 / 0.97 / 957,942
        - Class 1: 0.81 / 0.53 / 0.64 / 106,438
    - Macro avg: 0.88 / 0.76 / 0.81
    - Weighted avg: 0.94 / 0.94 / 0.94
- **Validation Set**:
    - ROC AUC: 0.934
    - Accuracy: 0.96
    - Precision/Recall/F1/Support:
        - Class 0: 0.97 / 0.99 / 0.98 / 91,671
        - Class 1: 0.69 / 0.54 / 0.61 / 5,649
    - Macro avg: 0.83 / 0.76 / 0.79
    - Weighted avg: 0.96 / 0.96 / 0.96
- **Test Set**:
    - ROC AUC: 0.934
    - Accuracy: 0.96
    - Precision/Recall/F1/Support:
        - Class 0: 0.97 / 0.99 / 0.98 / 91,532
        - Class 1: 0.69 / 0.53 / 0.60 / 5,788
    - Macro avg: 0.83 / 0.75 / 0.79
    - Weighted avg: 0.95 / 0.96 / 0.96

---

## Configuration 3: preprocessing=True, pos_proportion=0.25

- **Train Set**:
    - ROC AUC: 0.947
    - Accuracy: 0.90
    - Precision/Recall/F1/Support:
        - Class 0: 0.91 / 0.96 / 0.94 / 319,314
        - Class 1: 0.86 / 0.72 / 0.78 / 106,438
    - Macro avg: 0.89 / 0.84 / 0.86
    - Weighted avg: 0.90 / 0.90 / 0.90
- **Validation Set**:
    - ROC AUC: 0.944
    - Accuracy: 0.95
    - Precision/Recall/F1/Support:
        - Class 0: 0.98 / 0.96 / 0.97 / 91,671
        - Class 1: 0.52 / 0.73 / 0.61 / 5,649
    - Macro avg: 0.75 / 0.84 / 0.79
    - Weighted avg: 0.96 / 0.95 / 0.95
- **Test Set**:
    - ROC AUC: 0.942
    - Accuracy: 0.95
    - Precision/Recall/F1/Support:
        - Class 0: 0.98 / 0.96 / 0.97 / 91,532
        - Class 1: 0.53 / 0.72 / 0.61 / 5,788
    - Macro avg: 0.76 / 0.84 / 0.79
    - Weighted avg: 0.96 / 0.95 / 0.95

---

## Configuration 4: preprocessing=True, pos_proportion=0.5

- **Train Set**:
    - ROC AUC: 0.952
    - Accuracy: 0.88
    - Precision/Recall/F1/Support:
        - Class 0: 0.87 / 0.91 / 0.89 / 106,438
        - Class 1: 0.90 / 0.86 / 0.88 / 106,438
    - Macro avg: 0.89 / 0.88 / 0.88
    - Weighted avg: 0.89 / 0.88 / 0.88
- **Validation Set**:
    - ROC AUC: 0.946
    - Accuracy: 0.90
    - Precision/Recall/F1/Support:
        - Class 0: 0.99 / 0.90 / 0.94 / 91,671
        - Class 1: 0.35 / 0.84 / 0.50 / 5,649
    - Macro avg: 0.67 / 0.88 / 0.72
    - Weighted avg: 0.95 / 0.90 / 0.92
- **Test Set**:
    - ROC AUC: 0.944
    - Accuracy: 0.90
    - Precision/Recall/F1/Support:
        - Class 0: 0.99 / 0.90 / 0.94 / 91,532
        - Class 1: 0.36 / 0.85 / 0.50 / 5,788
    - Macro avg: 0.67 / 0.88 / 0.72
    - Weighted avg: 0.95 / 0.90 / 0.92

---

## Configuration 5: preprocessing=False, pos_proportion=None

- **Train Set**:
    - ROC AUC: 0.934
    - Accuracy: 0.96
    - Precision/Recall/F1/Support:
        - Class 0: 0.97 / 0.99 / 0.98 / 1,698,436
        - Class 1: 0.78 / 0.44 / 0.56 / 106,438
    - Macro avg: 0.87 / 0.72 / 0.77
    - Weighted avg: 0.95 / 0.96 / 0.95
- **Validation Set**:
    - ROC AUC: 0.932
    - Accuracy: 0.96
    - Precision/Recall/F1/Support:
        - Class 0: 0.97 / 0.99 / 0.98 / 91,671
        - Class 1: 0.76 / 0.45 / 0.56 / 5,649
    - Macro avg: 0.87 / 0.72 / 0.77
    - Weighted avg: 0.96 / 0.96 / 0.95
- **Test Set**:
    - ROC AUC: 0.931
    - Accuracy: 0.96
    - Precision/Recall/F1/Support:
        - Class 0: 0.97 / 0.99 / 0.98 / 91,532
        - Class 1: 0.76 / 0.44 / 0.55 / 5,788
    - Macro avg: 0.86 / 0.71 / 0.77
    - Weighted avg: 0.95 / 0.96 / 0.95

---

## Configuration 6: preprocessing=False, pos_proportion=0.1

- **Train Set**:
    - ROC AUC: 0.938
    - Accuracy: 0.94
    - Precision/Recall/F1/Support:
        - Class 0: 0.95 / 0.99 / 0.97 / 957,942
        - Class 1: 0.81 / 0.53 / 0.64 / 106,438
    - Macro avg: 0.88 / 0.76 / 0.81
    - Weighted avg: 0.94 / 0.94 / 0.94
- **Validation Set**:
    - ROC AUC: 0.934
    - Accuracy: 0.96
    - Precision/Recall/F1/Support:
        - Class 0: 0.97 / 0.99 / 0.98 / 91,671
        - Class 1: 0.69 / 0.54 / 0.61 / 5,649
    - Macro avg: 0.83 / 0.76 / 0.79
    - Weighted avg: 0.96 / 0.96 / 0.96
- **Test Set**:
    - ROC AUC: 0.934
    - Accuracy: 0.96
    - Precision/Recall/F1/Support:
        - Class 0: 0.97 / 0.99 / 0.98 / 91,532
        - Class 1: 0.69 / 0.53 / 0.60 / 5,788
    - Macro avg: 0.83 / 0.76 / 0.79
    - Weighted avg: 0.95 / 0.96 / 0.96

---

## Configuration 7: preprocessing=False, pos_proportion=0.25

- **Train Set**:
    - ROC AUC: 0.947
    - Accuracy: 0.90
    - Precision/Recall/F1/Support:
        - Class 0: 0.91 / 0.96 / 0.94 / 319,314
        - Class 1: 0.86 / 0.72 / 0.78 / 106,438
    - Macro avg: 0.89 / 0.84 / 0.86
    - Weighted avg: 0.90 / 0.90 / 0.90
- **Validation Set**:
    - ROC AUC: 0.944
    - Accuracy: 0.95
    - Precision/Recall/F1/Support:
        - Class 0: 0.98 / 0.96 / 0.97 / 91,671
        - Class 1: 0.52 / 0.73 / 0.61 / 5,649
    - Macro avg: 0.75 / 0.84 / 0.79
    - Weighted avg: 0.96 / 0.95 / 0.95
- **Test Set**:
    - ROC AUC: 0.942
    - Accuracy: 0.95
    - Precision/Recall/F1/Support:
        - Class 0: 0.98 / 0.96 / 0.97 / 91,532
        - Class 1: 0.53 / 0.72 / 0.61 / 5,788
    - Macro avg: 0.76 / 0.84 / 0.79
    - Weighted avg: 0.96 / 0.95 / 0.95

---

## Configuration 8: preprocessing=False, pos_proportion=0.5

- **Train Set**:
    - ROC AUC: 0.952
    - Accuracy: 0.88
    - Precision/Recall/F1/Support:
        - Class 0: 0.87 / 0.91 / 0.89 / 106,438
        - Class 1: 0.90 / 0.86 / 0.88 / 106,438
    - Macro avg: 0.89 / 0.88 / 0.88
    - Weighted avg: 0.89 / 0.88 / 0.88
- **Validation Set**:
    - ROC AUC: 0.946
    - Accuracy: 0.90
    - Precision/Recall/F1/Support:
        - Class 0: 0.99 / 0.90 / 0.94 / 91,671
        - Class 1: 0.35 / 0.85 / 0.50 / 5,649
    - Macro avg: 0.67 / 0.88 / 0.72
    - Weighted avg: 0.95 / 0.90 / 0.92
- **Test Set**:
    - ROC AUC: 0.944
    - Accuracy: 0.90
    - Precision/Recall/F1/Support:
        - Class 0: 0.99 / 0.90 / 0.94 / 91,532
        - Class 1: 0.36 / 0.86 / 0.51 / 5,788
    - Macro avg: 0.67 / 0.88 / 0.73
    - Weighted avg: 0.95 / 0.90 / 0.92


