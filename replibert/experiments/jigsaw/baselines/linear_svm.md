# LinearSVC Results Summary
# Best Config: 3

## Configuration 1: preprocessing=True, pos_proportion=None

- **Train Set**:
    - ROC AUC: 0.9803832094840448
    - Accuracy: 0.96
    - Precision/Recall/F1/Support:
        - Class 0: 0.97 / 0.99 / 0.98 / 115422
        - Class 1: 0.91 / 0.69 / 0.79 / 12235
    - Macro avg: 0.94 / 0.84 / 0.88
    - Weighted avg: 0.96 / 0.96 / 0.96

- **Validation Set**:
    - ROC AUC: 0.9574703957954147
    - Accuracy: 0.96
    - Precision/Recall/F1/Support:
        - Class 0: 0.96 / 0.99 / 0.98 / 28855
        - Class 1: 0.88 / 0.64 / 0.74 / 3059
    - Macro avg: 0.92 / 0.81 / 0.86
    - Weighted avg: 0.95 / 0.96 / 0.95

- **Test Set**:
    - ROC AUC: 0.9492159027885567
    - Accuracy: 0.93
    - Precision/Recall/F1/Support:
        - Class 0: 0.97 / 0.95 / 0.96 / 57888
        - Class 1: 0.60 / 0.76 / 0.67 / 6090
    - Macro avg: 0.79 / 0.86 / 0.82
    - Weighted avg: 0.94 / 0.93 / 0.93


---

## Configuration 2: preprocessing=True, pos_proportion=0.1

Train Set:

    ROC AUC: 0.981
    Accuracy: 0.96
    Precision/Recall/F1/Support:
        Class 0: 0.97 / 0.99 / 0.98 / 110,115
        Class 1: 0.92 / 0.71 / 0.80 / 12,235
    Macro avg: 0.94 / 0.85 / 0.89
    Weighted avg: 0.96 / 0.96 / 0.96

Validation Set:

    ROC AUC: 0.955
    Accuracy: 0.96
    Precision/Recall/F1/Support:
        Class 0: 0.96 / 0.99 / 0.98 / 28,855
        Class 1: 0.86 / 0.65 / 0.74 / 3,059
    Macro avg: 0.91 / 0.82 / 0.86
    Weighted avg: 0.95 / 0.96 / 0.95

Test Set:

    ROC AUC: 0.948
    Accuracy: 0.93
    Precision/Recall/F1/Support:
        Class 0: 0.98 / 0.94 / 0.96 / 57,888
        Class 1: 0.58 / 0.78 / 0.67 / 6,090
    Macro avg: 0.78 / 0.86 / 0.81
    Weighted avg: 0.94 / 0.93 / 0.93

---

## Configuration 3: preprocessing=True, pos_proportion=0.25

    Train Set:
        ROC AUC: 0.985
        Accuracy: 0.95
        Precision/Recall/F1/Support:
            Class 0: 0.95 / 0.98 / 0.96 / 36,705
            Class 1: 0.94 / 0.83 / 0.88 / 12,235
        Macro avg: 0.94 / 0.91 / 0.92
        Weighted avg: 0.95 / 0.95 / 0.94
    Validation Set:
        ROC AUC: 0.959
        Accuracy: 0.95
        Precision/Recall/F1/Support:
            Class 0: 0.97 / 0.97 / 0.97 / 28,855
            Class 1: 0.73 / 0.76 / 0.75 / 3,059
        Macro avg: 0.85 / 0.87 / 0.86
        Weighted avg: 0.95 / 0.95 / 0.95
    Test Set:
        ROC AUC: 0.950
        Accuracy: 0.90
        Precision/Recall/F1/Support:
            Class 0: 0.98 / 0.90 / 0.94 / 57,888
            Class 1: 0.48 / 0.86 / 0.62 / 6,090
        Macro avg: 0.73 / 0.88 / 0.78
        Weighted avg: 0.94 / 0.90 / 0.91



---

## Configuration 4: preprocessing=True, pos_proportion=0.5

Train Set:

    ROC AUC: 0.989
    Accuracy: 0.95
    Precision/Recall/F1/Support:
        Class 0: 0.93 / 0.96 / 0.95 / 12,235
        Class 1: 0.96 / 0.93 / 0.95 / 12,235
    Macro avg: 0.95 / 0.95 / 0.95
    Weighted avg: 0.95 / 0.95 / 0.95

Validation Set:

    ROC AUC: 0.959
    Accuracy: 0.91
    Precision/Recall/F1/Support:
        Class 0: 0.99 / 0.92 / 0.95 / 28,855
        Class 1: 0.52 / 0.87 / 0.65 / 3,059
    Macro avg: 0.75 / 0.89 / 0.80
    Weighted avg: 0.94 / 0.91 / 0.92

Test Set:

    ROC AUC: 0.948
    Accuracy: 0.85
    Precision/Recall/F1/Support:
        Class 0: 0.99 / 0.84 / 0.91 / 57,888
        Class 1: 0.38 / 0.92 / 0.54 / 6,090
    Macro avg: 0.68 / 0.88 / 0.72
    Weighted avg: 0.93 / 0.85 / 0.87
---

## Configuration 5: preprocessing=False, pos_proportion=None

Train Set:

    ROC AUC: 0.980
    Accuracy: 0.96
    Precision/Recall/F1/Support:
        Class 0: 0.97 / 0.99 / 0.98 / 115,422
        Class 1: 0.91 / 0.69 / 0.79 / 12,235
    Macro avg: 0.94 / 0.84 / 0.88
    Weighted avg: 0.96 / 0.96 / 0.96

Validation Set:

    ROC AUC: 0.957
    Accuracy: 0.96
    Precision/Recall/F1/Support:
        Class 0: 0.96 / 0.99 / 0.98 / 28,855
        Class 1: 0.88 / 0.64 / 0.74 / 3,059
    Macro avg: 0.92 / 0.81 / 0.86
    Weighted avg: 0.95 / 0.96 / 0.95

Test Set:

    ROC AUC: 0.949
    Accuracy: 0.93
    Precision/Recall/F1/Support:
        Class 0: 0.97 / 0.95 / 0.96 / 57,888
        Class 1: 0.60 / 0.76 / 0.67 / 6,090
    Macro avg: 0.79 / 0.86 / 0.82
    Weighted avg: 0.94 / 0.93 / 0.93
---

## Configuration 6: preprocessing=False, pos_proportion=0.1

Train Set:

    ROC AUC: 0.981
    Accuracy: 0.96
    Precision/Recall/F1/Support:
        Class 0: 0.97 / 0.99 / 0.98 / 110,115
        Class 1: 0.92 / 0.71 / 0.80 / 12,235
    Macro avg: 0.94 / 0.85 / 0.89
    Weighted avg: 0.96 / 0.96 / 0.96

Validation Set:

    ROC AUC: 0.955
    Accuracy: 0.96
    Precision/Recall/F1/Support:
        Class 0: 0.96 / 0.99 / 0.98 / 28,855
        Class 1: 0.86 / 0.65 / 0.74 / 3,059
    Macro avg: 0.91 / 0.82 / 0.86
    Weighted avg: 0.95 / 0.96 / 0.95

Test Set:

    ROC AUC: 0.948
    Accuracy: 0.93
    Precision/Recall/F1/Support:
        Class 0: 0.98 / 0.94 / 0.96 / 57,888
        Class 1: 0.58 / 0.78 / 0.67 / 6,090
    Macro avg: 0.78 / 0.86 / 0.81
    Weighted avg: 0.94 / 0.93 / 0.93
---

## Configuration 7: preprocessing=False, pos_proportion=0.25

Train Set:

    ROC AUC: 0.985
    Accuracy: 0.95
    Precision/Recall/F1/Support:
        Class 0: 0.95 / 0.98 / 0.96 / 36,705
        Class 1: 0.94 / 0.83 / 0.88 / 12,235
    Macro avg: 0.94 / 0.91 / 0.92
    Weighted avg: 0.95 / 0.95 / 0.94

Validation Set:

    ROC AUC: 0.959
    Accuracy: 0.95
    Precision/Recall/F1/Support:
        Class 0: 0.97 / 0.97 / 0.97 / 28,855
        Class 1: 0.73 / 0.76 / 0.75 / 3,059
    Macro avg: 0.85 / 0.87 / 0.86
    Weighted avg: 0.95 / 0.95 / 0.95

Test Set:

    ROC AUC: 0.950
    Accuracy: 0.90
    Precision/Recall/F1/Support:
        Class 0: 0.98 / 0.90 / 0.94 / 57,888
        Class 1: 0.48 / 0.86 / 0.62 / 6,090
    Macro avg: 0.73 / 0.88 / 0.78
    Weighted avg: 0.94 / 0.90 / 0.91
---

## Configuration 8: preprocessing=False, pos_proportion=0.5

Train Set:

    ROC AUC: 0.989
    Accuracy: 0.95
    Precision/Recall/F1/Support:
        Class 0: 0.93 / 0.96 / 0.95 / 12,235
        Class 1: 0.96 / 0.93 / 0.95 / 12,235
    Macro avg: 0.95 / 0.95 / 0.95
    Weighted avg: 0.95 / 0.95 / 0.95

Validation Set:

    ROC AUC: 0.959
    Accuracy: 0.91
    Precision/Recall/F1/Support:
        Class 0: 0.99 / 0.92 / 0.95 / 28,855
        Class 1: 0.52 / 0.87 / 0.65 / 3,059
    Macro avg: 0.75 / 0.89 / 0.80
    Weighted avg: 0.94 / 0.91 / 0.92

Test Set:

    ROC AUC: 0.948
    Accuracy: 0.85
    Precision/Recall/F1/Support:
        Class 0: 0.99 / 0.84 / 0.91 / 57,888
        Class 1: 0.38 / 0.92 / 0.54 / 6,090
    Macro avg: 0.68 / 0.88 / 0.72
    Weighted avg: 0.93 / 0.85 / 0.87