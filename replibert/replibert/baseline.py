from copy import deepcopy
from typing import Tuple

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.utils import resample
from torch.utils.data import Subset, random_split

from configuration.config import get_logger
from data.finetuning.datasets import FineTuningDataset
from data.finetuning.transform import tf_idf_vectorize, preprocess

log = get_logger(__name__)


def binary_classification(train_data: FineTuningDataset, test_data: FineTuningDataset):
    """
    Perform binary classification using various machine learning models and evaluate their performance.

    Args:
        train_data (FineTuningDataset): The training dataset.
        test_data (FineTuningDataset): The test dataset.
    """
    log.info("Splitting train data set...")
    train_subset, val_subset = _split_train_dataset(train_data)

    log.info("Performing grid search for binary classification...")
    methods = [CalibratedClassifierCV(LinearSVC()), LogisticRegression(), RandomForestClassifier(), MultinomialNB()]
    do_preprocessing = [True, False]
    pos_proportions = [0.1, 0.25, 0.5]
    for method in methods:
        for do_prep in do_preprocessing:
            for pos_proportion in pos_proportions:
                log.info(f"Using {method} with preprocessing={do_prep} and pos_proportion={pos_proportion}...")
                x_train, y_train, x_val, y_val, x_test, y_test \
                    = _prepare_data(train_subset.dataset, val_subset.dataset, test_data, do_prep, pos_proportion)

                log.info(f"Fitting {method} model...")
                binary_clf = method
                binary_clf.fit(x_train, y_train)

                _evaluate_model(binary_clf, x_train, y_train, x_val, y_val, x_test, y_test)


def _split_train_dataset(train_data: FineTuningDataset) -> Tuple[Subset, Subset]:
    """
    Split the training dataset into training and validation subsets.

    Args:
        train_data (FineTuningDataset): The training dataset.

    Returns:
        Tuple[Subset, Subset]: The training and validation subsets.
    """
    train_size = len(train_data)
    val_size = int(train_size * 0.1)
    train_size = train_size - val_size
    train_subset, val_subset = random_split(train_data, [train_size, val_size])
    train_subset.dataset = deepcopy(train_subset.dataset)
    val_subset.dataset = deepcopy(val_subset.dataset)
    train_subset.dataset.hf_dataset = train_subset.dataset.hf_dataset.select(train_subset.indices)
    val_subset.dataset.hf_dataset = val_subset.dataset.hf_dataset.select(val_subset.indices)

    return train_subset, val_subset


def _prepare_data(train_data: FineTuningDataset, val_data: FineTuningDataset, test_data: FineTuningDataset,
                  do_prep: bool = True, pos_proportion: float = None) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare the data for training, validation, and testing.

    Args:
        train_data (FineTuningDataset): The training dataset.
        val_data (FineTuningDataset): The validation dataset.
        test_data (FineTuningDataset): The test dataset.
        do_prep (bool): Whether to perform preprocessing.
        pos_proportion (float): The proportion of positive samples in the training dataset.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: The prepared data arrays.
    """
    if pos_proportion:
        train_data = _balance_dataset(train_data, pos_proportion)
        if do_prep:
            preprocess([train_data.dataset, test_data, val_data])
        tf_idf_vectorize(train_data.dataset, val_data, test_data)
        log.info("Loading examples from datasets...")
        x_train, y_train = _to_numpy_arrays(train_data.dataset)
    else:
        if do_prep:
            preprocess([train_data, test_data, val_data])
        tf_idf_vectorize(train_data, val_data, test_data)
        log.info("Loading examples from datasets...")
        x_train, y_train = _to_numpy_arrays(train_data)

    x_val, y_val = _to_numpy_arrays(val_data)
    x_test, y_test = _to_numpy_arrays(test_data)

    return x_train, y_train, x_val, y_val, x_test, y_test,


def _to_numpy_arrays(dataset: FineTuningDataset) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a dataset to numpy arrays.

    Args:
        dataset (FineTuningDataset): The dataset to convert.

    Returns:
        Tuple[np.ndarray, np.ndarray]: The feature and label arrays.
    """
    x, y = zip(*[(xi.numpy(), yi.item()) for xi, yi in dataset])
    x = np.array(x)
    y = np.rint(np.array(y)).astype(int)

    return x, y


def _balance_dataset(dataset: FineTuningDataset, pos_proportion: float = 0.5) -> Subset:
    """
    Balance the dataset to have a specified proportion of positive samples.

    Args:
        dataset (FineTuningDataset): The dataset to balance.
        pos_proportion (float): The desired proportion of positive samples.

    Returns:
        Subset: The balanced dataset subset.
    """
    targets = dataset.get_label_vector()

    # Extract positive and negative indices for the current label
    positive_indices = (targets == 1).nonzero(as_tuple=True)[0]
    negative_indices = (targets == 0).nonzero(as_tuple=True)[0]
    log.info(f"The initial proportion of positive samples is {len(positive_indices) / len(targets)}.")

    # Balance positive and negative samples
    pos_count = len(positive_indices)
    neg_count = int((1 / pos_proportion - 1) * pos_count)
    balanced_negative_indices = resample(negative_indices.tolist(), n_samples=neg_count, random_state=42)

    # Combine indices and create a balanced subset
    balanced_indices = positive_indices.tolist() + balanced_negative_indices
    balanced_subset = Subset(deepcopy(dataset), balanced_indices)
    balanced_subset.dataset.hf_dataset = balanced_subset.dataset.hf_dataset.select(balanced_indices)
    log.info(
        f"The balanced proportion of positive samples is {len(positive_indices) / len(balanced_indices)}.")

    return balanced_subset


def _evaluate_model(model: CalibratedClassifierCV, x_train: np.ndarray, y_train: np.ndarray,
                    x_val: np.ndarray, y_val: np.ndarray, x_test: np.ndarray, y_test: np.ndarray) -> None:
    """
    Evaluate the performance of a model on training, validation, and test sets.

    Args:
        model (CalibratedClassifierCV): The trained model.
        x_train (np.ndarray): The training features.
        y_train (np.ndarray): The training labels.
        x_val (np.ndarray): The validation features.
        y_val (np.ndarray): The validation labels.
        x_test (np.ndarray): The test features.
        y_test (np.ndarray): The test labels.
    """
    log.info("Train set evaluation...")
    train_roc_auc = roc_auc_score(y_train, model.predict_proba(x_train)[:, 1])
    log.info(f"Train set ROC AUC: {train_roc_auc}")
    report = classification_report(y_train, model.predict(x_train))
    log.info(f"Train set classification report:\n{report}")

    log.info("Validation set evaluation...")
    val_roc_auc = roc_auc_score(y_val, model.predict_proba(x_val)[:, 1])
    log.info(f"Validation set ROC AUC: {val_roc_auc}")
    report = classification_report(y_val, model.predict(x_val))
    log.info(f"Validation set classification report:\n{report}")

    log.info("Test set evaluation...")
    test_roc_auc = roc_auc_score(y_test, model.predict_proba(x_test)[:, 1])
    log.info(f"Test set ROC AUC: {test_roc_auc}")
    report = classification_report(y_test, model.predict(x_test))
    log.info(f"Test set classification report:\n{report}")
