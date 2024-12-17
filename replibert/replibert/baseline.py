from typing import Tuple

import numpy as np
import torch
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from configuration.config import get_logger
from data.finetuning.datasets import FineTuningDataset
from data.finetuning.transform import tf_idf_vectorize, preprocess, balance_dataset

log = get_logger(__name__)


def binary_classification(train_data: FineTuningDataset, val_data: FineTuningDataset, test_data: FineTuningDataset):
    """
    Perform binary classification using various machine learning models and evaluate their performance.

    Args:
        train_data (FineTuningDataset): The training dataset.
        val_data (FineTuningDataset): The validation dataset.
        test_data (FineTuningDataset): The test dataset.
    """
    log.info("Performing grid search for binary classification...")
    methods = [LogisticRegression(), CalibratedClassifierCV(LinearSVC()), MultinomialNB()]
    do_preprocessing = [True, False]
    pos_proportions = [None, 0.1, 0.25, 0.5]
    for method in methods:
        for do_prep in do_preprocessing:
            for pos_proportion in pos_proportions:
                log.info(f"Using {method} with preprocessing={do_prep} and pos_proportion={pos_proportion}...")
                x_train, y_train, x_val, y_val, x_test, y_test \
                    = _prepare_data(train_data, val_data, test_data, do_prep, pos_proportion)

                log.info(f"Fitting {method} model...")
                binary_clf = method
                binary_clf.fit(x_train, y_train)

                _evaluate_model(binary_clf, x_train, y_train, x_val, y_val, x_test, y_test)


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
        train_data = balance_dataset(train_data, pos_proportion, in_place=True)
        if do_prep:
            train_data.dataset.hf_dataset, val_data.hf_dataset, test_data.hf_dataset \
                = preprocess([train_data.dataset.hf_dataset, val_data.hf_dataset,
                              test_data.hf_dataset], test_data.text_field)
        tf_idf_vectorize(train_data.dataset, val_data, test_data)
        log.info("Processing examples from datasets...")
        x_train, y_train = _to_numpy_arrays(train_data.dataset)
    else:
        if do_prep:
            train_data.hf_dataset, val_data.hf_dataset, test_data.hf_dataset \
                = preprocess([train_data.hf_dataset, val_data.hf_dataset, test_data.hf_dataset],
                             test_data.text_field)
        tf_idf_vectorize(train_data, val_data, test_data)
        log.info("Processing examples from datasets...")
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
    x = dataset.get_input_vector().numpy()
    y = torch.round(dataset.get_label_vector()).int().numpy()
    return x, y


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
