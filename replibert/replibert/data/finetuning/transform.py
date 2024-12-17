import re
from copy import deepcopy

import nltk
import torch
from datasets import Dataset
from nltk import WordNetLemmatizer, word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils import resample
from torch.utils.data import Subset
from transformers import BertTokenizer

from configuration.config import get_logger
from data.finetuning.datasets import FineTuningDataset
from utils import get_available_cpus

try:
    nltk.data.find('stopwords')
    nltk.data.find('wordnet')
except LookupError:
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)

log = get_logger(__name__)


def tf_idf_vectorize(train_dataset: FineTuningDataset, val_dataset: FineTuningDataset, test_dataset: FineTuningDataset):
    """
    Vectorizes the text data in the given datasets using TF-IDF.

    Args:
        train_dataset (FineTuningDataset): The training dataset.
        val_dataset (FineTuningDataset): The validation dataset.
        test_dataset (FineTuningDataset): The test dataset.

    Returns:
        None
    """
    log.info("Fitting TF-IDF vectorizer to training dataset...")
    train_texts = train_dataset.get_texts()
    vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=5000).fit(train_texts)

    def _transform_batch(batch):
        texts = batch[test_dataset.text_field]
        vectorized = vectorizer.transform(texts).toarray()
        return {'tf_idf': torch.tensor(vectorized, dtype=torch.float32)}

    train_dataset.hf_dataset \
        = train_dataset.hf_dataset.map(_transform_batch, batched=True, batch_size=512,
                                       num_proc=get_available_cpus(), desc="Vectorizing train dataset")
    val_dataset.hf_dataset \
        = val_dataset.hf_dataset.map(_transform_batch, batched=True, batch_size=512, num_proc=get_available_cpus(),
                                     desc="Vectorizing validation dataset")
    test_dataset.hf_dataset \
        = test_dataset.hf_dataset.map(_transform_batch, batched=True, batch_size=512,
                                      num_proc=get_available_cpus(), desc="Vectorizing test dataset")


def bert_tokenize(dataset: Dataset, text_field: str) -> Dataset:
    """
    Tokenizes the text data in the given dataset using the BERT tokenizer.

    Args:
        dataset (Dataset): The HuggingFace dataset to tokenize.
        text_field (str): The field containing the text data.

    Returns:
        Dataset: The tokenized dataset.
    """
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", use_fast=True)

    def _tokenize_batch(batch):
        texts = batch[text_field]
        tokenized = tokenizer(
            texts,
            max_length=256,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            'input_ids': tokenized['input_ids'].tolist(),
            'attention_mask': tokenized['attention_mask'].tolist()
        }

    dataset = dataset.map(
        _tokenize_batch,
        batched=True,
        batch_size=512,
        num_proc=get_available_cpus(),
        desc="Tokenizing dataset"
    )

    return dataset


def preprocess(datasets: list[Dataset], text_field: str) -> list[Dataset]:
    """
    Preprocesses the text data in the given datasets using NLTK.

    Args:
        datasets (list[Dataset]): The datasets to preprocess.
        text_field (str): The field containing the text data.

    Returns:
        list[Dataset]: The preprocessed datasets.
    """
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    transformed = []
    for dataset in datasets:
        ds = _preprocess_dataset(dataset, text_field, lemmatizer, stop_words)
        transformed.append(ds)

    return transformed


def _preprocess_dataset(dataset: Dataset, text_field: str, lemmatizer: WordNetLemmatizer, stop_words: set) -> Dataset:
    """
    Preprocesses the text data in a single dataset.

    Args:
        dataset (Dataset): The dataset to preprocess.
        text_field (str): The field containing the text data.
        lemmatizer (WordNetLemmatizer): The lemmatizer to use.
        stop_words (set): The set of stopwords to remove.

    Returns:
        Dataset: The preprocessed dataset.
    """

    def _preprocess_batch(batch):
        batch[text_field] = [_preprocess_text(text, lemmatizer, stop_words) for text in batch[text_field]]
        return batch

    return dataset.map(_preprocess_batch, batched=True, batch_size=512, num_proc=get_available_cpus(),
                       desc="Preprocessing dataset")


def _preprocess_text(text: str, lemmatizer: WordNetLemmatizer, stop_words: set) -> str:
    """
    Preprocesses a single text string.

    Args:
        text (str): The text string to preprocess.
        lemmatizer (WordNetLemmatizer): The lemmatizer to use.
        stop_words (set): The set of stopwords to remove.

    Returns:
        str: The preprocessed text string.
    """
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)  # Remove URLs
    text = re.sub(r'@\w+|#\w+', '', text)  # Remove mentions and hashtags
    text = re.sub(r'\b\d+\b', '', text)  # Remove standalone numbers
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word.isalpha()]  # Keep only alphabetic tokens
    tokens = [word for word in tokens if word not in stop_words]  # Remove stopwords
    tokens = [lemmatizer.lemmatize(word) for word in tokens]  # Lemmatize tokens
    return ' '.join(tokens)


def balance_dataset(dataset: FineTuningDataset, pos_proportion: float = 0.5, in_place: bool = False) -> Subset:
    """
    Balance the dataset to have a specified proportion of positive samples.

    Args:
        dataset (FineTuningDataset): The dataset to balance.
        pos_proportion (float): The desired proportion of positive samples.
        in_place (bool): Whether to balance the dataset in place.

    Returns:
        Subset: The balanced dataset subset.
    """
    targets = dataset.get_label_vector()
    targets = targets.round().int()

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
    if in_place:
        balanced_subset.dataset.hf_dataset = balanced_subset.dataset.hf_dataset.select(balanced_indices)

    log.info(
        f"The balanced proportion of positive samples is {len(positive_indices) / len(balanced_indices)}.")

    return balanced_subset
