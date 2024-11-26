import re

import nltk
import torch
from nltk import WordNetLemmatizer, word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import BertTokenizer

from configuration.config import get_logger
from data.finetuning.datasets import FineTuningDataset

try:
    nltk.data.find('stopwords')
    nltk.data.find('wordnet')
except LookupError:
    nltk.download('stopwords')
    nltk.download('wordnet')

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
        = train_dataset.hf_dataset.map(_transform_batch, batched=True, batch_size=512, desc="Vectorizing train dataset")
    val_dataset.hf_dataset \
        = val_dataset.hf_dataset.map(_transform_batch, batched=True, batch_size=512,
                                     desc="Vectorizing validation dataset")
    test_dataset.hf_dataset \
        = test_dataset.hf_dataset.map(_transform_batch, batched=True, batch_size=512, desc="Vectorizing test dataset")


def bert_tokenize(text: str) -> dict:
    """
    Tokenizes the input text using the BERT tokenizer.

    Args:
        text (str): The text string to tokenize.

    Returns:
        dict: A dictionary containing the tokenized text as tensors.
    """
    return BertTokenizer.from_pretrained("bert-base-uncased")(
        text,
        max_length=512,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )


def preprocess(datasets: list[FineTuningDataset]):
    """
    Preprocesses the text data in the given datasets using NLTK.

    Args:
        datasets (list[FineTuningDataset]): A list of datasets to preprocess.

    Returns:
        None
    """
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    for dataset in datasets:
        _preprocess_dataset(dataset, dataset.text_field, lemmatizer, stop_words, "Preprocessing dataset")


def _preprocess_dataset(dataset, text_field, lemmatizer, stop_words, description):
    """
    Preprocesses the text data in a single dataset.

    Args:
        dataset (FineTuningDataset): The dataset to preprocess.
        text_field (str): The field containing the text data.
        lemmatizer (WordNetLemmatizer): The lemmatizer to use.
        stop_words (set): The set of stopwords to remove.
        description (str): A description for the preprocessing task.

    Returns:
        None
    """

    def _preprocess_batch(batch):
        batch[text_field] = [_preprocess_text(text, lemmatizer, stop_words) for text in batch[text_field]]
        return batch

    dataset.hf_dataset = dataset.hf_dataset.map(_preprocess_batch, batched=True, batch_size=512, desc=description)


def _preprocess_text(text, lemmatizer, stop_words):
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
