import nltk
from datasets import Dataset, concatenate_datasets

from data import tokenizer


def concatenate(datasets: list[Dataset], keep: list[str] = None) -> Dataset:
    """
    Concatenate multiple datasets into a single dataset, keeping only the 'text' column.

    Args:
        datasets (list[Dataset]): A list of datasets to be concatenated.
        keep (list[str], optional): A list of columns to keep in the concatenated dataset. Defaults to None (keeps all).

    Returns:
        Dataset: A single dataset containing the 'text' column from all input datasets.
    """
    combined_dataset = concatenate_datasets(datasets)
    if keep:
        combined_dataset = combined_dataset.remove_columns(
            [col for col in combined_dataset.column_names if col not in keep]
        )
    return combined_dataset


def break_down_into_spans_and_tokenize(batch) -> dict:
    """
    Split a batch of text documents into spans and tokenize each span.

    Args:
        batch (dict): A dictionary containing a 'text' key with a list of text strings.

    Returns:
        dict: A dictionary with tokenized data.
    """
    all_input_ids, all_attention_masks, all_special_tokens_masks, all_doc_ids = [], [], [], []
    for idx, text in enumerate(batch['text']):
        spans = split_text_into_spans(text)
        # If all sentences are too long, skip the document
        if not spans:
            continue

        tokenized_spans = tokenize_spans(spans)

        for i, span_input_ids in enumerate(tokenized_spans['input_ids']):
            all_input_ids.append(span_input_ids)
            all_attention_masks.append(tokenized_spans['attention_mask'][i])
            all_special_tokens_masks.append(tokenized_spans['special_tokens_mask'][i])
            all_doc_ids.append(idx)
    return {
        'input_ids': all_input_ids,
        'attention_mask': all_attention_masks,
        'special_tokens_mask': all_special_tokens_masks,
        'doc_id': all_doc_ids,
    }


def split_text_into_spans(text: str, max_tokens=256) -> list[str]:
    """
    Split text into spans of sentences, ensuring each span does not exceed a specified number of tokens.

    Args:
        text (str): The input text to be split into spans.
        max_tokens (int, optional): The maximum number of tokens per span. Defaults to 256.

    Returns:
        list[str]: A list of text spans, each containing sentences whose total token count does not exceed max_tokens.
    """
    sentences = nltk.sent_tokenize(text)
    spans, current_span, current_length = [], [], 0
    for sentence in sentences:
        tokens = tokenizer.BERT.tokenize(sentence)
        if len(tokens) > max_tokens:
            continue  # Skip sentences longer than max length
        if current_length + len(tokens) <= max_tokens:
            current_span.append(sentence)
            current_length += len(tokens)
        else:
            spans.append(' '.join(current_span))
            current_span, current_length = [sentence], len(tokens)
    if current_span:
        spans.append(' '.join(current_span))

    return spans


def tokenize_spans(spans: list[str], max_length=256) -> dict:
    """
    Tokenize a list of text spans into input IDs, attention masks, and special tokens masks.

    Args:
        spans (list[str]): A list of text spans to be tokenized.
        max_length (int, optional): The maximum length of the tokenized output. Defaults to 256.

    Returns:
        dict: A dictionary containing the tokenized output, including input IDs, attention mask and special tokens mask.
    """
    return tokenizer.BERT(
        spans,
        max_length=max_length,
        truncation=True,
        padding='max_length',
        return_special_tokens_mask=True,
        return_attention_mask=True,
    )
