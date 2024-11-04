import pytest
from datasets import Dataset

from data import tokenizer
from data.preprocess import concatenate, split_text_into_spans, break_down_into_spans_and_tokenize, tokenize_spans


def test_split_text_normal_case():
    text = "Hello world. This is a test. Testing sentence splitting."
    spans = split_text_into_spans(text, max_tokens=10)
    assert len(spans) > 0
    for span in spans:
        tokens = tokenizer.BERT.encode(span, add_special_tokens=False)
        assert len(tokens) <= 10


def test_split_text_with_long_sentence():
    long_sentence = " ".join(["word"] * 300)
    text = f"{long_sentence}. Short sentence."
    spans = split_text_into_spans(text, max_tokens=50)
    assert all(long_sentence not in span for span in spans)


def test_split_text_empty_input():
    text = ""
    spans = split_text_into_spans(text)
    assert spans == []


def test_split_text_single_word():
    text = "Word"
    spans = split_text_into_spans(text)
    assert spans == ["Word"]


def test_split_text_exactly_max_tokens():
    exact_length_sentence = " ".join(["word"] * 255)
    text = f"{exact_length_sentence}. Another sentence."
    spans = split_text_into_spans(text, max_tokens=256)
    assert exact_length_sentence in spans[0]


def test_tokenize_spans_normal_case():
    spans = ["This is a test span."]
    outputs = tokenize_spans(spans, max_length=10)
    assert 'input_ids' in outputs
    assert 'attention_mask' in outputs
    assert 'special_tokens_mask' in outputs
    assert len(outputs['input_ids'][0]) == 10
    assert len(outputs['attention_mask'][0]) == 10
    assert len(outputs['special_tokens_mask'][0]) == 10


def test_tokenize_spans_exceeding_max_length():
    spans = [" ".join(["word"] * 300)]
    outputs = tokenize_spans(spans, max_length=256)
    assert len(outputs['input_ids'][0]) == 256
    assert outputs['input_ids'][0][-1] == tokenizer.BERT.sep_token_id


def test_break_down_into_spans_and_tokenize():
    # Test normal batch
    batch = {'text': ["First document. Second sentence.", "Another document here."]}
    outputs = break_down_into_spans_and_tokenize(batch)
    assert 'input_ids' in outputs
    assert 'attention_mask' in outputs
    assert 'special_tokens_mask' in outputs
    assert 'doc_id' in outputs
    assert len(outputs['input_ids']) == len(outputs['doc_id']) > 0

    # Test empty batch
    batch = {'text': []}
    outputs = break_down_into_spans_and_tokenize(batch)
    assert outputs == {
        'input_ids': [],
        'attention_mask': [],
        'special_tokens_mask': [],
        'doc_id': [],
    }

    # Test batch with empty string
    batch = {'text': [""]}
    outputs = break_down_into_spans_and_tokenize(batch)
    assert outputs == {
        'input_ids': [],
        'attention_mask': [],
        'special_tokens_mask': [],
        'doc_id': [],
    }

    # Test batch with long sentences
    long_sentence = " ".join(["word"] * 300)
    batch = {'text': [long_sentence]}
    outputs = break_down_into_spans_and_tokenize(batch)
    assert outputs == {
        'input_ids': [],
        'attention_mask': [],
        'special_tokens_mask': [],
        'doc_id': [],
    }


def test_concatenate_with_valid_datasets():
    # Create sample datasets with 'text' and other columns
    data1 = {'text': ['Hello world', 'This is a test'], 'label': [0, 1]}
    data2 = {'text': ['Another example', 'More data'], 'category': ['A', 'B']}
    ds1 = Dataset.from_dict(data1)
    ds2 = Dataset.from_dict(data2)

    # Concatenate datasets
    unified_ds = concatenate([ds1, ds2], keep=['text'])

    # Check that the concatenated dataset has only the 'text' column
    assert unified_ds.column_names == ['text']
    # Check that the 'text' data is combined correctly
    expected_text = data1['text'] + data2['text']
    assert unified_ds['text'] == expected_text


def test_concatenate_with_no_datasets():
    # Expect an error when trying to concatenate an empty list
    with pytest.raises(ValueError, match="Unable to concatenate an empty list of datasets."):
        concatenate([])


def test_concatenate_with_mixed_columns():
    # Create datasets where only some have the 'text' column
    data1 = {'text': ['Hello world'], 'label': [0]}
    data2 = {'category': ['A']}
    ds1 = Dataset.from_dict(data1)
    ds2 = Dataset.from_dict(data2)

    # Concatenate datasets
    unified_ds = concatenate([ds1, ds2], keep=['text'])

    # Check that the concatenated dataset has only the 'text' column
    assert unified_ds.column_names == ['text']
    # Check that missing 'text' entries are handled (e.g., as None)
    expected_text = data1['text'] + [None]
    assert unified_ds['text'] == expected_text
