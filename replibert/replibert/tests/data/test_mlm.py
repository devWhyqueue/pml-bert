from unittest.mock import patch

import numpy as np

import data


# Create a mock tokenizer to simulate the tokenizer behavior
class MockTokenizer:
    class BERT:
        vocab_size = 100  # Arbitrary small vocab size for testing
        all_special_ids = [0, 1, 2, 3]  # Example special token IDs
        mask_token_id = 4  # ID of the mask token


# Patch the tokenizer in the module with the mock tokenizer
@patch('data.mlm.tokenizer', new=MockTokenizer)
def test_mask_sequences_full_masking():
    # Recompute VALID_TOKEN_IDS after patching
    data.mlm.VALID_TOKEN_IDS = np.array([
        tok for tok in range(MockTokenizer.BERT.vocab_size)
        if tok not in MockTokenizer.BERT.all_special_ids
    ])

    # Prepare input_ids
    input_ids = np.array([
        [5, 6, 7, 8, 9],  # A sequence of token IDs
        [10, 0, 11, 12, 3],  # Sequence with special tokens (0 and 3)
    ])

    # Set mask_prob to 1.0 to force masking all non-special tokens
    mask_prob = 1.0

    # Set random seed for reproducibility
    np.random.seed(42)

    masked_input_ids, labels = data.mlm.mask_sequences(input_ids, mask_prob)

    # Expected labels
    expected_labels = np.full_like(input_ids, -100)
    special_tokens_mask = np.isin(input_ids, MockTokenizer.BERT.all_special_ids)
    mask = (~special_tokens_mask)
    expected_labels[mask] = input_ids[mask]

    assert np.array_equal(labels, expected_labels), "Labels are not correctly set."

    # Check that special tokens are not masked
    assert np.array_equal(masked_input_ids[special_tokens_mask],
                          input_ids[special_tokens_mask]), "Special tokens should not be masked."

    # Recompute mask, mask_token_mask, random_token_mask as in the function
    mask = (~special_tokens_mask)
    masking_probs = np.array([
        [0.02058449, 0.96990985, 0.83244264, 0.21233911, 0.18182497],
        [0.18340451, 0.30424224, 0.52475643, 0.43194502, 0.29122914]
    ])
    mask_token_mask = mask & (masking_probs < 0.8)
    random_token_mask = mask & (masking_probs >= 0.8) & (masking_probs < 0.9)
    unchanged_mask = mask & (~mask_token_mask) & (~random_token_mask)

    # Assert that masked_input_ids[mask_token_mask] == mask_token_id
    assert np.array_equal(masked_input_ids[mask_token_mask], np.full(np.sum(mask_token_mask),
                                                                     MockTokenizer.BERT.mask_token_id)), "Masked tokens not set to mask_token_id correctly."

    # Assert that masked_input_ids[random_token_mask] are in VALID_TOKEN_IDS
    assert np.all(
        np.isin(masked_input_ids[random_token_mask],
                data.mlm.VALID_TOKEN_IDS)), "Random tokens are not valid token IDs."

    # Assert that masked_input_ids[unchanged_mask] == input_ids[unchanged_mask]
    assert np.array_equal(masked_input_ids[unchanged_mask],
                          input_ids[unchanged_mask]), "Unchanged masked tokens do not match the original input IDs."

    # Assert that special tokens are unchanged
    assert np.array_equal(masked_input_ids[special_tokens_mask],
                          input_ids[special_tokens_mask]), "Special tokens should remain unchanged."


@patch('data.mlm.tokenizer', new=MockTokenizer)
def test_mask_sequences_no_masking():
    # Recompute VALID_TOKEN_IDS after patching
    data.mlm.VALID_TOKEN_IDS = np.array([
        tok for tok in range(MockTokenizer.BERT.vocab_size)
        if tok not in MockTokenizer.BERT.all_special_ids
    ])

    # Prepare input_ids
    input_ids = np.array([
        [5, 6, 7, 8, 9],
        [10, 0, 11, 12, 3],
    ])

    # Set mask_prob to 0.0 to prevent any masking
    mask_prob = 0.0

    # Set random seed
    np.random.seed(42)

    masked_input_ids, labels = data.mlm.mask_sequences(input_ids, mask_prob)

    # Labels should be all -100
    expected_labels = np.full_like(input_ids, -100)
    assert np.array_equal(labels, expected_labels), "Labels should be all -100 when no data.mlm."

    # Masked input IDs should be same as input_ids
    assert np.array_equal(masked_input_ids, input_ids), "Masked input IDs should match input IDs when no data.mlm."


@patch('data.mlm.tokenizer', new=MockTokenizer)
def test_mask_batch_properties():
    # Recompute VALID_TOKEN_IDS after patching
    data.mlm.VALID_TOKEN_IDS = np.array([
        tok for tok in range(MockTokenizer.BERT.vocab_size)
        if tok not in MockTokenizer.BERT.all_special_ids
    ])

    # Prepare batch
    batch = {
        'input_ids': [
            [5, 6, 7, 8, 9],
            [10, 0, 11, 12, 3],
        ],
        'attention_mask': [
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
        ],
        'special_tokens_mask': [
            [0, 0, 0, 0, 0],
            [0, 1, 0, 0, 1],
        ],
        'doc_id': [1, 2],
    }

    mask_prob = 1.0
    np.random.seed(42)  # Set random seed for reproducibility

    # Mask the batch
    masked_batch = data.mlm.mask_batch(batch, mask_prob)

    # Convert lists to numpy arrays for easier handling
    input_ids = np.array(batch['input_ids'])
    masked_input_ids = np.array(masked_batch['input_ids'])
    labels = np.array(masked_batch['mlm_labels'])

    # Assert that masked tokens match the expected masking probability
    num_masked_tokens = np.sum(labels != -100)
    assert num_masked_tokens > 0, "No tokens were masked, but some tokens should be masked with mask_prob = 1.0"

    # Assert that all special tokens remain unmasked
    special_tokens_mask = np.isin(input_ids, MockTokenizer.BERT.all_special_ids)
    assert np.array_equal(masked_input_ids[special_tokens_mask], input_ids[special_tokens_mask]), (
        "Special tokens should not be masked."
    )

    # Assert that masked tokens are replaced correctly
    mask_token_id = MockTokenizer.BERT.mask_token_id
    mask_token_replaced = (masked_input_ids == mask_token_id) & (labels != -100)
    random_token_replaced = (~mask_token_replaced) & (labels != -100)

    assert np.sum(mask_token_replaced) > 0, "Some tokens should be replaced with the mask token."
    assert np.all(np.isin(masked_input_ids[random_token_replaced], data.mlm.VALID_TOKEN_IDS)), (
        "Randomly replaced tokens should be valid token IDs."
    )

    # Assert that the labels match the input_ids where masking occurred
    assert np.all((labels == input_ids) | (labels == -100)), (
        "Labels should match input IDs where tokens are masked and be -100 elsewhere."
    )

    # Check that other fields remain unchanged
    assert masked_batch['attention_mask'] == batch['attention_mask'], "Attention masks should be unchanged."
    assert masked_batch['special_tokens_mask'] == batch[
        'special_tokens_mask'], "Special tokens masks should be unchanged."
    assert masked_batch['doc_id'] == batch['doc_id'], "Document IDs should be unchanged."


@patch('data.mlm.tokenizer', new=MockTokenizer)
def test_mask_sequences_special_tokens_only():
    # Recompute VALID_TOKEN_IDS after patching
    data.mlm.VALID_TOKEN_IDS = np.array([
        tok for tok in range(MockTokenizer.BERT.vocab_size)
        if tok not in MockTokenizer.BERT.all_special_ids
    ])

    # Prepare input_ids with only special tokens
    input_ids = np.array([
        [0, 1, 2, 3],
        [3, 2, 1, 0],
    ])

    # Set mask_prob to 1.0
    mask_prob = 1.0

    # Set random seed
    np.random.seed(42)

    masked_input_ids, labels = data.mlm.mask_sequences(input_ids, mask_prob)

    # Since all tokens are special tokens, they should not be masked
    assert np.array_equal(masked_input_ids, input_ids), "Special tokens should not be masked."

    # Labels should be all -100
    expected_labels = np.full_like(input_ids, -100)
    assert np.array_equal(labels, expected_labels), "Labels should be all -100 when only special tokens are present."


@patch('data.mlm.tokenizer', new=MockTokenizer)
def test_mask_sequences_empty_input():
    # Recompute VALID_TOKEN_IDS after patching
    data.mlm.VALID_TOKEN_IDS = np.array([
        tok for tok in range(MockTokenizer.BERT.vocab_size)
        if tok not in MockTokenizer.BERT.all_special_ids
    ])

    # Prepare empty input_ids
    input_ids = np.array([[]])

    # Set mask_prob to 1.0
    mask_prob = 1.0

    # Set random seed
    np.random.seed(42)

    masked_input_ids, labels = data.mlm.mask_sequences(input_ids, mask_prob)

    # Both masked_input_ids and labels should be empty arrays
    assert masked_input_ids.size == 0, "Masked input IDs should be empty."
    assert labels.size == 0, "Labels should be empty."
