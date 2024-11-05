from typing import List, Dict, Tuple

import numpy as np

from configuration.config import get_logger
from data import tokenizer

log = get_logger(__name__)

# Precompute valid token IDs (non-special tokens)
VALID_TOKEN_IDS = np.array([
    tok for tok in range(tokenizer.BERT.vocab_size)
    if tok not in tokenizer.BERT.all_special_ids
])


def mask_batch(batch: Dict[str, List[List[int]]], mask_prob=0.15) -> Dict[str, List[List[int]]]:
    """
    Apply masking to a batch of sequences using NumPy for speed.

    Returns:
        Dict[str, List[List[int]]]: A dictionary containing masked input IDs and corresponding labels.
    """
    input_ids = np.array(batch['input_ids'])
    masked_input_ids, labels = mask_sequences(input_ids, mask_prob)
    return {
        'input_ids': masked_input_ids.tolist(),
        'attention_mask': batch['attention_mask'],
        'special_tokens_mask': batch['special_tokens_mask'],
        'doc_id': batch['doc_id'],
        'mlm_labels': labels.tolist(),
    }


def mask_sequences(input_ids: np.ndarray, mask_prob: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Mask sequences of input IDs.

    Args:
        input_ids (np.ndarray): Array of input IDs.
        mask_prob (float): Probability of masking a token.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Masked input IDs and labels.
    """
    special_tokens_mask = np.isin(input_ids, tokenizer.BERT.all_special_ids)
    probs = np.random.rand(*input_ids.shape)
    mask = (probs < mask_prob) & (~special_tokens_mask)
    masked_input_ids = np.copy(input_ids)
    labels = np.full_like(input_ids, fill_value=-100)
    labels[mask] = input_ids[mask]
    masking_probs = np.random.rand(*input_ids.shape)
    mask_token_mask = mask & (masking_probs < 0.8)
    random_token_mask = mask & (masking_probs >= 0.8) & (masking_probs < 0.9)
    masked_input_ids[mask_token_mask] = tokenizer.BERT.mask_token_id
    num_random_tokens = np.sum(random_token_mask)
    if num_random_tokens > 0:
        random_tokens = np.random.choice(VALID_TOKEN_IDS, size=num_random_tokens)
        masked_input_ids[random_token_mask] = random_tokens
    return masked_input_ids, labels
