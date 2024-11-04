import nltk
from transformers import AutoTokenizer

BERT = AutoTokenizer.from_pretrained('bert-base-uncased', use_fast=True)

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')
