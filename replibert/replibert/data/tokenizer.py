import nltk
from transformers import AutoTokenizer

BERT = AutoTokenizer.from_pretrained('bert-base-uncased', use_fast=True, model_max_length=512)

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')
