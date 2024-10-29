# Datasets

The datasets will not be included in this repository because they are too large.
You can proceed as follows.

## BookCorpus

The BooksCorpus dataset was initially a large corpus of books for language modeling.
However, it is no longer freely available via the original University of Toronto website due to copyright restrictions.
There are substitutes on Hugging Face Datasets but slightly different.
Also keep ethical considerations in mind when using this dataset.

You can download the dataset via the `datasets` library from Hugging Face.

```python
from datasets import load_dataset

# Load BookCorpus dataset from Hugging Face
dataset = load_dataset("bookcorpus/bookcorpus")

# Optional: save locally in your data directory for future use
dataset.save_to_disk("data/bookcorpus")
```

## English Wikipedia

1. Download the latest English Wikipedia dump from [here](https://dumps.wikimedia.org/enwiki/latest/).

  ```
   wget https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2
  ```

2. Install WikiExtractor:

- WikiExtractor is a tool that extracts and cleans Wikipedia data into plain text format.
  ```bash
  git clone https://github.com/attardi/wikiextractor.git
  cd wikiextractor
  python setup.py install
  ```

3. Extract and Clean Wikipedia Data:

- Run WikiExtractor to process the Wikipedia XML file and produce plain text files.
  ```bash
  python -m wikiextractor.WikiExtractor enwiki-latest-pages-articles.xml.bz2 -o data/wikipedia
  ```
- The output will be saved in data/wikipedia as plain text files, which you can use for training.
