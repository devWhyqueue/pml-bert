# replibert

**Project Goal:** Replicate the BERT model from scratch based on the original paper, implementing data processing, model
architecture, and training components from the ground up.

## Project Structure

```
replibert/
├── replibert/               # Main source code directory
│   ├── __init__.py
│   ├── data/                # Data loading and preprocessing (CustomDataset class)
│   ├── model.py             # BERT model architecture
│   ├── training.py          # Training loop and evaluation code
│   ├── visualize.py         # Data visualization tools (Word Clouds, sample outputs)
│   └── config.yaml          # Configuration file for model and data settings
├── scripts/                 # Directory for bash scripts (cluster)
│   └── build_image.sh       # Build the SIF
│   └── run_container.sh     # Run the container
├── experiments/             # Directory for storing experiment logs and results
│   ├── logs/                # Training logs
│   └── checkpoints/         # Model checkpoints
├── tests/                   # Folder for all project tests
│   ├── test_main.py         # Test for main script
│   └── [other tests].py
├── README.md                # Project documentation
└── pyproject.toml           # Poetry project dependencies and settings

```

## Local Installation

### 1. Prerequisites

Ensure you have Python 3.12+ and Poetry installed. To install Poetry, follow:

https://python-poetry.org/docs/

### 2. Install Dependencies

Navigate to the project root and use Poetry to install dependencies:

```bash
cd replibert
poetry install -E cpu --with cpu --sync
# Alternatively, for GPU support:
# poetry install -E gpu --with gpu --sync
```

### 3. Activate the Virtual Environment

To activate the environment, use:

```bash
poetry shell
```

## Deployment on cluster

### 1. Clone the repo into your home directory

```bash
git clone https://github.com/devWhyqueue/pml-bert.git
```

### 2. Build the container

```bash
cd pml-bert/replibert
chmod +x scripts/build_image.sh
./scripts/build_image.sh
```

### 3. Run replibert

```bash
chmod +x scripts/run_container.sh
./scripts/run_container.sh
```

## Configuration

All configuration options are managed through `config.yaml`. Here, you can specify paths, model parameters, and training
settings.

## Usage

### Run Training

To start training, simply run the main script:

```bash
python replibert/main.py
```

### Data Visualization

To visualize the dataset (e.g., word clouds or sample outputs), use the visualization script:

```bash
python replibert/visualize.py
```

## Testing

All tests are located in the `tests/` folder and follow the pytest format. To run the tests:

```bash
pytest tests/
```

This includes a basic test for the main script (`test_main.py`) and any additional tests you add.

## Datasets

The datasets will not be included in this repository because they are too large.
They will be downloaded from Hugging Face's datasets library to the location specified in `app.yaml`.
Total file size is about 72 GB.
The BookCorpus is not used in contrast to the original paper, more information on the rationale can be found here:

https://www.notion.so/Document-data-collection-12eacf24f0f880289da9d7cd99f8b84a?pvs=4
