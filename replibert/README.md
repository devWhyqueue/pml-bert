# replibert

**Project Goal:** Replicate the BERT model from scratch based on the original paper, implementing data processing, model
architecture, and training components from the ground up.

## Project Structure

```
replibert/
├── replibert/               # Main source code directory
│   ├── configuration/       # App and logging configuration
│   ├── data/                # Data download, preprocessing and labeling
|   ├── tests/               # Test suite
│   ├── main.py              # Click CLI with commands like download or train
│   ├── model.py             # BERT model architecture
│   ├── baseline.py          # Baseline models for the datasets
│   ├── training.py          # Training loop and evaluation code
├── scripts/                 # Directory for bash scripts to be executed on the cluster
│   ├── training/
│   │   └── combine.sh       # Combine the datasets
│   │   └── tokenize.sh      # Tokenize the combined dataset
│   │   └── mlm.sh           # Prepare training data for MLM
│   └── run.sh               # Run the replibert CLI (command and options must be specified)
│   └── baseline.sh          # Run the baseline model
│   └── build_image.sh       # Build the SIF
├── data/                    # Jigsaw Dataset (for download with HF) and notebook for data exploration
│   ├── plots/               # Data exploration plots
├── experiments/             # Directory for storing experiment logs and results
│   ├── logs/                # Training logs
│   └── checkpoints/         # Model checkpoints
├── README.md                # Project documentation
└── pyproject.toml           # Poetry project dependencies and settings

```

## Local Installation

### 1. Prerequisites

Ensure you have Python 3.10+ and Poetry installed. To install Poetry, follow:

https://python-poetry.org/docs/

### 2. Install Dependencies

Navigate to the project root and use Poetry to install dependencies:

```bash
cd replibert
poetry install -E cpu
# or poetry install -E gpu --with gpu
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

### 3. Run a replibert script

To run the baseline method for example, use:

```bash
chmod +x scripts/run.sh
./scripts/run.sh baseline --dataset_name sst2 --dataset_dir /home/space/datasets/sst2
```

You can watch the logs with:

```bash
tail -f logs/run.log
```

## Configuration

All configuration options are managed through `config.yaml`. Here, you can specify paths, model parameters, and training
settings.

## Usage

To see all available commands and options, run:

```bash
python replibert/main.py --help
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
