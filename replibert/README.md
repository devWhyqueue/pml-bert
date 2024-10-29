# replibert

**Project Goal:** Replicate the BERT model from scratch based on the original paper, implementing data processing, model
architecture, and training components from the ground up.

## Project Structure

```
replibert/
├── replibert/               # Main source code directory
│   ├── __init__.py
│   ├── data_loader.py       # Data loading and preprocessing (CustomDataset class)
│   ├── model.py             # BERT model architecture
│   ├── training.py          # Training loop and evaluation code
│   ├── visualize.py         # Data visualization tools (Word Clouds, sample outputs)
│   └── config.yaml          # Configuration file for model and data settings
├── data/                    # Directory for dataset files
│   └── <dataset_name>/      # Directory for each dataset
├── experiments/             # Directory for storing experiment logs and results
│   ├── logs/                # Training logs
│   └── checkpoints/         # Model checkpoints
├── tests/                   # Folder for all project tests
│   ├── test_main.py         # Test for main script
│   └── [other tests].py
├── README.md                # Project documentation
└── pyproject.toml           # Poetry project dependencies and settings

```

### Key Files

- **`replibert/data_loader.py`**: Contains the `CustomDataset` class for data loading and processing.
- **`replibert/model.py`**: Defines the BERT model architecture.
- **`replibert/train.py`**: Handles the main training loop and model evaluation.
- **`replibert/visualize.py`**: Implements data visualization functions.
- **`replibert/config.yaml`**: Stores configuration parameters for the project, such as paths and hyperparameters.
- **`tests/`**: Includes all tests, following the pytest format.

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
settings. **Example configuration**:

```yaml
data:
  path: "data/bookcorpus"
model:
  hidden_size: 768
  num_layers: 12
  num_heads: 12
training:
  batch_size: 32
  learning_rate: 0.00001
  num_epochs: 3
```

Update these values as needed for your experiments.

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

## Project Notes

- **Data Directory**: Download and place your dataset in `data/` as specified in `config.yaml`.
- **Custom Modifications**: You can adjust any hyperparameters, paths, or additional settings in `config.yaml` for
  different experiment configurations.
- **Dependencies**: If additional packages are required, update `pyproject.toml` and run `poetry install` to add them to
  the environment.

**Project Members**: Please document any changes you make for ease of collaboration and version control.
