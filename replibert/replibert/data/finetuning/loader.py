from datasets import load_dataset
from torch.utils.data import IterableDataset, DataLoader

def load_data(dataset="glue", task="sst2", transformation=None, n_train=None, n_test=None, batch_size=16):
    """
    Load data lazily with transformations and create train/test DataLoaders.

    Parameters:
        dataset (str): Dataset name
        task (str): Subdataset
        transformation (callable): Transformation applied to each sample.
        n_train (int): Number of training samples (None for all available).
        n_test (int): Number of test samples (None for all available).
        batch_size (int): Batch size for DataLoaders.

    Returns:
        tuple: DataLoaders for training and testing datasets.
    """

    # Load dataset with streaming
    hf_dataset = load_dataset(dataset, task, streaming=True)

    # Helper function to limit samples
    def limit_samples(data_stream, n_samples):
        if n_samples:
            return (sample for i, sample in enumerate(data_stream) if i < n_samples)
        return data_stream

    # Define a PyTorch IterableDataset
    class HuggingFaceIterableDataset(IterableDataset):
        def __init__(self, data_stream, transformation=None):
            self.data_stream = data_stream
            self.transformation = transformation

        def __iter__(self):
            for example in self.data_stream:
                if self.transformation:
                    yield self.transformation(example)
                else:
                    yield example

    # Prepare train and test datasets
    train_stream = limit_samples(hf_dataset["train"], n_train)
    test_stream = limit_samples(hf_dataset["validation"], n_test)

    train_dataset = HuggingFaceIterableDataset(train_stream, transformation=transformation)
    test_dataset = HuggingFaceIterableDataset(test_stream, transformation=transformation)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, test_loader