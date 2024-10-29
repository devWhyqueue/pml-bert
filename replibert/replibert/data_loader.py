from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, data_path):
        # Load data from data_path
        self.data = self.load_data(data_path)

    def __len__(self):
        # Return the number of samples in the dataset
        return len(self.data)

    def __getitem__(self, idx):
        # Return a sample from the dataset
        return self.data[idx]

    def load_data(self, path):
        # Implement loading logic; replace with actual loading mechanism
        # This example assumes data is already a list of samples
        return [sample for sample in range(100)]  # Example with 100 samples
