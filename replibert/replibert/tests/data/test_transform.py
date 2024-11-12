from unittest.mock import MagicMock, patch, call

import pytest
from datasets import Dataset

from data.training.transform import combine_datasets, spanify_and_tokenize


@pytest.fixture
def mock_datasets():
    # Create mock datasets
    data1 = {'text': ['Sample text 1', 'Sample text 2']}
    data2 = {'text': ['Sample text 3', 'Sample text 4']}
    ds1 = Dataset.from_dict(data1)
    ds2 = Dataset.from_dict(data2)
    return [ds1, ds2]


@patch("data.transform.log")
@patch("data.transform.concatenate")
@patch("data.transform.ds.Dataset.save_to_disk")
def test_combine_datasets(mock_save_to_disk, mock_concatenate, mock_log, mock_datasets):
    # Mock concatenate return value
    mock_combined = Dataset.from_dict({'text': ['Combined text']})
    mock_combined.shuffle = MagicMock(return_value=mock_combined)
    mock_concatenate.return_value = mock_combined

    # Call function
    destination = "mock_destination"
    keep = ['text']
    shards = 2
    shuffle = True
    combine_datasets(mock_datasets, destination, keep, shards, shuffle)

    # Assertions
    mock_concatenate.assert_called_once_with(mock_datasets, keep)
    mock_combined.shuffle.assert_called_once_with(seed=42)
    mock_combined.save_to_disk.assert_called_once_with(destination, num_proc=shards)

    # Log calls
    assert mock_log.info.call_count == 4
    expected_log_calls = [
        call(f"Unifying {len(mock_datasets)} datasets..."),
        call("Shuffling unified dataset..."),
        call(f"Saving unified dataset to {destination}..."),
        call(f"Saved unified dataset to {destination}.")
    ]
    mock_log.info.assert_has_calls(expected_log_calls, any_order=False)


@patch("data.transform.log")
@patch("data.transform.break_down_into_spans_and_tokenize")
@patch("data.transform.ds.Dataset.shard")
@patch("data.transform.ds.Dataset.save_to_disk")
@patch("data.transform.ds.load_from_disk")
def test_spanify_and_tokenize(mock_load_from_disk, mock_save_to_disk, mock_shard, mock_tokenize, mock_log):
    # Mock dataset and shard instances for each call
    mock_datasets = [MagicMock(), MagicMock()]
    mock_shards = [MagicMock(), MagicMock()]

    # Setup return values for each call
    mock_load_from_disk.side_effect = mock_datasets
    for mock_dataset, m_shard in zip(mock_datasets, mock_shards):
        mock_dataset.shard.return_value = m_shard
        m_shard.map.return_value = m_shard  # Tokenized result

    # Call the function with multiple shard indices
    dataset_dir = "mock_dataset_dir"
    destination = "mock_destination"
    shard_indices = [0, 1]
    spanify_and_tokenize(dataset_dir, destination, shard_indices)

    # Verify that load_from_disk was called twice, once for each shard index
    assert mock_load_from_disk.call_count == len(shard_indices)

    # Check that each mock dataset was sharded with the correct index
    mock_datasets[0].shard.assert_called_once_with(num_shards=100, index=0)
    mock_datasets[1].shard.assert_called_once_with(num_shards=100, index=1)

    # Verify save_to_disk was called for each processed shard with the correct destination
    mock_shards[0].save_to_disk.assert_called_once_with(f"{destination}/0")
    mock_shards[1].save_to_disk.assert_called_once_with(f"{destination}/1")

    # Check log messages for each shard
    expected_log_calls = [
        call(f"Tokenizing {dataset_dir} shard 0..."),
        call(f"Tokenizing {dataset_dir} shard 1...")
    ]
    mock_log.info.assert_has_calls(expected_log_calls, any_order=False)
