import pytest
import yaml

from replibert.main import main


def test_main_runs():
    """Test that the main function runs without errors and loads configuration."""

    # Load the configuration file as a basic check
    with open("../configuration/app.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Verify that configuration is loaded as a dictionary and contains expected keys
    assert isinstance(config, dict), "Config should be a dictionary."
    assert "model" in config, "Config should contain a 'model' section."
    assert "training" in config, "Config should contain a 'training' section."

    # Run the main function
    try:
        main()  # Assuming main() handles configuration and execution
    except Exception as e:
        pytest.fail(f"Main function failed with exception: {e}")
