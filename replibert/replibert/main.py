from pathlib import Path

import yaml

from training import train


def main():
    print("Running main.py")

    # Load configuration
    with open(Path(__file__).parent / "config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Run the desired function
    train(config)

    print("Finished running main.py")


if __name__ == "__main__":
    main()
