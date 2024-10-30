from data.download import download_datasets
from training import train


def main():
    download_datasets()
    train()


if __name__ == "__main__":
    main()
