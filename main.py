from pathlib import Path
from setup_and_baseline import setup_and_baseline, get_data





if __name__ == '__main__':
    cifar_folder_path = Path(r"C:\Users\ozgra\PycharmProjects\Foundations_of_DL\cifar-10-batches-py")
    # Part 1 - Setup and Baseline
    setup_and_baseline(*get_data(cifar_folder_path))

