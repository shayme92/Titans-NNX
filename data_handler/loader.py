import numpy as np


def load_dataset(data_path: str) -> np.ndarray:
    data = np.memmap(data_path, dtype=np.uint16, mode="r")
    return data
