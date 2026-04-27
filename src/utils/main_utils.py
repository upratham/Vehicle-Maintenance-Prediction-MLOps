import os

import numpy as np
import dill
import yaml


def read_yaml_file(file_path: str) -> dict:
    with open(file_path, "rb") as yaml_file:
        return yaml.safe_load(yaml_file)


def write_yaml_file(file_path: str, content: object, replace: bool = False) -> None:
    if replace and os.path.exists(file_path):
        os.remove(file_path)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as file:
        yaml.dump(content, file)


def load_object(file_path: str) -> object:
    """Load a dill-pickled object from disk."""
    with open(file_path, "rb") as file_obj:
        return dill.load(file_obj)


def save_numpy_array_data(file_path: str, array: np.array):
    """Save a numpy array to disk, creating parent directories as needed."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "wb") as file_obj:
        np.save(file_obj, array)


def load_numpy_array_data(file_path: str) -> np.array:
    with open(file_path, "rb") as file_obj:
        return np.load(file_obj)


def save_object(file_path: str, obj: object) -> None:
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "wb") as file_obj:
        dill.dump(obj, file_obj)
