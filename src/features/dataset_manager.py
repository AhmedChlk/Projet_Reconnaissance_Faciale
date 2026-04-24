import os
import numpy as np
from typing import List, Tuple, Optional

class DatasetManager:
    def __init__(self, csv_path: str = "data/dataset.csv"):
        self.csv_path = csv_path
        self._cache: Optional[List[Tuple[str, np.ndarray]]] = None
        # Ensure the directory exists
        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)

    def name_exists(self, name: str) -> bool:
        """Checks if a name already exists in the dataset."""
        dataset = self.load_dataset()
        for existing_name, _ in dataset:
            if existing_name.lower() == name.lower():
                return True
        return False

    def save_entry(self, name: str, vector: np.ndarray) -> None:
        """
        Saves a name and its 30D vector to the CSV.
        Format: name;v1;v2;...;v30
        """
        vector_str = ";".join(map(str, vector))
        with open(self.csv_path, "a") as f:
            f.write(f"{name};{vector_str}\n")
        # Invalidate cache
        self._cache = None

    def load_dataset(self) -> List[Tuple[str, np.ndarray]]:
        """
        Loads the entire dataset from the CSV.
        Returns a list of tuples (name, vector).
        """
        if self._cache is not None:
            return self._cache

        if not os.path.exists(self.csv_path):
            return []
            
        dataset = []
        try:
            with open(self.csv_path, "r") as f:
                for line in f:
                    parts = line.strip().split(";")
                    if len(parts) < 31: # Name + 30 features
                        continue
                    name = parts[0]
                    vector = np.array(list(map(float, parts[1:])), dtype=np.float64)
                    dataset.append((name, vector))
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return []

        self._cache = dataset
        return dataset
