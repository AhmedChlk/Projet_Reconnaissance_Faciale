import os
import pandas as pd
import numpy as np

class DatasetManager:
    def __init__(self, csv_path="data/dataset.csv"):
        self.csv_path = csv_path
        # Ensure the directory exists
        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)

    def save_entry(self, name, vector):
        """
        Saves a name and its 30D vector to the CSV.
        Format: name;v1,v2,...,v30
        """
        vector_str = ",".join(map(str, vector))
        with open(self.csv_path, "a") as f:
            f.write(f"{name};{vector_str}\n")

    def load_dataset(self):
        """
        Loads the entire dataset from the CSV.
        Returns a list of tuples (name, vector).
        """
        if not os.path.exists(self.csv_path):
            return []
            
        data = []
        with open(self.csv_path, "r") as f:
            for line in f:
                if ";" not in line:
                    continue
                name, vector_str = line.strip().split(";")
                vector = np.array(list(map(float, vector_str.split(","))))
                data.append((name, vector))
        return data
