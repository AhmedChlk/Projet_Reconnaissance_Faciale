import numpy as np
import os
import pytest
from src.features.extracteur import FeatureExtractor
from src.features.dataset_manager import DatasetManager

def test_feature_extraction_vector_size():
    extractor = FeatureExtractor(num_features=30)
    eye_centers = [(40, 50), (80, 50)]
    contour = np.array([(i, i) for i in range(50)]) # Dummy contour
    
    vector = extractor.extract_30d_vector(eye_centers, contour)
    assert vector.shape == (30,)
    assert isinstance(vector[0], float)

def test_dataset_save_and_load(tmp_path):
    # Use tmp_path for isolated testing
    csv_file = tmp_path / "test_dataset.csv"
    manager = DatasetManager(csv_path=str(csv_file))
    
    name = "TestPerson"
    vector = np.random.rand(30)
    
    manager.save_entry(name, vector)
    data = manager.load_dataset()
    
    assert len(data) == 1
    loaded_name, loaded_vector = data[0]
    assert loaded_name == name
    assert np.allclose(loaded_vector, vector)
