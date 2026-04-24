import numpy as np
import pytest
import os
from unittest.mock import patch, mock_open
from src.features.extracteur import FeatureExtractor
from src.features.dataset_manager import DatasetManager
from src.features.identifier import FaceIdentifier

def test_feature_extractor_success():
    extractor = FeatureExtractor(num_features=30)
    eye_centers = [(40, 50), (80, 50)]
    contour = np.array([(i, i) for i in range(100)], dtype=float)
    
    vector = extractor.extract_30d_vector(eye_centers, contour)
    assert vector.shape == (30,)
    assert vector[0] > 0

def test_feature_extractor_none_cases():
    extractor = FeatureExtractor()
    assert extractor.extract_30d_vector(None, None) is None
    assert extractor.extract_30d_vector([(0,0)], None) is None
    assert extractor.extract_30d_vector([(0,0), (0,0)], np.array([])) is None
    # Test inter-eye distance 0
    assert extractor.extract_30d_vector([(0,0), (0,0)], np.array([(1,1)])) is None

def test_dataset_manager_flow(tmp_path):
    csv_file = tmp_path / "subdir" / "test_data.csv"
    manager = DatasetManager(str(csv_file))
    
    # Check directory creation
    assert os.path.exists(os.path.dirname(str(csv_file)))
    
    # Empty load
    assert manager.load_dataset() == []
    
    # Save
    v1 = np.ones(30)
    manager.save_entry("Ahmed", v1)
    
    # Load (first time fills cache)
    data = manager.load_dataset()
    assert len(data) == 1
    assert data[0][0] == "Ahmed"
    
    # Second load (uses cache)
    data2 = manager.load_dataset()
    assert data2 is data

def test_dataset_manager_malformed(tmp_path):
    csv_file = tmp_path / "malformed.csv"
    with open(csv_file, "w") as f:
        f.write("wrong;line\n") # Too short
        f.write("Good;0;1;2;3;4;5;6;7;8;9;10;11;12;13;14;15;16;17;18;19;20;21;22;23;24;25;26;27;28;29\n")
    
    manager = DatasetManager(str(csv_file))
    data = manager.load_dataset()
    assert len(data) == 1
    assert data[0][0] == "Good"

def test_dataset_manager_exception():
    # Use a path that has a directory to avoid os.makedirs("") error
    with patch("builtins.open", mock_open()) as mocked_file:
        manager = DatasetManager("data/dummy.csv")
        mocked_file.side_effect = Exception("Read error")
        # To avoid exists check failure, we need to mock exists too
        with patch("os.path.exists", return_value=True):
            data = manager.load_dataset()
            assert data == []

def test_face_identifier_logic():
    identifier = FaceIdentifier(threshold=1.0)
    
    dataset = [
        ("A", np.zeros(30)),
        ("B", np.ones(30) * 0.1),
        ("C", np.ones(30) * 0.5),
        ("D", np.ones(30) * 2.0)
    ]
    
    target = np.zeros(30)
    name, dist, is_match, top_3 = identifier.compare(target, dataset)
    
    assert name == "A"
    assert is_match is True
    assert len(top_3) == 3
    assert top_3[0][0] == "A"
    
    # Check match false
    target_far = np.ones(30) * 5.0
    name_f, dist_f, is_match_f, _ = identifier.compare(target_far, dataset)
    assert is_match_f is False

def test_face_identifier_empty_dataset():
    identifier = FaceIdentifier()
    name, dist, is_match, top_3 = identifier.compare(np.zeros(30), [])
    assert name is None
    assert dist == float('inf')
    assert top_3 == []
