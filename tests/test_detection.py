import cv2
import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from src.detection.alignement import FaceAligner

@pytest.fixture
def aligner():
    with patch('cv2.CascadeClassifier') as mock_cascade:
        # Mock classifiers to not fail on __init__
        yield FaceAligner()

def test_aligner_none_image(aligner):
    result, eyes = aligner.align(None)
    assert result is None
    assert eyes is None

def test_detect_largest_face_none(aligner):
    # Mock face_cascade to return no faces
    aligner.face_cascade.detectMultiScale = MagicMock(return_value=())
    img = np.zeros((100, 100), dtype=np.uint8)
    res = aligner.detect_largest_face(img)
    assert res is None

def test_detect_largest_face_success(aligner):
    # Mock face_cascade to return two faces, one larger
    # (x, y, w, h)
    faces = np.array([[0, 0, 10, 10], [10, 10, 20, 20]])
    aligner.face_cascade.detectMultiScale = MagicMock(return_value=faces)
    img = np.zeros((100, 100), dtype=np.uint8)
    res = aligner.detect_largest_face(img)
    assert np.array_equal(res, [10, 10, 20, 20])

def test_get_eye_centers_fail(aligner):
    aligner.eye_cascade.detectMultiScale = MagicMock(return_value=[[0, 0, 5, 5]]) # Only 1 eye
    res = aligner.get_eye_centers(np.zeros((50, 50), dtype=np.uint8))
    assert res is None

def test_get_eye_centers_success(aligner):
    # Return 3 eyes, should pick 2 largest
    eyes = np.array([[0, 0, 10, 10], [20, 0, 10, 10], [5, 5, 2, 2]])
    aligner.eye_cascade.detectMultiScale = MagicMock(return_value=eyes)
    res = aligner.get_eye_centers(np.zeros((50, 50), dtype=np.uint8))
    assert len(res) == 2
    # Midpoints of [0,0,10,10] and [20,0,10,10] are (5,5) and (25,5)
    assert res[0] == (5, 5)
    assert res[1] == (25, 5)

def test_align_full_flow(aligner):
    # Mock face and eye detection to simulate full flow
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    # Face at (50, 50, 100, 100)
    aligner.detect_largest_face = MagicMock(return_value=(50, 50, 100, 100))
    # Eyes at (20, 30) and (80, 40) relative to face
    aligner.get_eye_centers = MagicMock(side_effect=[
        [(20, 30), (80, 40)], # First call for alignment
        [(30, 40), (70, 40)]  # Second call after normalization
    ])
    
    final_face, eye_centers = aligner.align(img)
    
    assert final_face.shape == (128, 128)
    assert eye_centers == [(30, 40), (70, 40)]
    assert aligner.get_eye_centers.call_count == 2

def test_align_no_face(aligner):
    aligner.detect_largest_face = MagicMock(return_value=None)
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    res, eyes = aligner.align(img)
    assert res is None
