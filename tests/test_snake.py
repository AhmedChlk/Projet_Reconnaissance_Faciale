import cv2
import numpy as np
import pytest
from src.contour.snake import ActiveContourSnake

def test_snake_initialization():
    snake = ActiveContourSnake()
    center = (64, 64)
    radius = 50
    points = snake.initialize_circle(center, radius, num_points=10)
    assert points.shape == (10, 2)
    # Check if first point is (114, 64) approx
    assert np.allclose(points[0], [114, 64], atol=1e-5)

def test_snake_evolution():
    # Create a 128x128 image with a white circle
    img = np.zeros((128, 128), dtype=np.uint8)
    cv2.circle(img, (64, 64), 40, 255, -1)
    
    snake = ActiveContourSnake(alpha=0.1, beta=0.1, gamma=0.1)
    # Start with a larger circle
    initial_contour = snake.initialize_circle((64, 64), 50, num_points=20)
    
    # Evolve
    final_contour = snake.evolve(img, initial_contour, iterations=20)
    
    assert final_contour.shape == (20, 2)
    # The final contour should be closer to radius 40 than initial 50
    dist_initial = np.mean(np.sqrt(np.sum((initial_contour - [64, 64])**2, axis=1)))
    dist_final = np.mean(np.sqrt(np.sum((final_contour - [64, 64])**2, axis=1)))
    
    assert dist_final < dist_initial
    assert dist_final > 35 # Should stay near 40
