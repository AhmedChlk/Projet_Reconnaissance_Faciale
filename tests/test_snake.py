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
    # Check if first point is (114, 64) approx (center_x + radius, center_y)
    assert np.allclose(points[0], [114, 64], atol=1e-5)

def test_snake_evolution():
    # Create a 128x128 image with a white circle on black background
    img = np.zeros((128, 128), dtype=np.uint8)
    cv2.circle(img, (64, 64), 40, 255, -1)
    
    # Use larger gamma for faster movement in test
    snake = ActiveContourSnake(alpha=0.01, beta=0.01, gamma=1.0)
    # Start with a larger circle (radius 50)
    initial_contour = snake.initialize_circle((64, 64), 50, num_points=40)
    
    # Evolve
    final_contour = snake.evolve(img, initial_contour, iterations=10)
    
    assert final_contour.shape == (40, 2)
    
    # Calculate average radius
    dist_initial = np.mean(np.linalg.norm(initial_contour - np.array([64, 64]), axis=1))
    dist_final = np.mean(np.linalg.norm(final_contour - np.array([64, 64]), axis=1))
    
    # The contour should shrink towards the circle in the image
    assert dist_final < dist_initial

def test_snake_external_force_flat_image():
    snake = ActiveContourSnake()
    img = np.zeros((100, 100), dtype=np.uint8)
    fx, fy = snake._get_external_force(img)
    # On a flat image, forces should be zero (or very close to it)
    assert np.allclose(fx, 0)
    assert np.allclose(fy, 0)

def test_snake_evolution_edge_cases():
    snake = ActiveContourSnake()
    img = np.zeros((100, 100), dtype=np.uint8)
    # Initial contour partially outside image
    initial_contour = np.array([[-10, -10], [50, 50], [110, 110]], dtype=float)
    # Should not crash due to np.clip
    final_contour = snake.evolve(img, initial_contour, iterations=1)
    assert final_contour.shape == (3, 2)
