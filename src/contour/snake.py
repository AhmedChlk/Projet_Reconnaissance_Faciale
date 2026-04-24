import cv2
import numpy as np
from typing import Tuple

class ActiveContourSnake:
    def __init__(self, alpha: float = 0.1, beta: float = 0.1, gamma: float = 0.01):
        """
        Initialize Snake parameters.
        alpha: Elasticity (continuity)
        beta: Rigidity (curvature)
        gamma: Step size for iteration (viscosity)
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def initialize_circle(self, center: Tuple[int, int], radius: float, num_points: int = 60) -> np.ndarray:
        """Creates an initial circular contour."""
        t = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
        x = center[0] + radius * np.cos(t)
        y = center[1] + radius * np.sin(t)
        return np.array([x, y], dtype=np.float64).T

    def _get_external_force(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calculates the external force (gradient of edge energy)."""
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        
        # Gradient magnitude using Sobel
        dx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        dy = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
        
        mag = np.sqrt(dx**2 + dy**2)
        # Normalize to [0, 1]
        mag_min, mag_max = mag.min(), mag.max()
        if mag_max > mag_min:
            mag = (mag - mag_min) / (mag_max - mag_min)
            
        # External energy: -magnitude (edges are minima)
        # Force: -grad(Energy) = grad(magnitude)
        fy, fx = np.gradient(mag)
        return fx, fy

    def evolve(self, image: np.ndarray, initial_contour: np.ndarray, iterations: int = 100) -> np.ndarray:
        """
        Evolves the contour using an implicit Euler method for stability.
        Matrix form: (I + gamma * A) * x_{t+1} = x_t + gamma * f_ext(x_t)
        """
        n = len(initial_contour)
        x = initial_contour[:, 0]
        y = initial_contour[:, 1]
        
        fx_map, fy_map = self._get_external_force(image)
        
        # Build internal energy matrix A (Pentadiagonal)
        # A = alpha * A_continuity + beta * A_rigidity
        a = self.beta
        b = -self.alpha - 4 * self.beta
        c = 2 * self.alpha + 6 * self.beta
        
        # Construct the matrix A using circulation
        row = np.zeros(n)
        row[0], row[1], row[2], row[-1], row[-2] = c, b, a, b, a
        A = np.zeros((n, n))
        for i in range(n):
            A[i] = np.roll(row, i)
            
        # Invert matrix for implicit step: (I + gamma * A)^-1
        inv = np.linalg.inv(np.eye(n) + self.gamma * A)
        
        for _ in range(iterations):
            # Bilinear interpolation of external forces at contour points
            indices_x = np.clip(x, 0, image.shape[1] - 1).astype(int)
            indices_y = np.clip(y, 0, image.shape[0] - 1).astype(int)
            
            fx = fx_map[indices_y, indices_x]
            fy = fy_map[indices_y, indices_x]
            
            # Update x and y coordinates
            x = np.dot(inv, x + self.gamma * fx)
            y = np.dot(inv, y + self.gamma * fy)
            
        return np.stack([x, y], axis=1)
