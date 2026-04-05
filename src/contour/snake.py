import cv2
import numpy as np

class ActiveContourSnake:
    def __init__(self, alpha=0.1, beta=0.1, gamma=0.01):
        """
        Initialize Snake parameters.
        alpha: Elasticity (continuity)
        beta: Rigidity (curvature)
        gamma: Step size for iteration
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def initialize_circle(self, center, radius, num_points=50):
        """Creates an initial circular contour."""
        t = np.linspace(0, 2*np.pi, num_points, endpoint=False)
        x = center[0] + radius * np.cos(t)
        y = center[1] + radius * np.sin(t)
        return np.array([x, y]).T

    def get_external_energy(self, image):
        """Calculates the external energy based on image gradients."""
        # Smoothing to reduce noise
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        
        # Gradient magnitude using Sobel
        dx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        dy = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
        
        mag = np.sqrt(dx**2 + dy**2)
        # Normalize energy to [0, 1] and invert so gradients are minima
        mag = (mag - mag.min()) / (mag.max() - mag.min() + 1e-8)
        return -mag

    def evolve(self, image, initial_contour, iterations=100):
        """Evolves the contour towards the image edges."""
        contour = initial_contour.copy()
        n = len(contour)
        
        # Precompute gradient of external energy
        ext_energy = self.get_external_energy(image)
        ext_grad_y, ext_grad_x = np.gradient(ext_energy)
        
        # Matrix for internal energy (Pentadiagonal)
        # This is a simplified version of the implicit Euler method matrix
        # For small steps, we can use an explicit update for simplicity here
        for _ in range(iterations):
            new_contour = contour.copy()
            for i in range(n):
                # Neighbors
                prev = contour[(i - 1) % n]
                curr = contour[i]
                nxt = contour[(i + 1) % n]
                prev2 = contour[(i - 2) % n]
                nxt2 = contour[(i + 2) % n]
                
                # Internal forces (Continuity + Curvature)
                f_int = self.alpha * (prev + nxt - 2 * curr) + \
                        self.beta * (prev2 + nxt2 - 4 * (prev + nxt) + 6 * curr)
                
                # External forces (Image Gradient)
                x_idx = int(np.clip(curr[0], 0, image.shape[1] - 1))
                y_idx = int(np.clip(curr[1], 0, image.shape[0] - 1))
                
                f_ext = np.array([ext_grad_x[y_idx, x_idx], ext_grad_y[y_idx, x_idx]])
                
                # Update point
                new_contour[i] = curr + self.gamma * (f_int - f_ext)
            
            contour = new_contour
            
        return contour
