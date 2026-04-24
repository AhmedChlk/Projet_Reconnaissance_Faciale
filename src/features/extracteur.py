import numpy as np
from typing import Optional

class FeatureExtractor:
    def __init__(self, num_features: int = 30):
        self.num_features = num_features

    def extract_30d_vector(self, eye_centers: Optional[np.ndarray], contour: np.ndarray) -> Optional[np.ndarray]:
        """
        Generates a 30D vector from the facial contour.
        Uses the contour centroid as the reference point for high stability.
        Normalizes distances by the inter-eye distance (fallback to contour width if eyes missing).
        """
        if contour is None or len(contour) == 0:
            return None

        # 1. Calculate normalization factor (Scale invariance)
        if eye_centers is not None and len(eye_centers) >= 2:
            e1, e2 = np.array(eye_centers[0]), np.array(eye_centers[1])
            norm_factor = np.linalg.norm(e1 - e2)
        else:
            # Fallback: use the bounding box width of the contour
            x_min, x_max = np.min(contour[:, 0]), np.max(contour[:, 0])
            norm_factor = (x_max - x_min) * 0.4 # Empirical ratio

        if norm_factor == 0:
            return None

        # 2. Calculate stable reference point: Centroid of the contour
        centroid = np.mean(contour, axis=0)

        # 3. Sample exactly num_features points from the contour
        # Ensure we always sample the same points relative to the starting angle
        indices = np.linspace(0, len(contour) - 1, self.num_features, dtype=int)
        sampled_points = contour[indices]

        # 4. Calculate normalized distances from centroid to contour points
        vector = []
        for p in sampled_points:
            dist = np.linalg.norm(p - centroid)
            vector.append(dist / norm_factor)

        return np.array(vector, dtype=np.float64)
