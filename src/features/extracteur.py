import numpy as np

class FeatureExtractor:
    def __init__(self, num_features=30):
        self.num_features = num_features

    def extract_30d_vector(self, eye_centers, contour):
        """
        Generates a 30D vector from eye centers and facial contour.
        Normalizes distances by the inter-eye distance.
        """
        if eye_centers is None or len(eye_centers) < 2 or contour is None:
            return None

        # Calculate inter-eye distance for normalization
        e1, e2 = np.array(eye_centers[0]), np.array(eye_centers[1])
        inter_eye_dist = np.linalg.norm(e1 - e2)
        
        if inter_eye_dist == 0:
            return None

        # Calculate eye midpoint
        midpoint = (e1 + e2) / 2.0

        # Sample exactly num_features points from the contour
        indices = np.linspace(0, len(contour) - 1, self.num_features, dtype=int)
        sampled_points = contour[indices]

        # Calculate normalized distances from midpoint to contour points
        vector = []
        for p in sampled_points:
            dist = np.linalg.norm(p - midpoint)
            vector.append(dist / inter_eye_dist)

        return np.array(vector, dtype=float)
