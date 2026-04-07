import numpy as np

class FeatureExtractor:
    def __init__(self, num_features=30):
        self.num_features = num_features

    def extract_30d_vector(self, eye_centers, contour):
        """
        Generates a 30D radial signature vector.
        1. Calculate contour centroid.
        2. Sample 30 points using fixed angles from the centroid.
        3. Normalize distances from eye midpoint by inter-eye distance.
        """
        if eye_centers is None or len(eye_centers) < 2 or contour is None:
            return None

        e1, e2 = np.array(eye_centers[0]), np.array(eye_centers[1])
        inter_eye_dist = np.linalg.norm(e1 - e2)
        if inter_eye_dist == 0: return None
        midpoint = (e1 + e2) / 2.0

        # Calculate centroid of the contour
        centroid = np.mean(contour, axis=0)
        
        vector = []
        angles = np.linspace(0, 2*np.pi, self.num_features, endpoint=False)
        
        for angle in angles:
            # Direction vector for this angle
            dir_vec = np.array([np.cos(angle), np.sin(angle)])
            
            # Find the point on the contour that is closest to this radial direction
            # We look for the point P such that the vector (P - centroid) 
            # has the smallest angle with dir_vec.
            
            # Vectors from centroid to all contour points
            vectors = contour - centroid
            magnitudes = np.linalg.norm(vectors, axis=1) + 1e-8
            unit_vectors = vectors / magnitudes[:, np.newaxis]
            
            # Dot product gives cos(theta)
            dots = np.dot(unit_vectors, dir_vec)
            best_idx = np.argmax(dots)
            best_point = contour[best_idx]
            
            # Calculate normalized distance from midpoint to this point
            dist = np.linalg.norm(best_point - midpoint)
            vector.append(dist / inter_eye_dist)

        return np.array(vector, dtype=float)
