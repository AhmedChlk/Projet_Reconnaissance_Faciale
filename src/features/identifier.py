import numpy as np

class FaceIdentifier:
    def __init__(self, threshold=0.65):
        self.threshold = threshold

    def compare(self, target_vector, dataset):
        """
        Compares target_vector against a list of (name, vector) from dataset.
        Returns (best_match, min_dist, is_match, confidence, top_3).
        """
        if not dataset:
            return None, float('inf'), False, 0.0, []

        # Calculate all distances
        results = []
        for name, vector in dataset:
            dist = np.linalg.norm(target_vector - vector)
            results.append((name, dist))
        
        # Sort by distance
        results.sort(key=lambda x: x[1])
        
        # Extract best match
        best_match, min_dist = results[0]
        is_match = min_dist <= self.threshold
        
        # Calculate confidence score
        confidence = max(0.0, 100.0 * (1.0 - (min_dist / self.threshold)))
        
        # Get top 3
        top_3 = results[:3]
        
        return best_match, min_dist, is_match, confidence, top_3
