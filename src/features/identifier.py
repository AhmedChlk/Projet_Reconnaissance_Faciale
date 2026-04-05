import numpy as np

class FaceIdentifier:
    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def compare(self, target_vector, dataset):
        """
        Compares target_vector against a list of (name, vector) from dataset.
        Returns (name, distance, is_match) for the best match.
        """
        if not dataset:
            return None, float('inf'), False

        best_match = None
        min_dist = float('inf')

        for name, vector in dataset:
            dist = np.linalg.norm(target_vector - vector)
            if dist < min_dist:
                min_dist = dist
                best_match = name

        is_match = min_dist <= self.threshold
        return best_match, min_dist, is_match
