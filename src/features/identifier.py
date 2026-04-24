import numpy as np
from typing import List, Tuple, Optional

class FaceIdentifier:
    def __init__(self, threshold: float = 0.25): # Adjusted for Cosine Distance
        self.threshold = threshold

    def _cosine_distance(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Calculates cosine distance (1 - cosine similarity). Range [0, 2]."""
        dot = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        if norm_v1 == 0 or norm_v2 == 0:
            return 2.0
        similarity = dot / (norm_v1 * norm_v2)
        # Clip to avoid precision issues
        return 1.0 - np.clip(similarity, -1.0, 1.0)

    def compare(self, target_vector: np.ndarray, dataset: List[Tuple[str, np.ndarray]]) -> Tuple[Optional[str], float, bool, List[Tuple[str, float, float]]]:
        """
        Compares target_vector using Cosine Distance.
        Aggregates results by person (takes the minimum distance per name).
        Returns (best_name, best_dist, is_match, top_3_list).
        """
        if not dataset:
            return None, 1.0, False, []

        # Dictionary to store the minimum distance found for each person
        # name -> min_dist
        person_min_dists = {}

        for name, vector in dataset:
            dist = self._cosine_distance(target_vector, vector)
            if name not in person_min_dists or dist < person_min_dists[name]:
                person_min_dists[name] = dist

        # Convert to list of results and calculate confidence
        results = []
        for name, min_dist in person_min_dists.items():
            confidence = max(0.0, 100.0 * (1.0 - min_dist / self.threshold))
            results.append((name, float(min_dist), float(confidence)))

        # Sort by distance (ascending)
        results.sort(key=lambda x: x[1])
        
        top_3 = results[:3]
        
        if not results:
            return None, 1.0, False, []
            
        best_name, min_dist, _ = results[0]
        is_match = min_dist <= self.threshold
        
        return best_name, min_dist, is_match, top_3
