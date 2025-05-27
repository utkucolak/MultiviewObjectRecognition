import numpy as np
from typing import List, Tuple

class Matcher:
    def __init__(self, ratio_thresh: float = 0.75):
        self.ratio_thresh = ratio_thresh
        self.matches: List[Tuple[int, int]] = []

    def match_descriptors(self, desc1: List[np.ndarray], desc2: List[np.ndarray]) -> List[Tuple[int, int]]:
        """
        Performs matching using ratio distance: ||f1 - f2|| / ||f1 - f2'|| < threshold

        Args:
            desc1: Descriptors from image 1 (list of 128D numpy arrays)
            desc2: Descriptors from image 2

        Returns:
            List of matched pairs: (index_in_desc1, index_in_desc2)
        """
        self.matches = []

        for i, f1 in enumerate(desc1):
            if f1 is None or len(f1) == 0:
                continue

            best_dist = float("inf")
            second_best_dist = float("inf")
            best_j = -1

            for j, f2 in enumerate(desc2):
                if f2 is None or len(f2) == 0:
                    continue

                dist = np.linalg.norm(f1 - f2)

                if dist < best_dist:
                    second_best_dist = best_dist
                    best_dist = dist
                    best_j = j
                elif dist < second_best_dist:
                    second_best_dist = dist

            if second_best_dist == 0:
                continue  # Avoid division by zero

            ratio = best_dist / second_best_dist

            if ratio < self.ratio_thresh:
                self.matches.append((i, best_j))

        return self.matches