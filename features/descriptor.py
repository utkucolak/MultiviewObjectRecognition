import numpy as np

from features.gradients import Gradients
from typing import Tuple, List
class Descriptor():
    def __init__(self, keypoints: List[Tuple[float, float]], image: np.ndarray):
        self.keypoints = keypoints
        self.image = image
        self.descriptors = []
    def compute_descriptors(self, patch_size=8):
        half = patch_size // 2
        Ix, Iy = Gradients().compute_gradients(self.image)

        magnitude = np.sqrt(Ix**2 + Iy**2)
        orientation = np.rad2deg(np.arctan2(Iy, Ix)) % 360
        for x,y in self.keypoints:
            x, y = int(x), int(y)
            if x - half < 0 or x + half >= self.image.shape[1] or y - half < 0 or y + half >= self.image.shape[0]:
                continue
            patch_mag = magnitude[y - half:y + half, x - half:x + half]
            patch_ori = orientation[y - half:y + half, x - half:x + half]

            descriptor = []
            cell_size = patch_size // 4

            for i in range(4):
                for j in range(4):
                    cell_mag = patch_mag[i * cell_size:(i + 1) * cell_size, j * cell_size:(j + 1) * cell_size]
                    cell_ori = patch_ori[i * cell_size:(i + 1) * cell_size, j * cell_size:(j + 1) * cell_size]

                    hist, _ = np.histogram(cell_ori, bins=8, range=(0, 360), weights=cell_mag)
                    descriptor.extend(hist)
            descriptor = np.array(descriptor, dtype=np.float32)
            norm = np.linalg.norm(descriptor)
            if norm > 1e-5:
                descriptor /= norm
            self.descriptors.append(descriptor)