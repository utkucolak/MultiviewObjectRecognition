from scipy.signal import convolve2d
import numpy as np
from scipy.ndimage import gaussian_filter
import cv2

from features.gradients import Gradients
from typing import Tuple

class Detector():
    def __init__(self):
        self.keypoints = np.zeros(0)
        self.R = np.zeros(0)
        self.k = 0.05
        self.threshold = 1e4

    def detect_keypoints(self, image) -> Tuple[list[tuple[float, float]], np.ndarray]:
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        Ix, Iy = Gradients().compute_gradients(image)

        Ixx = Ix ** 2
        Ixy = Ix * Iy
        Iyy = Iy ** 2

        Ixx = gaussian_filter(Ixx, sigma=1)
        Ixy = gaussian_filter(Ixy, sigma=1)
        Iyy = gaussian_filter(Iyy, sigma=1)

        #M = np.array([
        #    [Ixx, Ixy],
        #    [Ixy, Iyy]
        #])

        #Harris response formula. We use it since computing eigenvalues 
        #directly is expensive.
        #R = det(M) - k. (trace(M))^2

        det_M = Ixx*Iyy - Ixy ** 2
        tr_M = Ixx + Iyy
        self.R = det_M - self.k * ((tr_M) ** 2)
        self.keypoints = [(float(x), float(y)) for y, x in np.argwhere(self.R > self.threshold)]

        return self.keypoints, self.R

