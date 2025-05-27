import numpy as np
from scipy.signal import convolve2d

# Sobel kernels
class Gradients():
    def __init__(self):
        self.P_x = np.array([
            [-1, 0, 1],
            [-1, 0, 1],
            [-1, 0, 1]
        ], dtype=int)

        self.P_y = np.array([
            [-1, -1, -1],
            [0, 0, 0],
            [1, 1, 1]
        ], dtype=int)


    def compute_gradients(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute gradient in x and y directions using Sobel-like kernels.

        Returns:
            Ix: Gradient in x direction
            Iy: Gradient in y direction
        """
        if len(image.shape) == 3:
            import cv2
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        Ix = convolve2d(image, self.P_x, mode="same", boundary="symm")
        Iy = convolve2d(image, self.P_y, mode="same", boundary="symm")

        return Ix, Iy