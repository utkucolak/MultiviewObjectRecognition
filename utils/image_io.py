import cv2
import numpy as np

class ImageIO():
    def __init__(self):
        self.image = None

    def draw_matches(self, img1, kps1, img2, kps2, matches, radius=4):
        """
        Draw matches between two images as a combined side-by-side image.

        Args:
            img1, img2: Input images (BGR or grayscale)
            kps1, kps2: Keypoints in each image [(x, y), ...]
            matches: List of (i, j) index pairs
            radius: Radius for keypoint circle

        Returns:
            combined: Image with lines connecting matched keypoints
        """
        # Convert to color if grayscale
        if len(img1.shape) == 2:
            img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
        if len(img2.shape) == 2:
            img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

        # Combine images side-by-side
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        height = max(h1, h2)
        combined = np.zeros((height, w1 + w2, 3), dtype=np.uint8)
        combined[:h1, :w1] = img1
        combined[:h2, w1:] = img2

        # Draw matches
        for i, j in matches:
            x1, y1 = map(int, kps1[i])
            x2, y2 = map(int, kps2[j])
            x2_shifted = x2 + w1  # adjust x-coordinate for second image

            # Circles
            cv2.circle(combined, (x1, y1), radius, (0, 255, 0), -1)
            cv2.circle(combined, (x2_shifted, y2), radius, (0, 0, 255), -1)

            # Line connecting them
            cv2.line(combined, (x1, y1), (x2_shifted, y2), (255, 0, 0), 1)

        return combined
    
    def save_image(self, image: np.ndarray, filename: str = "matches_result.png"):
        """
        Saves the given image to disk.

        Args:
            image: The image to save
            filename: Output filename (default "matches_result.png")
        """
        cv2.imwrite(filename, image)
        print(f"[INFO] Match image saved as '{filename}'")