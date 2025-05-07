import cv2
from features import detector, descriptor, matcher
from utils import image_io

# Load images
img1 = cv2.imread("data/object1/view1.jpg", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("data/scene.jpg", cv2.IMREAD_GRAYSCALE)

# Detect keypoints manually
kps1 = detector.detect_keypoints(img1)
kps2 = detector.detect_keypoints(img2)

# Extract descriptors manually
desc1 = descriptor.compute_descriptors(img1, kps1)
desc2 = descriptor.compute_descriptors(img2, kps2)

# Match features manually
matches = matcher.match_features(desc1, desc2)

# Visualize results
image_io.draw_matches(img1, kps1, img2, kps2, matches)