import cv2
import numpy as np

# Load medical image (grayscale)
image = cv2.imread(r'c:\Users\TIJI JOJI\Desktop\tumor_scan.jpg', 0)
# Apply thresholding
_, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

# Find contours
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Compute convex hull for each contour
hull_list = [cv2.convexHull(cnt) for cnt in contours]

# Draw convex hull on the original image
image_hull = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
cv2.drawContours(image_hull, hull_list, -1, (0, 255, 0), 2)

# Show the result
cv2.imshow('Convex Hull Tumor Detection', image_hull)
cv2.waitKey(0)
cv2.destroyAllWindows()
