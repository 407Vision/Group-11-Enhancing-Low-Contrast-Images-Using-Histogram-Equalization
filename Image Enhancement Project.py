!pip install OpenCV-python
import cv2
import numpy as np
import matplotlib.pyplot as plt
from google.colab import files

#uploaded = files.upload()

# Load a low-contrast grayscale image
image = cv2.imread('low contrast image.png', cv2.IMREAD_GRAYSCALE)

# Check if the image is loaded
if image is None:
    print("Error: Image not found or path is incorrect!")  # More informative error message
    exit()

# 1. Histogram Equalization
hist_eq_image = cv2.equalizeHist(image)
