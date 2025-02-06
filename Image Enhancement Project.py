!pip install opencv-python

import cv2
import numpy as np
import matplotlib.pyplot as plt
from google.colab import files

#uploaded = files.upload()

# Load a low-contrast grayscale image
image = cv2.imread('low contrast image.png', cv2.IMREAD_GRAYSCALE)

# 1. Histogram Equalization
hist_eq_image = cv2.equalizeHist(image)

# 2. CLAHE
# Create a CLAHE object with desired parameters
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
clahe_image = clahe.apply(image)
  
