from google.colab import files
import matplotlib.pyplot as plt
import numpy as np
import cv2
!pip install opencv-python


# uploaded = files.upload()

# Load a low-contrast grayscale image
image = cv2.imread('low contrast image.png' cv2.IMREAD_GRAYSCALE)
