
import cv2
import numpy as np
from skimage.feature import hog

# Konfigurasi HOG
IMAGE_SIZE = (64, 64)
ORIENTATIONS = 9
PIXELS_PER_CELL = (8, 8)
CELLS_PER_BLOCK = (2, 2)


def preprocess_image(image):
    """
    Mengubah citra BGR menjadi grayscale 64x64.
    """
    if image is None:
        raise ValueError("Citra tidak valid.")

    resized = cv2.resize(
        image,
        IMAGE_SIZE,
        interpolation=cv2.INTER_AREA
    )

    gray = cv2.cvtColor(
        resized,
        cv2.COLOR_BGR2GRAY
    )

    return gray


def extract_hog(gray_image, visualize=False):
    """
    Mengekstraksi fitur HOG dari citra grayscale.

    Jika visualize=True, mengembalikan fitur
    dan gambar visualisasi HOG.
    """
    result = hog(
        gray_image,
        orientations=ORIENTATIONS,
        pixels_per_cell=PIXELS_PER_CELL,
        cells_per_block=CELLS_PER_BLOCK,
        block_norm="L2-Hys",
        transform_sqrt=False,
        visualize=visualize,
        feature_vector=True
    )

    if visualize:
        features, hog_image = result
        return features.astype(np.float32), hog_image

    return result.astype(np.float32)
