"""
brightness_thresholding.py

Provides two helper functions:

    perform_brightness_thresholding(frame, brightness_threshold)
    perform_brightness_thresholding_on_image(image_file, brightness_threshold)

The public API (function names, arguments, and return values) is unchanged,
so existing imports in the rest of your project will keep working.
"""

import cv2
import numpy as np
from pathlib import Path


def perform_brightness_thresholding(frame, brightness_threshold):
    """
    Zero‑out pixels brighter than `brightness_threshold` in the V channel
    of the HSV colour space.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]
    v = np.where(v > brightness_threshold, 0, v)
    hsv[:, :, 2] = v
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def perform_brightness_thresholding_on_image(image_file, brightness_threshold):
    """
    Apply brightness thresholding to an image on disk and save the result.
    Returns the path to the processed image.
    """
    output_file = Path("image") / "brightness_threshold_image.jpg"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(str(image_file))
    if image is None:
        raise FileNotFoundError(f"Cannot read image file: {image_file}")

    processed_image = perform_brightness_thresholding(image, brightness_threshold)
    cv2.imwrite(str(output_file), processed_image)
    return str(output_file)


# --------------------------------------------------------------------------- #
# Example usage (runs only when the file is executed directly):
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    INPUT_FILE = "input_frame.jpg"   # ← replace with your own file
    THRESHOLD_VALUE = 200            # ← adjust as required

    src = cv2.imread(INPUT_FILE)
    if src is None:
        raise FileNotFoundError(f"Cannot read image file: {INPUT_FILE}")

    dst = perform_brightness_thresholding(src, THRESHOLD_VALUE)

    cv2.imshow("Result Frame", dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
