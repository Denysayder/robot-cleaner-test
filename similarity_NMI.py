# -*- coding: utf-8 -*-
"""
Utility for comparing two images (or video frames) and returning a single
similarity score based on Normalized Mutual Information (NMI), Structural
Similarity Index (SSIM) and root‑mean‑square error (RMSE).

Usage:
    score = compare_images(frame1, frame2)
"""

import cv2 as cv
import numpy as np
from skimage.metrics import structural_similarity as ssim
from scipy.stats import entropy
import skimage  # kept for backward compatibility / possible external references  # noqa: F401


def _normalize(img: np.ndarray) -> np.ndarray:
    """
    Normalize pixel values of an image to the range [0, 1].

    If the image is constant (all pixels identical), the original array is
    returned to avoid division‑by‑zero.
    """
    img = img.astype(np.float32)
    denom = img.max() - img.min()
    return img if denom == 0 else (img - img.min()) / denom


def compare_images(frame1: np.ndarray, frame2: np.ndarray) -> float:
    """
    Compare two images/frames and return a combined similarity score ∈ [0, 1].

    Parameters
    ----------
    frame1, frame2 : np.ndarray
        Images (BGR or grayscale).  If either is ``None`` the function
        returns 0.

    Returns
    -------
    float
        Combined similarity score.  Returns 0 if the score is NaN.
    """
    if frame1 is None or frame2 is None:
        return 0.0

    # Resize both images to the same size for fair comparison
    target_size = (300, 300)
    image1 = cv.resize(frame1, target_size)
    image2 = cv.resize(frame2, target_size)

    # Normalize to [0, 1]
    image1_norm = _normalize(image1)
    image2_norm = _normalize(image2)

    # Binarize with a fixed threshold at 0.5
    image1_bw = (image1_norm >= 0.5).astype(np.uint8)
    image2_bw = (image2_norm >= 0.5).astype(np.uint8)

    # Structural Similarity Index (SSIM)
    similarity = ssim(image1_bw, image2_bw, data_range=1.0)

    # Root‑mean‑square error (RMSE)
    mse = np.sqrt(np.mean((image1_bw - image2_bw) ** 2))

    # Histograms for Normalized Mutual Information (NMI)
    hist1, _ = np.histogram(image1_norm, bins=256, range=(0, 1))
    hist2, _ = np.histogram(image2_norm, bins=256, range=(0, 1))
    joint_hist, _, _ = np.histogram2d(
        image1_norm.ravel(),
        image2_norm.ravel(),
        bins=256,
        range=[[0, 1], [0, 1]],
    )

    joint_entropy = entropy(joint_hist.ravel())
    nmi = 0.0 if joint_entropy == 0 else (entropy(hist1) + entropy(hist2)) / joint_entropy

    # Combine metrics — all individual terms are in [0, 1]
    combined_score = (nmi * similarity * (1 - mse)) ** (1 / 3)

    return 0.0 if np.isnan(combined_score) else float(combined_score)


# --------------------------------------------------------------------------- #
# Example (commented out)
# --------------------------------------------------------------------------- #
# from skimage.io import imread
#
# img1 = imread("path/to/image1.jpg", as_gray=True)
# img2 = imread("path/to/image2.jpg", as_gray=True)
# score = compare_images(img1, img2)
# print("Similarity score:", score)
