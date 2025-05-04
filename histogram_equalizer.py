import cv2


def histogram_equalization(image_path: str) -> str:
    """
    Equalize the histogram of the image located at `image_path`
    and save the result to ``image/histogram_equalized_result.jpg``.

    Parameters
    ----------
    image_path : str
        Path to the input image (BGR).

    Returns
    -------
    str
        Path to the saved equalized image.
    """
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Unable to read image file: {image_path}")

    # Output path
    output_image_path = "image/histogram_equalized_result.jpg"

    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Histogram equalization
    equalized_image = cv2.equalizeHist(gray_image)

    # Save the result
    cv2.imwrite(output_image_path, equalized_image)

    return output_image_path


def histogram_equalization_on_frame(frame):
    """
    Equalize the histogram of a single video frame (BGR).

    Parameters
    ----------
    frame : numpy.ndarray
        Input video frame in BGR format.

    Returns
    -------
    numpy.ndarray
        Equalized grayscale frame.
    """
    # Convert to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Histogram equalization
    equalized_frame = cv2.equalizeHist(gray_frame)

    return equalized_frame
