import numpy as np
import cv2


def extract_spectrum(image):
    """
    Load an image from *image* and return its magnitude spectrum.

    Parameters
    ----------
    image : str
        Path to the image file.

    Returns
    -------
    np.ndarray
        Magnitude spectrum of the image.
    """
    load_image = cv2.imread(image)
    if load_image is None:
        raise FileNotFoundError(f"Не удалось открыть файл: {image}")

    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(load_image)))
    return spectrum


def extract_spectrum_on_frame(frame):
    """
    Compute the magnitude spectrum of a single (already‑loaded) video frame.

    Parameters
    ----------
    frame : np.ndarray
        Image/frame array.

    Returns
    -------
    np.ndarray
        Magnitude spectrum of the frame.
    """
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(frame)))
    return spectrum


def spectrum_to_see(picture):
    """
    Convert *picture* to an 8‑bit image of its log‑scaled magnitude spectrum,
    ready for on‑screen inspection.

    Parameters
    ----------
    picture : np.ndarray
        2‑D or 3‑D image array.

    Returns
    -------
    np.ndarray
        Normalised 8‑bit magnitude‑spectrum image.
    """
    spectrum = np.fft.fftshift(np.fft.fft2(picture))

    # Magnitude
    magnitude_spectrum = np.abs(spectrum)

    # Avoid log(0)
    small_value = 1e-10   # Small enough not to distort the spectrum
    magnitude_spectrum[magnitude_spectrum == 0] = small_value

    # Log scale
    magnitude_spectrum = 20 * np.log(magnitude_spectrum)
    magnitude_spectrum[np.isinf(magnitude_spectrum) | np.isnan(magnitude_spectrum)] = 0

    # Normalise to 0‑255 and convert to uint8
    magnitude_spectrum = cv2.normalize(
        magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U
    )

    return magnitude_spectrum


def compare_spectra(spectrum1, spectrum2):
    """
    Compare two spectra using the Pearson correlation coefficient.

    Parameters
    ----------
    spectrum1, spectrum2 : np.ndarray
        Spectra to compare (must have the same shape).

    Returns
    -------
    float
        Correlation coefficient in the range [-1, 1].
    """
    correlation = np.corrcoef(spectrum1.flatten(), spectrum2.flatten())[0, 1]
    return correlation


if __name__ == '__main__':
    # Replace with real paths before running
    image1 = 'path/to/first/image.jpg'
    image2 = 'path/to/second/image.jpg'

    spec1 = extract_spectrum(image1)
    spec2 = extract_spectrum(image2)

    corr = compare_spectra(spec1, spec2)
    print(f'Correlation: {corr:.4f}')
