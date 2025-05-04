import cv2 as cv
import numpy as np


def process_image_fourier(input_file: str) -> str:
    """
    Compute the Fourier‑magnitude spectrum of a single image and save it.

    Parameters
    ----------
    input_file : str
        Path to the input image.

    Returns
    -------
    str
        Path to the saved spectrum image.
    """
    output_file = "image/fourier_result.jpg"

    # Load the image in grayscale
    img = cv.imread(input_file, cv.IMREAD_GRAYSCALE)
    assert img is not None, f"Image file could not be read: {input_file}"

    # Fourier transform
    fshift = np.fft.fftshift(np.fft.fft2(img))
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-8)  # avoid log(0)

    # Normalize and save
    magnitude_spectrum_uint8 = cv.normalize(
        magnitude_spectrum, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U
    )
    cv.imwrite(output_file, magnitude_spectrum_uint8)

    return output_file


def process_video_fourier(input_file: str) -> str:
    """
    Apply Fourier‑magnitude conversion to every frame of a video.

    Parameters
    ----------
    input_file : str
        Path to the input video.

    Returns
    -------
    str
        Path to the saved processed video.
    """
    output_file = "video/fourier_result.mp4"

    cap = cv.VideoCapture(input_file)
    assert cap.isOpened(), f"Video file could not be opened: {input_file}"

    fps = cap.get(cv.CAP_PROP_FPS)
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv.VideoWriter_fourcc(*"mp4v")
    out = cv.VideoWriter(output_file, fourcc, fps, (width, height), isColor=False)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        fshift = np.fft.fftshift(np.fft.fft2(gray))
        magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-8)

        magnitude_spectrum_uint8 = cv.normalize(
            magnitude_spectrum, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U
        )
        out.write(magnitude_spectrum_uint8)

    cap.release()
    out.release()

    return output_file


def perform_fourier_transform_and_compare_to_reference_fourier_image(
    frame: np.ndarray, reference_image: np.ndarray
) -> tuple[np.ndarray, float]:
    """
    Compute a frame’s Fourier‑magnitude spectrum, compare it to a reference,
    and return the spectrum (as BGR) together with the percentage difference.

    Parameters
    ----------
    frame : np.ndarray
        Input frame in BGR format.
    reference_image : np.ndarray
        Grayscale reference magnitude‑spectrum image.

    Returns
    -------
    tuple[np.ndarray, float]
        (BGR visualisation of the spectrum, percentage difference 0–100).
    """
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    dft_shift = np.fft.fftshift(
        cv.dft(np.float32(gray), flags=cv.DFT_COMPLEX_OUTPUT)
    )
    magnitude_spectrum = 20 * np.log(
        cv.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]) + 1e-8
    )

    magnitude_spectrum_norm = cv.normalize(
        magnitude_spectrum, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U
    )

    diff = cv.absdiff(reference_image, magnitude_spectrum_norm)
    diff_percentage = float(np.mean(diff)) / 255.0 * 100.0

    result_bgr = cv.cvtColor(magnitude_spectrum_norm, cv.COLOR_GRAY2BGR)
    return result_bgr, diff_percentage
