import cv2
import numpy as np


def adjust_brightness_on_frame(frame: np.ndarray, target_brightness: float) -> np.ndarray:
    """
    Adjust the brightness of a single frame so that its average brightness
    approaches ``target_brightness``.

    Parameters
    ----------
    frame : np.ndarray
        The input BGR frame.
    target_brightness : float
        Desired average brightness value (0–255).

    Returns
    -------
    np.ndarray
        Brightness‑adjusted frame.
    """
    # Current average brightness
    avg_brightness = np.mean(frame)

    # Avoid division by zero
    if avg_brightness == 0:
        return frame.copy()

    # Adjustment ratio = desired / current
    adjustment_ratio = target_brightness / avg_brightness

    # Scale pixel intensities, keeping data type unchanged
    adjusted = cv2.convertScaleAbs(frame, alpha=adjustment_ratio)

    # Clip maximum per‑pixel intensity to 150 (optional safety limit)
    # Comment out the following line if you need the full [0,255] range.
    adjusted = np.clip(adjusted, 0, 150)

    return adjusted


def adjust_brightness_on_image(image_file: str, target_brightness: float) -> str:
    """
    Adjust the brightness of a static image and save the result.

    Parameters
    ----------
    image_file : str
        Path to the source image.
    target_brightness : float
        Desired average brightness value (0–255).

    Returns
    -------
    str
        Path to the saved adjusted image.
    """
    output_file = "image/brightness_adjusted_image.jpg"
    image = cv2.imread(image_file)

    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_file}")

    adjusted = adjust_brightness_on_frame(image, target_brightness)
    cv2.imwrite(output_file, adjusted)

    return output_file


def main() -> None:
    """
    Real‑time webcam demo showing original and brightness‑adjusted frames.
    Press 'q' to quit.
    """
    target_brightness = 100.0

    # Open default webcam (index 0)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Không thể mở webcam.")  # "Cannot open webcam."
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Không thể đọc khung hình.")  # "Cannot read frame."
            break

        adjusted_frame = adjust_brightness_on_frame(frame, target_brightness)

        # Calculate average brightness values
        avg_original = np.mean(frame)
        avg_adjusted = np.mean(adjusted_frame)

        # Display averages on the frames
        cv2.putText(
            frame,
            f"Original: {avg_original:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            adjusted_frame,
            f"Adjusted: {avg_adjusted:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        # Show frames
        cv2.imshow("Original", frame)
        cv2.imshow("Adjusted", adjusted_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
