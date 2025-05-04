import cv2 as cv
import numpy as np


def process_video_perspective(input_file: str) -> str:
    """
    Apply a fixed perspective transform to every frame of a video.
    Returns the path to the written result.
    """
    output_file = "video/view_result.mp4"
    multiplier_width = 0.246
    multiplier_height = 0.1875

    # Open the source video
    cap = cv.VideoCapture(input_file)
    assert cap.isOpened(), f"Cannot open video file: {input_file}"

    # Retrieve FPS and frame size
    fps = cap.get(cv.CAP_PROP_FPS)
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    # Transformation matrix (private to this function)
    def _get_transform_matrix() -> np.ndarray:
        src_pts = np.float32(
            [
                [width * multiplier_width, height * multiplier_height],
                [width * (1 - multiplier_width), height * multiplier_height],
                [width * (1 - multiplier_width), height * (1 - multiplier_height)],
                [width * multiplier_width, height * (1 - multiplier_height)],
            ]
        )
        dst_pts = np.float32([[0, 0], [300, 0], [300, 300], [0, 300]])
        return cv.getPerspectiveTransform(src_pts, dst_pts)

    m = _get_transform_matrix()

    # Prepare the video writer
    fourcc = cv.VideoWriter_fourcc(*"mp4v")
    out = cv.VideoWriter(output_file, fourcc, fps, (300, 300))

    # Process each frame
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        transformed = cv.warpPerspective(frame, m, (300, 300))
        out.write(transformed.astype("uint8"))

    # Clean up
    cap.release()
    out.release()
    cv.destroyAllWindows()

    return output_file


def process_image_perspective(input_file: str) -> str:
    """
    Apply a fixed perspective transform to a single image.
    Returns the path to the written result.
    """
    output_file = "image/view_result.jpg"
    multiplier_width = 0.246
    multiplier_height = 0.1875

    img = cv.imread(input_file)
    assert img is not None, f"Cannot read image file: {input_file}"

    height, width = img.shape[:2]

    # Build the transformation matrix
    src_pts = np.float32(
        [
            [width * multiplier_width, height * multiplier_height],
            [width * (1 - multiplier_width), height * multiplier_height],
            [width * (1 - multiplier_width), height * (1 - multiplier_height)],
            [width * multiplier_width, height * (1 - multiplier_height)],
        ]
    )
    dst_pts = np.float32([[0, 0], [300, 0], [300, 300], [0, 300]])
    m = cv.getPerspectiveTransform(src_pts, dst_pts)

    # Apply and save
    transformed = cv.warpPerspective(img, m, (300, 300))
    cv.imwrite(output_file, transformed)

    return output_file


def perform_realtime_perspective_transform(
    frame: np.ndarray, width: int, height: int
) -> np.ndarray:
    """
    Apply the same perspective transform to a single frame (e.g., live video).
    """
    multiplier_width = 0.246
    multiplier_height = 0.1875

    src_pts = np.float32(
        [
            [width * multiplier_width, height * multiplier_height],
            [width * (1 - multiplier_width), height * multiplier_height],
            [width * (1 - multiplier_width), height * (1 - multiplier_height)],
            [width * multiplier_width, height * (1 - multiplier_height)],
        ]
    )
    dst_pts = np.float32([[0, 0], [300, 0], [300, 300], [0, 300]])
    m = cv.getPerspectiveTransform(src_pts, dst_pts)

    transformed = cv.warpPerspective(frame, m, (300, 300))
    return transformed.astype("uint8")
