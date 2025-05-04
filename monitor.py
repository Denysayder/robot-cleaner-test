"""
Real‑time monitoring and control pipeline.

Receives a video stream from the Raspberry Pi camera, pre‑processes each frame
(brightness adjustment, thresholding, histogram equalisation, Fourier spectrum),
compares it with a clean reference spectrum (NMI metric + Kalman smoothing),
classifies the situation with a Bayesian decision rule and sends a command to
an Arduino controller.
Communication with the outside world goes through Redis (for images / high‑level
controls) and the Arduino serial line (for low‑level actuator commands).

All domain‑specific helper functions are imported from the project’s own
modules.  Make sure they’re available in PYTHONPATH before you run the script.
"""

import base64
import csv
import time

import cv2 as cv
import numpy as np
import picamera

# ──────────────────────────────────────────────── custom modules ──────────────
from arduino_sender import (
    close_serial_connection,
    command_to_send_to_arduino,
    find_port,
    open_serial_connection,
    receive_data,
    send_data,
)
from Bayes_class_decision import get_parameters, predict_group
from correlation import extract_spectrum, extract_spectrum_on_frame, spectrum_to_see
from fourier_processors import (
    perform_fourier_transform_and_compare_to_reference_fourier_image,
    process_image_fourier,
)
from histogram_equalizer import (
    histogram_equalization,
    histogram_equalization_on_frame,
)
from light_to_dark import (
    perform_brightness_thresholding,
    perform_brightness_thresholding_on_image,
)
from adjust_brightness import adjust_brightness_on_frame, adjust_brightness_on_image
from perspective_processor import (
    perform_realtime_perspective_transform,
    process_image_perspective,
)
from redis_processors import (
    connect_redis,
    disconnect_redis,
    receive_signal,
    send_signal,
)
from similarity_NMI import compare_images

# from remove_reflection import remove_reflection, remove_reflection_on_frame
# ──────────────────────────────────────────────────────────────────────────────


def main_run(reference_image: str,
             baud_rate: int,
             redis_receive_keys: list[str],
             arduino_data_keys: list[str]) -> None:
    """
    Main processing loop.

    Parameters
    ----------
    reference_image : str
        Path to the pre‑computed, “clean” reference image (grayscale) used for
        the Fourier‑domain comparison.
    baud_rate : int
        Serial speed for the Arduino link.
    redis_receive_keys : list[str]
        Redis keys from which we expect high‑level commands.
    arduino_data_keys : list[str]
        Names that map Arduino response bytes to semantic identifiers.
    """

    # ── Redis connection ──────────────────────────────────────────────────
    redis_host = "redis.gce.cloud.redislabs.com"
    redis_port = 17060
    redis_password = "YS9EKyvuTG5Q3HGxKFem48ePa7yka123"
    redis_conn = connect_redis(redis_host, redis_port, redis_password)

    # ── Arduino serial link ───────────────────────────────────────────────
    arduino_port = find_port()
    ser = open_serial_connection(arduino_port, baud_rate)

    # ── Load & prepare reference image (Fourier spectrum) ────────────────
    ref_img = cv.imread(reference_image, cv.IMREAD_GRAYSCALE)
    reference_fourier_frame = spectrum_to_see(ref_img)
    cv.imwrite("image/fourier_reference.jpg", reference_fourier_frame)

    # ── Raspberry‑Pi camera initialisation ───────────────────────────────
    with picamera.PiCamera() as camera, \
            open("data_test.csv", "w", newline="") as csvfile:

        camera.resolution = (480, 240)
        camera.framerate = 10
        camera.start_preview()

        width, height = camera.resolution
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["Timestamp", "NMI_Score_Filtered"])

        # ── Kalman filter for NMI smoothing ────────────────────────────
        kalman = cv.KalmanFilter(1, 1)
        kalman.transitionMatrix = np.array([[1]], np.float32)
        kalman.measurementMatrix = np.array([[1]], np.float32)
        kalman.processNoiseCov = np.array([[1e-5]], np.float32)
        kalman.measurementNoiseCov = np.array([[1e-3]], np.float32)
        kalman.errorCovPost = np.array([[1]], np.float32)
        kalman.statePost = np.array([[0]], np.float32)

        frame_interval = 1 / 10       # 10 FPS max
        last_frame_time = time.time()
        start_time = last_frame_time

        # ── Bayes classifier parameters ───────────────────────────────
        mean_clean, std_clean = get_parameters("clean_parameters.csv")
        mean_dirty, std_dirty = get_parameters("dirty_parameters.csv")

        try:
            while True:
                now = time.time()
                if now - last_frame_time < frame_interval:
                    continue
                last_frame_time = now

                # ── Grab a frame from the camera ──────────────────
                raw = np.empty(height * width * 3, dtype=np.uint8)
                camera.capture(raw, "bgr")
                frame = raw.reshape((height, width, 3))

                # ── Optional remote camera switch via Redis ───────
                if receive_signal(redis_conn, "camera") != "on":
                    continue

                # ── Push low‑resolution preview to Redis ──────────
                _, enc = cv.imencode(".jpg", frame)
                b64 = base64.b64encode(enc).decode()
                chunk_size = 60_000
                chunks = [b64[i:i + chunk_size] for i in range(0, len(b64), chunk_size)]
                chunks = [f"{i + 1}_{c}" for i, c in enumerate(chunks)]
                chunks[-1] += "_endframe"
                chunks = [f"{c}_notyet" if i < len(chunks) - 1 else c
                          for i, c in enumerate(chunks)]
                for c in chunks:
                    redis_conn.set("video", c)

                # ── Image processing pipeline ─────────────────────
                bright = adjust_brightness_on_frame(frame, 100)
                thresh = perform_brightness_thresholding(bright, 150)
                equalised = histogram_equalization_on_frame(thresh)
                fourier_frame = spectrum_to_see(equalised)

                # ── Similarity (NMI) + Kalman filter ──────────────
                nmi = compare_images(reference_fourier_frame, fourier_frame)
                filtered = kalman.correct(np.array([[nmi]], np.float32))[0, 0]

                # ── Log to CSV ─────────────────────────────────────
                csvwriter.writerow([now - start_time, filtered])

                # ── High‑level commands from Redis ────────────────
                redis_values = [receive_signal(redis_conn, k) for k in redis_receive_keys]
                for i, v in enumerate(redis_values):
                    send_data(ser, baud_rate, v, i)

                # ── Bayesian decision → Arduino command ───────────
                command = predict_group(filtered, mean_clean, std_clean,
                                        mean_dirty, std_dirty)
                send_data(ser, baud_rate, command, 1)

                # ── Debug prints ──────────────────────────────────
                print(f"NMI_Score: {filtered:.3f} | Command: {command}")

        except KeyboardInterrupt:
            print("Interrupted by user – shutting down …")

        finally:
            # Clean‑up
            camera.stop_preview()
            close_serial_connection(ser)
            disconnect_redis(redis_conn)


# ────────────────────────────────────────────────── CLI entry point ────────────
if __name__ == "__main__":
    redis_keys = ["move"]
    arduino_data_keys = [
        "Temperature", "Sensor1", "Sensor2", "Sensor3",
        "Sensor4", "water", "battery", "Moved Distance",
    ]

    input_image = "image/webcam_image5.jpg"
    bright = adjust_brightness_on_image(input_image, 100)
    thresh = perform_brightness_thresholding_on_image(bright, 150)
    reference = histogram_equalization(thresh)

    main_run(reference, baud_rate=9600,
             redis_receive_keys=redis_keys,
             arduino_data_keys=arduino_data_keys)
