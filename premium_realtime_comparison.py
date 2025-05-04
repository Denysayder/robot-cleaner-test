#!/usr/bin/env python3
# premium_realtime_comparison.py
"""
Основной конвейер:
1. Захват кадров с Raspberry Pi камеры (или веб‑камеры, если picamera недоступна).
2. Предобработка кадра – выравнивание яркости, пороговая обработка, гистограммирование.
3. Преобразование Фурье и вычисление сходства (NMI) с эталонным изображением.
4. Байесовская классификация → команда Arduino.
5. Передача/приём вспомогательных сигналов через Redis и UART.
"""

from __future__ import annotations

import base64
import csv
import time
from pathlib import Path
from typing import Generator, List, Tuple

import cv2 as cv
import numpy as np

# --------------------------------------------------------------------------- #
# Опциональный импорт picamera (RPi). Если модуля нет – используем cv2.VideoCapture
# --------------------------------------------------------------------------- #
try:
    import picamera
    _HAS_PICAMERA = True
except ModuleNotFoundError:
    picamera = None            # type: ignore
    _HAS_PICAMERA = False

# --------------------------------------------------------------------------- #
# Внутренние модули проекта
# --------------------------------------------------------------------------- #
from fourier_processor import (
    perform_fourier_transform_and_compare_to_reference_fourier_image,
    process_image_fourier,
)
from arduino_sender import (
    open_serial_connection,
    close_serial_connection,
    send_data,
    command_to_send_to_arduino,
    find_port,
    receive_data,
)
from perspective_processor import (
    perform_realtime_perspective_transform,
    process_image_perspective,
)
from redis_processor import (
    connect_redis,
    receive_signal,
    disconnect_redis,
    send_signal,
)
from histogram_equalizer import (
    histogram_equalization,
    histogram_equalization_on_frame,
)
# from remove_reflection import remove_reflection, remove_reflection_on_frame
from adjust_brightness import (
    adjust_brightness_on_frame,
    adjust_brightness_on_image,
)
from correlation import (
    extract_spectrum,
    extract_spectrum_on_frame,
    spectrum_to_see,
)
from similarity_NMI import compare_images
from light_to_dark import (
    perform_brightness_thresholding,
    perform_brightness_thresholding_on_image,
)
from bayes_class_decision import predict_group, get_parameters


# --------------------------------------------------------------------------- #
# Service: универсальный генератор кадров
# --------------------------------------------------------------------------- #
def _frame_stream(resolution: Tuple[int, int],
                  framerate: int) -> Generator[np.ndarray, None, None]:
    """
    Возвращает бесконечный генератор кадров.
    Если есть picamera — используется она; иначе — cv2.VideoCapture(0).

    :param resolution: (width, height) кадров
    :param framerate:  требуемый FPS
    """
    width, height = resolution

    if _HAS_PICAMERA:
        with picamera.PiCamera() as camera:          # type: ignore[attr-defined]
            camera.resolution = resolution
            camera.framerate = framerate
            # Буфер для raw‑захвата
            stream = np.empty((height * width * 3), dtype=np.uint8)

            while True:
                camera.capture(stream, 'bgr')
                yield stream.reshape((height, width, 3))
    else:
        cap = cv.VideoCapture(0, cv.CAP_AVFOUNDATION)  # AVFoundation – для macOS; на других ОС можно опустить
        cap.set(cv.CAP_PROP_FRAME_WIDTH,  width)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv.CAP_PROP_FPS,         framerate)

        if not cap.isOpened():
            raise RuntimeError("Не удалось открыть веб‑камеру (index 0)")

        while True:
            ok, frame = cap.read()
            if ok:
                yield frame
            else:
                time.sleep(0.01)  # небольшая пауза при ошибке чтения


# --------------------------------------------------------------------------- #
# Main pipeline
# --------------------------------------------------------------------------- #
def Main_Run(
    reference_image: str | Path,
    baud_rate: int,
    redis_receive_keys: List[str],
    arduino_data_keys: List[str],
) -> None:
    # --------------------------------------------------------------------- #
    # 1. Connections
    # --------------------------------------------------------------------- #
    redis_conn = connect_redis(
        host="redis-",
        port=17942,
        password="123",
    )

    arduino_port = find_port()
    ser = open_serial_connection(arduino_port, baud_rate)

    # --------------------------------------------------------------------- #
    # 2. Reference image → Fourier spectrum
    # --------------------------------------------------------------------- #
    ref_img = cv.imread(str(reference_image), cv.IMREAD_GRAYSCALE)
    if ref_img is None:
        raise FileNotFoundError(f"Не удалось открыть эталонное изображение: {reference_image}")

    reference_fourier_frame = spectrum_to_see(ref_img)
    cv.imwrite("image/fourier_image.jpg", reference_fourier_frame)

    # --------------------------------------------------------------------- #
    # 3. Получаем бесконечный поток кадров
    # --------------------------------------------------------------------- #
    resolution = (480, 240)
    framerate  = 10
    frames = _frame_stream(resolution, framerate)

    width, height = resolution

    # --------------------------------------------------------------------- #
    # 4. Kalman filter setup
    # --------------------------------------------------------------------- #
    kalman = cv.KalmanFilter(1, 1, 0)
    kalman.transitionMatrix    = np.array([[1]],    np.float32)
    kalman.measurementMatrix   = np.array([[1]],    np.float32)
    kalman.processNoiseCov     = np.array([[1e-5]], np.float32)
    kalman.measurementNoiseCov = np.array([[1e-3]], np.float32)
    kalman.errorCovPost        = np.array([[1]],    np.float32)
    kalman.statePost           = np.array([[0]],    np.float32)

    frame_rate_limit = framerate
    frame_interval   = 1 / frame_rate_limit
    last_frame_time  = time.time()
    first_frame_time = last_frame_time

    mean_clean, std_clean = get_parameters("clean_parameters.csv")
    mean_dirty, std_dirty = get_parameters("dirty_parameters.csv")

    with open("data_test.csv", "w", newline="") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["Timestamp", "NMI_Score_Filtered"])

        # ------------------------------------------------------------- #
        # 5. Main loop
        # ------------------------------------------------------------- #
        for frame in frames:
            current_time = time.time()
            if current_time - last_frame_time < frame_interval:
                continue
            last_frame_time = current_time

            # --------------------------------------------------------- #
            # 5.1 Redis: отдаём кадр по запросу
            # --------------------------------------------------------- #
            if receive_signal(redis_conn, "camera") == "on":
                _, img_encoded = cv.imencode(".jpg", frame)
                img_base64 = base64.b64encode(img_encoded).decode()

                chunk_size = 60_000
                chunks = [
                    img_base64[i : i + chunk_size]
                    for i in range(0, len(img_base64), chunk_size)
                ]
                flagged_chunks = [f"{idx + 1}_{chunk}" for idx, chunk in enumerate(chunks)]
                flagged_chunks[-1] += "_endframe"
                flagged_chunks = [
                    f"{c}_notyet" if i != len(flagged_chunks) - 1 else c
                    for i, c in enumerate(flagged_chunks)
                ]
                for fc in flagged_chunks:
                    redis_conn.set("video", fc)

            # --------------------------------------------------------- #
            # 5.2 Предобработка изображения
            # --------------------------------------------------------- #
            # transformed = perform_realtime_perspective_transform(frame, width, height)

            bright_frame    = adjust_brightness_on_frame(frame, 100)
            thresh_frame    = perform_brightness_thresholding(bright_frame, 150)
            equalized_frame = histogram_equalization_on_frame(thresh_frame)
            # reflection_frame = remove_reflection_on_frame(equalized_frame, 50)

            fourier_frame = spectrum_to_see(equalized_frame)

            # 5.3 NMI similarity
            nmi_score = compare_images(reference_fourier_frame, fourier_frame)

            # 5.4 Kalman‑сглаживание
            _ = kalman.predict()
            kalman_corrected = kalman.correct(np.array([[nmi_score]], np.float32))
            nmi_filtered = float(kalman_corrected[0, 0])
            kalman.statePost = kalman_corrected

            # 5.5 Логирование
            timestamp = current_time - first_frame_time
            csvwriter.writerow([timestamp, nmi_filtered])

            # --------------------------------------------------------- #
            # 5.6 Передача/приём вспомогательных сигналов
            # --------------------------------------------------------- #
            redis_values = [receive_signal(redis_conn, k) for k in redis_receive_keys]
            for idx, val in enumerate(redis_values):
                send_data(ser, baud_rate, val, idx)

            # 5.7 Байес → команда
            the_command = predict_group(
                nmi_filtered, mean_clean, std_clean, mean_dirty, std_dirty
            )
            send_data(ser, baud_rate, the_command, 1)

            # 5.8 Отладочный вывод
            print(f"NMI_Score: {nmi_filtered:.2f}  |  Command → {the_command}")

    # --------------------------------------------------------------------- #
    # 6. Cleanup
    # --------------------------------------------------------------------- #
    close_serial_connection(ser)
    disconnect_redis(redis_conn)


# --------------------------------------------------------------------------- #
# Stand‑alone execution
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    redis_keys = ["move"]
    arduino_data_keys = [
        "Temperature",
        "Sensor1",
        "Sensor2",
        "Sensor3",
        "Sensor4",
        "water",
        "battery",
        "Moved Distance",
    ]

    input_image_file = "image/webcam_image5.jpg"

    # Предобработка эталона
    bright_file    = adjust_brightness_on_image(input_image_file, 100)
    thresh_file    = perform_brightness_thresholding_on_image(bright_file, 150)
    equalized_file = histogram_equalization(thresh_file)

    Main_Run(
        reference_image=equalized_file,
        baud_rate=9600,
        redis_receive_keys=redis_keys,
        arduino_data_keys=arduino_data_keys,
    )
