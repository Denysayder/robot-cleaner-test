import cv2
import numpy as np


def remove_reflection(image_path, threshold_adder):
    """
    Удаляет блики с изображения и сохраняет результат.

    :param image_path: путь к исходному изображению
    :param threshold_adder: значение, добавляемое к средней яркости
                             L‑канала для определения порога
    :return: путь к сохранённому обработанному изображению
    """
    # Читаем изображение
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Не удалось открыть файл: {image_path}")

    output_image_path = "image/reflection_remove_result.jpg"

    # Переход в цветовое пространство LAB
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # Порог по L‑каналу
    mean_brightness = np.mean(lab_image[:, :, 0])
    threshold = mean_brightness + threshold_adder

    # Маска возможных бликов
    reflection_mask = cv2.threshold(
        lab_image[:, :, 0], threshold, 255, cv2.THRESH_BINARY
    )[1]
    reflection_mask = cv2.dilate(reflection_mask, None, iterations=2)

    # Находим самый крупный контур (блик)
    contours, _ = cv2.findContours(
        reflection_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    processed_image = image.copy()
    if contours:
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        # «Затираем» блик медианным блюром
        processed_image[y : y + h, x : x + w] = cv2.medianBlur(
            processed_image[y : y + h, x : x + w], 15
        )

    # Сохраняем результат
    cv2.imwrite(output_image_path, processed_image)
    return output_image_path


def remove_reflection_on_frame(frame, threshold_adder):
    """
    Удаляет блики с отдельного кадра (numpy‑массива).

    :param frame: кадр BGR или grayscale
    :param threshold_adder: значение, добавляемое к средней яркости
                             L‑канала для определения порога
    :return: обработанный кадр (numpy‑массив)
    """
    if frame is None:
        raise ValueError("Входной кадр отсутствует")

    # Приводим к BGR, если кадр был в оттенках серого
    if len(frame.shape) == 2 or frame.shape[2] == 1:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    lab_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

    mean_brightness = np.mean(lab_frame[:, :, 0])
    threshold = mean_brightness + threshold_adder

    reflection_mask = cv2.threshold(
        lab_frame[:, :, 0], threshold, 255, cv2.THRESH_BINARY
    )[1]
    reflection_mask = cv2.dilate(reflection_mask, None, iterations=2)

    contours, _ = cv2.findContours(
        reflection_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    processed_frame = frame.copy()
    if contours:
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        processed_frame[y : y + h, x : x + w] = cv2.medianBlur(
            processed_frame[y : y + h, x : x + w], 15
        )

    return processed_frame
