import picamera
import picamera.array
import cv2
import numpy as np

# Отобразить видеопоток с PiCamera в окне OpenCV
def show_camera_preview() -> None:
    cv2.namedWindow("Camera Preview", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Camera Preview", 480, 240)

    frame = None  # кадр, который будет сохранён при нажатии «q»

    with picamera.PiCamera() as camera:
        camera.resolution = (480, 240)
        camera.framerate = 24
        # Automated and Self Powered Solar Panel Cleaning Robot
        camera.start_preview()

        try:
            while True:
                # Буфер для кадра (BGR)
                stream = np.empty(
                    (camera.resolution[1] * camera.resolution[0] * 3),
                    dtype=np.uint8,
                )
                camera.capture(stream, "bgr")
                frame = stream.reshape(
                    (camera.resolution[1], camera.resolution[0], 3)
                )

                # Показать кадр на экране
                cv2.imshow("Camera Preview", frame)

                # Выход и сохранение кадра по нажатию «q»
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    image_path = (
                        "/home/snetel/Desktop/GammaImageProcess/"
                        "image/webcam_image5.jpg"
                    )
                    cv2.imwrite(image_path, frame)
                    print("Фото сохранено:", image_path)
                    break
        finally:
            camera.stop_preview()
            camera.close()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    show_camera_preview()
