import base64
import sys
import termios
import tty
from typing import List, Optional

import cv2
import numpy as np
import redis
from arduino_sender import (
    send_to_arduino,
    open_serial_connection,
    send_data,
    close_serial_connection,
    find_port,
)

# ----------------------------------------------------------------------
# Redis connection parameters
# ----------------------------------------------------------------------
RedisHost: str = "redis-"
RedisPort: int = 17942
RedisPassword: str = "123"

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def getch() -> str:
    """
    Read a single character from stdin without waiting for a newline.
    Useful for quitting loops with a keypress (e.g., press 'q' to exit).
    """
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch


# ----------------------------------------------------------------------
# Redis utilities
# ----------------------------------------------------------------------
def connect_redis(host: str, port: int, password: str) -> redis.Redis:
    """
    Create a Redis connection.
    """
    return redis.Redis(host=host, port=port, db=0, password=password)


def send_signal(r: redis.Redis, key: str, value: str) -> None:
    """
    Send a signal (string value) to Redis at the specified key.
    """
    if r.ping():
        r.set(key, value)
        print(f"Signal successfully sent to key: {key}")
    else:
        print("No connection to Redis. Signal NOT sent.")


def receive_signal(r: redis.Redis, key: str) -> Optional[str]:
    """
    Receive a signal (string value) from Redis by key.
    Returns the decoded string, or None if the key is missing.
    """
    value = r.get(key)
    if value is None:
        print(f"Nothing from key: {key}")
        return None

    decoded = value.decode("utf-8")
    print(f"Received data from key: {key}")
    return decoded


def disconnect_redis(r: redis.Redis) -> None:
    """
    Close the Redis connection.
    """
    r.close()


# ----------------------------------------------------------------------
# Arduino & signal handling
# ----------------------------------------------------------------------
def receive_realtime_signal_and_send_to_arduino(redis_keys: List[str]) -> None:
    """
    Continuously read signals from Redis keys and forward them to an Arduino.
    Press 'q' in the terminal to stop.
    """
    redis_conn = connect_redis(RedisHost, RedisPort, RedisPassword)
    arduino_port = find_port()
    ser = open_serial_connection(arduino_port, 9600)

    try:
        while True:
            if getch() == "q":
                break

            received_values: List[Optional[str]] = []
            for key in redis_keys:
                received_values.append(receive_signal(redis_conn, key))

            for idx, value in enumerate(received_values):
                # Skip None values to avoid sending empty data
                if value is not None:
                    send_data(ser, value, idx + 1)
    finally:
        close_serial_connection(ser)
        disconnect_redis(redis_conn)


# ----------------------------------------------------------------------
# Video frame handling
# ----------------------------------------------------------------------
def decode_frame(frame_for_decode: str) -> Optional[np.ndarray]:
    """
    Decode a base64‐encoded video frame into a NumPy BGR image.
    Returns None if decoding fails.
    """
    if not frame_for_decode:
        return None

    try:
        img_bytes = base64.b64decode(frame_for_decode)
        nparr = np.frombuffer(img_bytes, dtype=np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return frame
    except (base64.binascii.Error, ValueError):
        return None


def receive_realtime_frame() -> None:
    """
    Continuously read base64‐encoded video chunks from Redis key 'video',
    reconstruct full frames, and display them in real time.
    Press 'q' in the display window OR in the terminal to stop.
    """
    redis_conn = connect_redis(RedisHost, RedisPort, RedisPassword)
    full_frame_data: str = ""
    frame_for_decode: str = ""

    try:
        while True:
            # Quit the outer loop via terminal keypress
            if getch() == "q":
                break

            # Read chunk from Redis
            frame_chunk = receive_signal(redis_conn, "video")
            if frame_chunk is None:
                continue

            # Parse chunk structure: "<prefix>_<data>_<suffix>"
            try:
                chunk_prefix, data, chunk_suffix = frame_chunk.split("_")
            except ValueError:
                # Malformed chunk; skip
                continue

            # Accumulate frame data
            full_frame_data += data

            # At end of frame, decode and show
            if chunk_suffix == "endframe":
                frame_for_decode = full_frame_data
                full_frame_data = ""

                video_frame = decode_frame(frame_for_decode)
                if video_frame is not None:
                    cv2.imshow("Video", video_frame)
                    # Exit via 'q' in the display window
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
    finally:
        cv2.destroyAllWindows()
        disconnect_redis(redis_conn)


# ----------------------------------------------------------------------
# Example usage (commented out to keep module import‑safe)
# ----------------------------------------------------------------------
# if __name__ == "__main__":
#     # Receive movement/toggle signals and forward to Arduino
#     redis_keys = ["move", "toggle"]
#     receive_realtime_signal_and_send_to_arduino(redis_keys)
#
#     # Receive and display real‑time video stream
#     # receive_realtime_frame()
