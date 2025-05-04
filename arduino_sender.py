"""
Utility functions for talking to an Arduino over a serial (USB) connection.

All helpers are kept in this single module so that other files can import them
without worrying about side‑effects.  Function names and signatures stay exactly
as in the original snippet—only formatting and obvious bugs were fixed.
"""

import sys
import time
import termios
import tty

import serial
import serial.tools.list_ports


# --------------------------------------------------------------------------- #
#  Low‑level helpers                                                          #
# --------------------------------------------------------------------------- #
def getch() -> str:
    """
    Read a single character from stdin without the need to press Enter.
    Works on Unix‑like systems.
    """
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch


# --------------------------------------------------------------------------- #
#  Port discovery & connection management                                     #
# --------------------------------------------------------------------------- #
def find_port() -> str | None:
    """
    Return the device name of the first serial port that looks like an
    Arduino/CH‑340 device.  Prints a short report on each port while scanning.
    """
    arduino_port = None
    for port in serial.tools.list_ports.comports():
        print("Port:", port.device)
        print("Description:", port.description)
        print()

        if (
            "Arduino" in port.description
            or "ACM" in port.description
            or "USB‑SERIAL CH340" in port.description
        ):
            arduino_port = port.device
            print("Arduino is connected to", arduino_port)
            break

    if arduino_port is None:
        print("Arduino is not connected to any port")

    return arduino_port


def open_serial_connection(arduino_port: str | None, baud_rate: int = 9600):
    """
    Open a Serial() object (8‑N‑1) to the selected port.  Returns None on failure.
    """
    if arduino_port is None:
        return None

    try:
        ser = serial.Serial(arduino_port, baud_rate, exclusive=True)
        # Give the board time to reset
        time.sleep(2)
        return ser
    except serial.SerialException as e:
        print("An error occurred while opening the serial connection:", e)
        return None


def close_serial_connection(ser):
    """Close the serial port if it is open."""
    if ser is not None:
        ser.close()


# --------------------------------------------------------------------------- #
#  Data exchange helpers                                                      #
# --------------------------------------------------------------------------- #
def send_data(ser, baudrate, data, channel = 0):
    """
    Send *data* to the Arduino on *channel* using the frame:

        <channel>:<data>#

    If *ser* is None the function tries to open a connection automatically.
    """
    if ser is None:
        arduino_port = find_port()
        ser = open_serial_connection(arduino_port, 9600)

    if ser is None:
        print("Connection to Arduino is not established.")
        return

    # Prepare payload
    if isinstance(data, bytes):
        data_bytes = data
    else:
        data_bytes = str(data).encode()

    packet = f"{channel}:".encode() + data_bytes + b"#"

    try:
        ser.write(packet)
        print(f"Data: {data} has been sent to channel {channel}")
    except serial.SerialException as e:
        print("An error occurred while sending data:", e)


def receive_data(ser):
    """
    Poll the serial buffer and return all complete <channel>:<data> frames
    as a list indexed by channel number (missing indices are None).
    """
    if ser is None:
        print("Connection is not established.")
        return []

    received: list[str | None] = []

    while ser.in_waiting:
        line = ser.readline().decode("latin-1").strip()
        if ":" not in line:
            continue

        prefix, data = line.split(":", 1)
        if not prefix.isdigit():
            continue

        idx = int(prefix)

        # Extend list if necessary
        if idx >= len(received):
            received.extend([None] * (idx - len(received) + 1))

        received[idx] = data

    if received:
        print("Received data from Arduino")
    else:
        print("No data received from Arduino")

    return received


# --------------------------------------------------------------------------- #
#  Convenience wrappers                                                       #
# --------------------------------------------------------------------------- #
def send_to_arduino(data, baud_rate: int = 9600, channel: int = 1):
    arduino_port = find_port()
    ser = open_serial_connection(arduino_port, baud_rate)
    send_data(ser, baud_rate, data, channel)
    close_serial_connection(ser)



def receive_from_arduino(baud_rate: int = 9600):
    """
    One‑shot helper: open port ➜ read pending data ➜ close port.
    """
    arduino_port = find_port()
    ser = open_serial_connection(arduino_port, baud_rate)
    data = receive_data(ser)
    close_serial_connection(ser)
    return data


# --------------------------------------------------------------------------- #
#  Small domain‑specific helper                                               #
# --------------------------------------------------------------------------- #
def command_to_send_to_arduino(float_value: float, base_data):
    """
    Example mapping:  return 'G' for negative values, else forward *base_data*.
    """
    return "G" if float_value < 0 else base_data
