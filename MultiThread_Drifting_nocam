from jetracer.nvidia_racecar import NvidiaRacecar
import time
import busio
import board
from adafruit_motorkit import MotorKit
import cv2
import numpy as np
import threading
import queue

# ==========================
# Hardware initialisatie
# ==========================
car = NvidiaRacecar()

motor_kit = MotorKit(
    i2c=busio.I2C(board.SCL, board.SDA),
    address=0x60
)

motor_kit.motor3.throttle = 0.0
car.steering = 0.0

time.sleep(1)

# ==========================
# Stop event voor veilige shutdown
# ==========================
stop_event = threading.Event()

# ==========================
# Camera pipeline
# ==========================
def gstreamer_pipeline(
        capture_width=1280,
        capture_height=720,
        display_width=640,
        display_height=480,
        framerate=30,
        flip_method=0,
):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        f"width=(int){capture_width}, "
        f"height=(int){capture_height}, "
        f"format=(string)NV12, "
        f"framerate=(fraction){framerate}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        "video/x-raw, "
        "width=(int){}, "
        "height=(int){}, "
        "format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
    ).format(display_width, display_height)

# ==========================
# Camera openen
# ==========================
cap = cv2.VideoCapture(
    gstreamer_pipeline(),
    cv2.CAP_GSTREAMER
)

# ==========================
# Queue voor frames
# ==========================
frame_queue = queue.Queue(maxsize=5)

# ==========================
# Thread 1 - Camera uitlezen
# ==========================
def camera_thread():

    while not stop_event.is_set():

        ret, frame = cap.read()

        if not ret:
            continue

        # Oude frames verwijderen
        if frame_queue.full():
            try:
                frame_queue.get_nowait()
            except queue.Empty:
                pass

        frame_queue.put(frame)

# ==========================
# Thread 2 - Beeldverwerking + sturen
# ==========================
def control_thread():

    while not stop_event.is_set():

        if not frame_queue.empty():

            frame = frame_queue.get()

            # Standaard snelheid
            motor_kit.motor3.throttle = 1.0

            # Blur voor stabielere detectie
            blurred = cv2.GaussianBlur(frame, (11, 11), 0)

            # BGR -> HSV
            hsv = cv2.cvtColor(
                blurred,
                cv2.COLOR_BGR2HSV
            )

            # ==========================
            # Groen detectie
            # ==========================
            lower_green = np.array([35, 80, 80])
            upper_green = np.array([85, 255, 255])

            mask_green = cv2.inRange(
                hsv,
                lower_green,
                upper_green
            )

            # ==========================
            # Rood detectie
            # ==========================
            lower_red1 = np.array([0, 100, 100])
            upper_red1 = np.array([10, 255, 255])

            lower_red2 = np.array([170, 120, 120])
            upper_red2 = np.array([180, 255, 255])

            mask_red1 = cv2.inRange(
                hsv,
                lower_red1,
                upper_red1
            )

            mask_red2 = cv2.inRange(
                hsv,
                lower_red2,
                upper_red2
            )

            mask_red = mask_red1 + mask_red2

            # ==========================
            # Contours zoeken
            # ==========================
            contours_green, _ = cv2.findContours(
                mask_green,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            contours_red, _ = cv2.findContours(
                mask_red,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            # ==========================
            # STUUR LOGICA
            # ==========================

            # Groen -> links
            if contours_green:

                motor_kit.motor3.throttle = 0.5
                car.steering = 1.0

            # Rood -> rechts
            elif contours_red:

                motor_kit.motor3.throttle = 0.5
                car.steering = -1.0

            # Geen kleur
            else:
                car.steering = 0.0

    # ==========================
    # Extra veiligheid bij stoppen
    # ==========================
    motor_kit.motor3.throttle = 0.0
    car.steering = 0.0

# ==========================
# Threads aanmaken
# ==========================
t1 = threading.Thread(target=camera_thread)
t2 = threading.Thread(target=control_thread)

# ==========================
# Threads starten
# ==========================
t1.start()
t2.start()

# ==========================
# Hoofdprogramma
# ==========================
try:

    while True:
        time.sleep(0.1)

except KeyboardInterrupt:

    print("Programma stoppen...")

    stop_event.set()

    t1.join()
    t2.join()

finally:

    # ==========================
    # Alles veilig stoppen
    # ==========================
    try:
        motor_kit.motor3.throttle = 0.0
        car.steering = 0.0
    except:
        pass

    cap.release()

    print("Motoren gestopt")
