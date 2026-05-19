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
# Instelbare parameters
# ==========================
MIN_CONTOUR_AREA = 300      # ruis filteren (pixels^2)
STEER_FULL = 1.0            # vol sturen
STEER_CORRECT = 0.4         # zachte correctie terug naar midden
THROTTLE_DRIVE = 1.0        # rechtdoor
THROTTLE_TURN = 0.5         # tijdens sturen langzamer
 
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
# Helper - grootste contour vinden + centroid X
# ==========================
def largest_contour_x(contours):
    """
    Zoekt de grootste contour boven MIN_CONTOUR_AREA
    en geeft de X-coordinaat van het zwaartepunt terug,
    samen met de oppervlakte. Returns (None, 0) als niets gevonden.
    """
    best_x = None
    best_area = 0
 
    for c in contours:
 
        area = cv2.contourArea(c)
 
        if area < MIN_CONTOUR_AREA:
            continue
 
        if area > best_area:
 
            M = cv2.moments(c)
 
            if M["m00"] == 0:
                continue
 
            cx = int(M["m10"] / M["m00"])
            best_x = cx
            best_area = area
 
    return best_x, best_area
 
# ==========================
# Thread 2 - Beeldverwerking + sturen
# ==========================
def control_thread():
 
    while not stop_event.is_set():
 
        if frame_queue.empty():
            continue
 
        frame = frame_queue.get()
 
        # Beeldbreedte ophalen (dynamisch, klopt altijd)
        h, w = frame.shape[:2]
 
        # ==========================
        # Zone-grenzen berekenen
        # ==========================
        # 1/3 grenzen
        third_left = w / 3.0          # linker 1/3 eindigt hier
        third_right = w * 2.0 / 3.0   # rechter 1/3 begint hier
 
        # 1/8 grenzen (uiterste randen voor terug-sturen)
        eighth_left = w / 8.0         # meest linker 1/8 eindigt hier
        eighth_right = w * 7.0 / 8.0  # meest rechter 1/8 begint hier
 
        # Blur voor stabielere detectie
        blurred = cv2.GaussianBlur(frame, (11, 11), 0)
 
        # BGR -> HSV
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
 
        # ==========================
        # Groen detectie
        # ==========================
        lower_green = np.array([35, 80, 80])
        upper_green = np.array([85, 255, 255])
 
        mask_green = cv2.inRange(hsv, lower_green, upper_green)
 
        # ==========================
        # Rood detectie
        # ==========================
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
 
        lower_red2 = np.array([170, 120, 120])
        upper_red2 = np.array([180, 255, 255])
 
        mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
 
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
 
        # Grootste contour per kleur
        green_x, green_area = largest_contour_x(contours_green)
        red_x, red_area = largest_contour_x(contours_red)
 
        # ==========================
        # STUUR LOGICA - grootste contour wint
        # ==========================
        steering = 0.0
        throttle = THROTTLE_DRIVE
 
        # Bepaal welke kleur dominant is (dichtstbij = grootste contour)
        use_green = green_x is not None and green_area >= red_area
        use_red = red_x is not None and red_area > green_area
 
        if use_green:
            # GROENE pilaar -> auto moet er linksom heen -> stuur LINKS (+)
            #
            # Zones (van links naar rechts):
            #   [0  ...  third_right]            -> vol links sturen
            #   [third_right ... eighth_right]   -> rechtdoor (pilaar zit al rechts genoeg)
            #   [eighth_right ... w]             -> zachte correctie naar rechts (terug naar midden)
 
            if green_x < third_right:
                # Pilaar nog niet ver genoeg rechts -> blijf links sturen
                steering = STEER_FULL
                throttle = THROTTLE_TURN
 
            elif green_x < eighth_right:
                # Pilaar zit in rechter 1/3 maar nog niet in rechter 1/8 -> rechtdoor
                steering = 0.0
                throttle = THROTTLE_DRIVE
 
            else:
                # Pilaar zit in uiterste rechter 1/8 -> zacht terug naar midden
                steering = -STEER_CORRECT
                throttle = THROTTLE_DRIVE
 
        elif use_red:
            # RODE pilaar -> auto moet er rechtsom heen -> stuur RECHTS (-)
            #
            # Zones (van links naar rechts):
            #   [0 ... eighth_left]              -> zachte correctie naar links (terug naar midden)
            #   [eighth_left ... third_left]     -> rechtdoor (pilaar zit al links genoeg)
            #   [third_left ... w]               -> vol rechts sturen
 
            if red_x > third_left:
                # Pilaar nog niet ver genoeg links -> blijf rechts sturen
                steering = -STEER_FULL
                throttle = THROTTLE_TURN
 
            elif red_x > eighth_left:
                # Pilaar zit in linker 1/3 maar nog niet in linker 1/8 -> rechtdoor
                steering = 0.0
                throttle = THROTTLE_DRIVE
 
            else:
                # Pilaar zit in uiterste linker 1/8 -> zacht terug naar midden
                steering = STEER_CORRECT
                throttle = THROTTLE_DRIVE
 
        else:
            # Geen pilaar gezien -> rechtdoor
            steering = 0.0
            throttle = THROTTLE_DRIVE
 
        # Toepassen
        car.steering = steering
        motor_kit.motor3.throttle = throttle
 
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
 
