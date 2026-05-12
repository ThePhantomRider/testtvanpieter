from jetracer.nvidia_racecar import NvidiaRacecar
import time
import busio
import board
from adafruit_motorkit import MotorKit
import cv2
import numpy as np
import threading

# Positief is naar links toe rijden (sturen) Negatief is naar rechts toe rijden (sturen)

car = NvidiaRacecar()

motor_kit = MotorKit(i2c=busio.I2C(board.SCL, board.SDA), address=0x60)

motor_kit.motor3.throttle = 0.0
car.steering = 0.0
time.sleep(1)

# -1 (full back) to 1 (full forward)
# steering -1 to 1

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
                f"width=(int){capture_width}, height=(int){capture_height}, "
                f"format=(string)NV12, framerate=(fraction){framerate}/1 ! "
                f"nvvidconv flip-method={flip_method} ! "
                "video/x-raw, width=(int){}, height=(int){}, format=(string)BGRx ! "
                "videoconvert ! video/x-raw, format=(string)BGR ! appsink"
        ).format(display_width, display_height)

cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)

while True:
        thread = threading.Thread(target=cap)
        ret, frame = cap.read()
        if not ret:
                break

        motor_kit.motor3.throttle = 1.0

        # Blur voor stabielere detectie
        blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        
        #BGR -> HVS (BELANGRIJK)
        hvs = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        # ==================
        # Groen detectie
        # ==================
        lower_green = np.array([35, 80, 80])
        upper_green = np.array([85, 255, 255])
        mask_green = cv2.inRange(hvs, lower_green, upper_green)

        # ==================
        # Rood detectie (2 ranges!)
        # ==================
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])

        lower_red2 = np.array([170, 120, 120])
        upper_red2 = np.array([180, 255, 255])

        mask_red1 = cv2.inRange(hvs, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(hvs, lower_red2, upper_red2)

        mask_red = mask_red1 + mask_red2

        # ==================
        # Contours vinden
        # ==================
        Contours_green, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        Contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        height, width, _ = frame.shape
        center_x = width // 2

        # ==================
        # GROEN Logica
        # ==================
        if Contours_green:
                motor_kit.motor3.throttle = 0.5
                car.steering = 1.0
#               c = max(Contours_green, key=cv2.contourArea)
#               if cv2.contourArea(c) > 500:
#                       x, y, w, h = cv2.boundingRect(c)
#                       cx = x + w // 2
#
#                       cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
#                       cv2.putText(frame, "GROEN -> LINKS", (x, y-10),
#                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
#
#                       print("Groen gezien -> LINKS")

        # ==================
        # ROOD Logica
        # ==================
        if Contours_red:
                motor_kit.motor3.throttle = 0.5
                car.steering = -1.0
#               c = max(Contours_red, key=cv2.contourArea)
#               if cv2.contourArea(c) > 500:
#                       x, y, w, h = cv2.boundingRect(c)
#                       cx = x + w // 2
#
#                       cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
#                       cv2.putText(frame, "ROOD -> RECHTS", (x, y-10),
#                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
#
#                       print("Rood gezien -> RECHTS")

        # Middenlijn tekenen
        cv2.line(frame, (center_x, 0), (center_x, height), (255,255,255), 2)

        cv2.imshow("kleur detectie", frame)

        if cv2.waitKey(1) == 27:
                break

motor_kit.motor3.throttle = 0.0
car.steering = 0.0
cap.release()
cv2.destroyAllWindows()


