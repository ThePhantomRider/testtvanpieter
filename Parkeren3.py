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

# S1
motor_kit.motor3.throttle = 0.0
for _ in range(3):
        car.steering = 1.0
        time.sleep(0.5)

motor_kit.motor3.throttle = 0.7
time.sleep(0.35)

# S2
motor_kit.motor3.throttle = 0.0
for _ in range(3):
        car.steering = -1.0
        time.sleep(0.5)

motor_kit.motor3.throttle = -0.7
time.sleep(0.25)

# S3
motor_kit.motor3.throttle = 0.0
for _ in range(3):
        car.steering = 1.0
        time.sleep(0.5)

motor_kit.motor3.throttle = 0.7
time.sleep(0.35)

# S4
motor_kit.motor3.throttle = 0.0
for _ in range(3):
        car.steering = -1.0
        time.sleep(0.5)

motor_kit.motor3.throttle = -0.7
time.sleep(0.25)

# S5
motor_kit.motor3.throttle = 0.0
for _ in range(3):
        car.steering = 1.0
        time.sleep(0.5)

motor_kit.motor3.throttle = 0.7
time.sleep(0.55)

# -1 (full back) to 1 (full forward)
# steering -1 to 1

# Eind
motor_kit.motor3.throttle = 0.0
car.steering = 0.0

# S-5
motor_kit.motor3.throttle = 0.0
for _ in range(3):
        car.steering = 1.0
        time.sleep(0.5)

motor_kit.motor3.throttle = -0.7
time.sleep(0.5)

# S-4
motor_kit.motor3.throttle = 0.0
for _ in range(3):
        car.steering = -1.0
        time.sleep(0.5)

motor_kit.motor3.throttle = 0.7
time.sleep(0.25)


# S-3
motor_kit.motor3.throttle = 0.0
for _ in range(3):
        car.steering = 1.0
        time.sleep(0.5)

motor_kit.motor3.throttle = -0.7
time.sleep(0.25)


# S-2
motor_kit.motor3.throttle = 0.0
for _ in range(3):
        car.steering = -1.0
        time.sleep(0.5)

motor_kit.motor3.throttle = 0.7
time.sleep(0.25)


# S-1
motor_kit.motor3.throttle = 0.0
for _ in range(3):
        car.steering = 1.0
        time.sleep(0.5)

motor_kit.motor3.throttle = -0.7
time.sleep(0.25)


# S-0
motor_kit.motor3.throttle = 0.0
for _ in range(3):
        car.steering = -1.0
        time.sleep(0.5)

motor_kit.motor3.throttle = 0.7
time.sleep(0.15)


# Eind
motor_kit.motor3.throttle = 0.0
car.steering = 0.0

