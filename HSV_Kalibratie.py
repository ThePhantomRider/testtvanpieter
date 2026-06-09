import cv2
import numpy as np
import time

# ==========================
# HSV KALIBRATIE TOOL
# ==========================
# Richt de camera op je lijn, schuif de trackbars tot ALLEEN jouw kleur
# wit oplicht in het maskervenster en de rest zwart blijft.
# Lees dan de waarden af (worden ook in de terminal geprint met 's').
#
# Bediening:
#   s   = print huidige HSV-grenzen in de terminal
#   ESC of q = stoppen
# ==========================

# ==========================
# Camera pipeline (zelfde als racer.py)
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

if not cap.isOpened():
    print("FOUT: camera kon niet geopend worden.")
    print("Check de GStreamer pipeline / camera-aansluiting.")
    raise SystemExit(1)

# ==========================
# Trackbar venster opzetten
# ==========================
WINDOW = "HSV instellingen"
cv2.namedWindow(WINDOW)

# Lege callback (trackbars worden direct uitgelezen, niet via callback)
def nothing(x):
    pass

# OpenCV HSV-schaal: H 0-179, S 0-255, V 0-255
# Startwaarden ruim ingesteld zodat je meteen iets ziet.
cv2.createTrackbar("H min", WINDOW, 0, 179, nothing)
cv2.createTrackbar("H max", WINDOW, 179, 179, nothing)
cv2.createTrackbar("S min", WINDOW, 0, 255, nothing)
cv2.createTrackbar("S max", WINDOW, 255, 255, nothing)
cv2.createTrackbar("V min", WINDOW, 0, 255, nothing)
cv2.createTrackbar("V max", WINDOW, 255, 255, nothing)

# Optioneel: blur aan/uit (1 = aan), zoals in het hoofdscript
cv2.createTrackbar("Blur (0/1)", WINDOW, 1, 1, nothing)

print("Kalibratie gestart.")
print("  s        = print huidige HSV-grenzen")
print("  ESC of q = stoppen")

try:

    while True:

        ret, frame = cap.read()

        if not ret:
            continue

        # Eventueel blurren (zelfde 11x11 als hoofdscript) voor eerlijke vergelijking
        use_blur = cv2.getTrackbarPos("Blur (0/1)", WINDOW)
        if use_blur == 1:
            proc = cv2.GaussianBlur(frame, (11, 11), 0)
        else:
            proc = frame

        # BGR -> HSV
        hsv = cv2.cvtColor(proc, cv2.COLOR_BGR2HSV)

        # Trackbar-waarden uitlezen
        h_min = cv2.getTrackbarPos("H min", WINDOW)
        h_max = cv2.getTrackbarPos("H max", WINDOW)
        s_min = cv2.getTrackbarPos("S min", WINDOW)
        s_max = cv2.getTrackbarPos("S max", WINDOW)
        v_min = cv2.getTrackbarPos("V min", WINDOW)
        v_max = cv2.getTrackbarPos("V max", WINDOW)

        lower = np.array([h_min, s_min, v_min])
        upper = np.array([h_max, s_max, v_max])

        # Masker maken
        mask = cv2.inRange(hsv, lower, upper)

        # Resultaat: originele beeld waar alleen de gevonden kleur zichtbaar is
        result = cv2.bitwise_and(frame, frame, mask=mask)

        # Aantal gevonden pixels (handig om te zien of detectie stabiel is)
        count = cv2.countNonZero(mask)
        cv2.putText(
            result, f"pixels: {count}", (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
        )
        cv2.putText(
            result,
            f"lower=[{h_min},{s_min},{v_min}] upper=[{h_max},{s_max},{v_max}]",
            (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
        )

        # Drie vensters: origineel, masker (zwart/wit), resultaat
        cv2.imshow("Origineel", frame)
        cv2.imshow("Masker", mask)
        cv2.imshow("Resultaat", result)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            # Grenzen netjes printen zodat je ze direct kunt kopieren
            print("\n=== Huidige HSV-grenzen (OpenCV-schaal) ===")
            print(f"lower = np.array([{h_min}, {s_min}, {v_min}])")
            print(f"upper = np.array([{h_max}, {s_max}, {v_max}])")
            print(f"(pixels in masker: {count})\n")

        if key == 27 or key == ord('q'):
            break

except KeyboardInterrupt:
    print("Kalibratie stoppen...")

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("Camera vrijgegeven, kalibratie gestopt.")
