import cv2
import numpy as np
import time

# ==========================
# HSV KALIBRATIE TOOL - DUBBEL BEREIK
# ==========================
# Twee volledige HSV-bereiken (A en B), elk met eigen min/max voor H, S en V.
# Het eindmasker = A OR B. Elk bereik heeft een eigen aan/uit-schuif.
#
# Waarom twee bereiken:
#   - ROOD wrapt rond de Hue-grens (laag bij 0, hoog bij 179). Zet range A op
#     0-10 en range B op 165-179 om beide kanten te vangen.
#   - LICHT vs DONKERBLAUW: zet A op de ene tint en B op de andere, en zie in
#     het maskervenster of ze overlappen of netjes gescheiden zijn.
#
# Voor een simpele kleur: zet range B uit ("B aan (0/1)" = 0), dan werkt het
# precies als een enkel bereik.
#
# In het resultaatvenster:
#   - pixels die ALLEEN door A worden gevonden  -> groen gemarkeerd
#   - pixels die ALLEEN door B worden gevonden  -> rood gemarkeerd
#   - pixels die door BEIDE worden gevonden     -> wit gemarkeerd (overlap!)
#
# Bediening:
#   s        = print huidige HSV-grenzen in de terminal
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

def nothing(x):
    pass

# OpenCV HSV-schaal: H 0-179, S 0-255, V 0-255

# --- Range A ---
cv2.createTrackbar("A H min", WINDOW, 0, 179, nothing)
cv2.createTrackbar("A H max", WINDOW, 179, 179, nothing)
cv2.createTrackbar("A S min", WINDOW, 0, 255, nothing)
cv2.createTrackbar("A S max", WINDOW, 255, 255, nothing)
cv2.createTrackbar("A V min", WINDOW, 0, 255, nothing)
cv2.createTrackbar("A V max", WINDOW, 255, 255, nothing)

# --- Range B ---
cv2.createTrackbar("B H min", WINDOW, 0, 179, nothing)
cv2.createTrackbar("B H max", WINDOW, 179, 179, nothing)
cv2.createTrackbar("B S min", WINDOW, 0, 255, nothing)
cv2.createTrackbar("B S max", WINDOW, 255, 255, nothing)
cv2.createTrackbar("B V min", WINDOW, 0, 255, nothing)
cv2.createTrackbar("B V max", WINDOW, 255, 255, nothing)

# --- Schakelaars ---
cv2.createTrackbar("B aan (0/1)", WINDOW, 0, 1, nothing)   # range B standaard UIT
cv2.createTrackbar("Blur (0/1)", WINDOW, 1, 1, nothing)    # blur standaard AAN

print("Kalibratie gestart (dubbel bereik).")
print("  s        = print huidige HSV-grenzen")
print("  ESC of q = stoppen")

def read_range(prefix):
    """Leest de zes trackbar-waarden van een bereik (A of B) uit."""
    h_min = cv2.getTrackbarPos(f"{prefix} H min", WINDOW)
    h_max = cv2.getTrackbarPos(f"{prefix} H max", WINDOW)
    s_min = cv2.getTrackbarPos(f"{prefix} S min", WINDOW)
    s_max = cv2.getTrackbarPos(f"{prefix} S max", WINDOW)
    v_min = cv2.getTrackbarPos(f"{prefix} V min", WINDOW)
    v_max = cv2.getTrackbarPos(f"{prefix} V max", WINDOW)
    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])
    return lower, upper

try:

    while True:

        ret, frame = cap.read()

        if not ret:
            continue

        # Eventueel blurren (zelfde 11x11 als hoofdscript)
        use_blur = cv2.getTrackbarPos("Blur (0/1)", WINDOW)
        if use_blur == 1:
            proc = cv2.GaussianBlur(frame, (11, 11), 0)
        else:
            proc = frame

        hsv = cv2.cvtColor(proc, cv2.COLOR_BGR2HSV)

        b_on = cv2.getTrackbarPos("B aan (0/1)", WINDOW)

        # Maskers per bereik
        lowerA, upperA = read_range("A")
        maskA = cv2.inRange(hsv, lowerA, upperA)

        if b_on == 1:
            lowerB, upperB = read_range("B")
            maskB = cv2.inRange(hsv, lowerB, upperB)
        else:
            maskB = np.zeros_like(maskA)

        # Gecombineerd masker (A OR B) - dit is wat je straks in het hoofdscript gebruikt
        mask = cv2.bitwise_or(maskA, maskB)

        # Resultaat opbouwen met kleurcodering:
        #   alleen A  -> groen
        #   alleen B  -> rood
        #   beide     -> wit (overlap, handig om te zien of A en B botsen)
        result = np.zeros_like(frame)
        only_a = cv2.bitwise_and(maskA, cv2.bitwise_not(maskB))
        only_b = cv2.bitwise_and(maskB, cv2.bitwise_not(maskA))
        both = cv2.bitwise_and(maskA, maskB)

        result[only_a > 0] = (0, 255, 0)      # groen
        result[only_b > 0] = (0, 0, 255)      # rood
        result[both > 0] = (255, 255, 255)    # wit

        countA = cv2.countNonZero(maskA)
        countB = cv2.countNonZero(maskB)
        count_both = cv2.countNonZero(both)

        cv2.putText(
            result, f"A: {countA}  B: {countB}  overlap: {count_both}",
            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
        )
        cv2.putText(
            result, "groen=alleen A  rood=alleen B  wit=overlap",
            (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1
        )

        cv2.imshow("Origineel", frame)
        cv2.imshow("Masker (A OR B)", mask)
        cv2.imshow("Resultaat", result)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            print("\n=== Huidige HSV-grenzen (OpenCV-schaal) ===")
            print(f"# Range A  (pixels: {countA})")
            print(f"lower_A = np.array([{lowerA[0]}, {lowerA[1]}, {lowerA[2]}])")
            print(f"upper_A = np.array([{upperA[0]}, {upperA[1]}, {upperA[2]}])")
            if b_on == 1:
                print(f"# Range B  (pixels: {countB})")
                print(f"lower_B = np.array([{lowerB[0]}, {lowerB[1]}, {lowerB[2]}])")
                print(f"upper_B = np.array([{upperB[0]}, {upperB[1]}, {upperB[2]}])")
                print("# Gebruik: mask = cv2.inRange(hsv, lower_A, upper_A) + "
                      "cv2.inRange(hsv, lower_B, upper_B)")
            else:
                print("# Range B staat UIT -> enkel bereik")
                print("# Gebruik: mask = cv2.inRange(hsv, lower_A, upper_A)")
            print(f"# overlap tussen A en B: {count_both} pixels\n")

        if key == 27 or key == ord('q'):
            break

except KeyboardInterrupt:
    print("Kalibratie stoppen...")

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("Camera vrijgegeven, kalibratie gestopt.")
