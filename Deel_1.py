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
SHOW_PREVIEW = True         # zet op False voor max snelheid (race mode)

MIN_PIXELS = 400            # minimaal aantal pixels per kleur om mee te tellen (ruisfilter)

# --- Stuur parameters ---
STEER_TURN = 0.6           # stevig sturen in de bocht (gevraagd)
STEER_LIMIT = 0.8          # HARDE servo-limiet! Nooit verder, anders kapot.

# --- Throttle parameters ---
THROTTLE_DRIVE = 1.0        # rechtdoor / geen bocht
THROTTLE_TURN = 0.5         # tijdens sturen langzamer

# --- Bocht-afronding ---
# Als de kleuren uit beeld verdwijnen zit de auto vaak nog midden in de bocht.
# Dan stuurt hij nog TURN_SLEEP seconden door in de laatst geziene richting
# voordat hij teruggaat naar rechtdoor. Pas deze waarde aan om de bocht korter
# of langer door te trekken.
# LET OP: tijdens deze sleep blokkeert de control-thread; de auto verwerkt dan
# even geen nieuwe frames. Houd de waarde daarom klein (orde 0.1 - 0.5 s).
TURN_SLEEP = 0.3

# ==========================
# HSV kleurwaarden (opgemeten met hsv_kalibratie_dubbel.py, Blur=1)
# OpenCV-schaal H 0-179, S/V 0-255
# ==========================
# Donkerblauw - opgemeten op de baan
lower_blue = np.array([92, 22, 135])
upper_blue = np.array([153, 79, 159])

# Oranje - opgemeten op de baan. TWEE bereiken die met OR gecombineerd
# worden: door de fisheye/roze hue ziet de camera oranje op sommige plekken
# net anders. Bereik A pakt de ene situatie, bereik B de andere; samen
# (A OR B) dekken ze alle oranje. Beide opgemeten met de andere kleuren in
# beeld, dus geen overlap met blauw/rood/groen.
lower_orange_a = np.array([0, 58, 166])
upper_orange_a = np.array([65, 140, 206])

lower_orange_b = np.array([0, 28, 124])
upper_orange_b = np.array([155, 255, 144])

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
# Queues voor frames
# ==========================
frame_queue = queue.Queue(maxsize=5)
display_queue = queue.Queue(maxsize=2)   # voor preview, kleine buffer

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
# Helper - gemiddelde y-positie en aantal pixels van een masker
# ==========================
def mask_position(mask):
    """
    Berekent het zwaartepunt (gemiddelde y) en het aantal pixels van een masker.
    Returns (cy, count):
      cy    : gemiddelde y-positie van de gevonden pixels (None als te weinig).
              Grotere y = lager in beeld (OpenCV telt y van boven naar onder).
      count : aantal pixels in het masker.
    """
    count = cv2.countNonZero(mask)

    if count < MIN_PIXELS:
        return None, count

    # Zwaartepunt via moments
    M = cv2.moments(mask, binaryImage=True)

    if M["m00"] == 0:
        return None, count

    cy = M["m01"] / M["m00"]

    return cy, count

# ==========================
# Thread 2 - Beeldverwerking + sturen
# ==========================
def control_thread():

    # Laatst geziene bochtrichting onthouden voor de bocht-afronding.
    # 0.0 = geen bocht (rechtdoor), -STEER_TURN = links, +STEER_TURN = rechts.
    last_steer = 0.0

    while not stop_event.is_set():

        if frame_queue.empty():
            continue

        frame = frame_queue.get()

        h, w = frame.shape[:2]

        # Blur voor stabielere detectie
        blurred = cv2.GaussianBlur(frame, (11, 11), 0)

        # BGR -> HSV
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        # ==========================
        # Maskers maken
        # ==========================
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

        # Oranje uit twee bereiken (A OR B) - zie uitleg bij de HSV-waarden
        mask_orange_a = cv2.inRange(hsv, lower_orange_a, upper_orange_a)
        mask_orange_b = cv2.inRange(hsv, lower_orange_b, upper_orange_b)
        mask_orange = cv2.bitwise_or(mask_orange_a, mask_orange_b)

        # ==========================
        # ROI: bovenste helft van het beeld negeren.
        # Alles boven y = h/2 op zwart zetten zodat alleen de onderste helft
        # meetelt voor de detectie. De y-coordinaten blijven kloppen met het
        # hele beeld, dus de onder/boven-volgorde-logica verandert niet.
        # ==========================
        roi_top = h // 2
        mask_blue[0:roi_top, :] = 0
        mask_orange[0:roi_top, :] = 0

        # ==========================
        # Gemiddelde y-positie per kleur bepalen
        # ==========================
        blue_y, blue_count = mask_position(mask_blue)
        orange_y, orange_count = mask_position(mask_orange)

        # ==========================
        # STUUR LOGICA - volgorde van onder naar boven
        # ==========================
        # In OpenCV is grotere y = lager in beeld (dichter bij de auto).
        #
        # Bocht LINKS  : van onder naar boven eerst BLAUW, dan ORANJE.
        #                blauw zit dus LAGER -> blue_y > orange_y
        # Bocht RECHTS : van onder naar boven eerst ORANJE, dan BLAUW.
        #                oranje zit dus LAGER -> orange_y > blue_y
        # ==========================
        steering = 0.0
        throttle = THROTTLE_DRIVE
        action_label = "GEEN -> RECHTDOOR"

        # Vlag: is er dit frame een geldige bocht-volgorde gezien?
        valid_turn = False

        # Beide kleuren moeten zichtbaar zijn voor een geldige volgorde
        if blue_y is not None and orange_y is not None:

            valid_turn = True

            if blue_y > orange_y:
                # Blauw onder, oranje boven -> bocht naar LINKS
                steering = -STEER_TURN
                throttle = THROTTLE_TURN
                action_label = "BLAUW->ORANJE -> BOCHT LINKS"

            else:
                # Oranje onder, blauw boven -> bocht naar RECHTS
                steering = STEER_TURN
                throttle = THROTTLE_TURN
                action_label = "ORANJE->BLAUW -> BOCHT RECHTS"

            # Richting onthouden voor de afronding straks
            last_steer = steering

        # ==========================
        # HARDE SERVO-LIMIET - nooit verder dan +/- STEER_LIMIT (anders kapot!)
        # ==========================
        steering = max(-STEER_LIMIT, min(STEER_LIMIT, steering))

        # Toepassen
        car.steering = steering
        motor_kit.motor3.throttle = throttle

        # ==========================
        # BOCHT-AFRONDING (sleep)
        # Geen geldige volgorde dit frame, MAAR we waren net in een bocht
        # (last_steer != 0). Dan nog TURN_SLEEP seconden doorsturen in de
        # laatst geziene richting voordat we teruggaan naar rechtdoor.
        # LET OP: tijdens de sleep blokkeert deze thread (auto verwerkt dan
        # geen nieuwe frames). Daarom TURN_SLEEP klein houden.
        # ==========================
        if not valid_turn and last_steer != 0.0:

            finish_steer = max(-STEER_LIMIT, min(STEER_LIMIT, last_steer))
            car.steering = finish_steer
            motor_kit.motor3.throttle = THROTTLE_TURN
            action_label = "BOCHT AFRONDEN (sleep)"

            time.sleep(TURN_SLEEP)

            # Bocht is afgerond -> terug naar rechtdoor en richting vergeten
            car.steering = 0.0
            motor_kit.motor3.throttle = THROTTLE_DRIVE
            steering = 0.0
            throttle = THROTTLE_DRIVE
            last_steer = 0.0

        # ==========================
        # Preview frame voorbereiden (alleen als SHOW_PREVIEW aan)
        # ==========================
        if SHOW_PREVIEW:

            # Genegeerde bovenhelft donkerder maken + grens aangeven
            frame[0:roi_top, :] = (frame[0:roi_top, :] * 0.4).astype(np.uint8)
            cv2.line(frame, (0, roi_top), (w, roi_top), (0, 255, 255), 2)
            cv2.putText(
                frame, "genegeerd", (10, roi_top - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1
            )

            # Blauwe pixels markeren (blauwe overlay waar gedetecteerd)
            frame[mask_blue > 0] = (255, 0, 0)
            # Oranje pixels markeren (oranje overlay waar gedetecteerd)
            frame[mask_orange > 0] = (0, 165, 255)

            # Horizontale lijn op gemiddelde y van blauw
            if blue_y is not None:
                yb = int(blue_y)
                cv2.line(frame, (0, yb), (w, yb), (255, 0, 0), 2)
                cv2.putText(
                    frame, f"BLAUW y={yb} ({blue_count})", (10, yb - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2
                )

            # Horizontale lijn op gemiddelde y van oranje
            if orange_y is not None:
                yo = int(orange_y)
                cv2.line(frame, (0, yo), (w, yo), (0, 165, 255), 2)
                cv2.putText(
                    frame, f"ORANJE y={yo} ({orange_count})", (10, yo + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2
                )

            # Actie + stuurinfo bovenaan
            cv2.putText(
                frame, action_label, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
            )
            cv2.putText(
                frame,
                f"stuur={steering:+.2f} gas={throttle:.2f} (limiet +/-{STEER_LIMIT})",
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
            )

            # Naar display queue sturen (oude weg als vol)
            if display_queue.full():
                try:
                    display_queue.get_nowait()
                except queue.Empty:
                    pass

            display_queue.put(frame)

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
# Hoofdprogramma - toont preview op main thread
# ==========================
try:

    while not stop_event.is_set():

        if SHOW_PREVIEW:

            # Frame uit display queue halen en tonen
            if not display_queue.empty():
                preview = display_queue.get()
                cv2.imshow("Racer view", preview)

            # ESC of 'q' om te stoppen
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                stop_event.set()
                break

        else:
            # Geen preview -> gewoon wachten
            time.sleep(0.1)

except KeyboardInterrupt:

    print("Programma stoppen...")
    stop_event.set()

finally:

    stop_event.set()
    t1.join()
    t2.join()

    # ==========================
    # Alles veilig stoppen
    # ==========================
    try:
        motor_kit.motor3.throttle = 0.0
        car.steering = 0.0
    except:
        pass

    cap.release()

    if SHOW_PREVIEW:
        cv2.destroyAllWindows()

    print("Motoren gestopt")
