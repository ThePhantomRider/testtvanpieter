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

MIN_PIXELS = 400            # minimaal TOTAAL aantal pixels per kleur (grove ruisfilter)

# Minimale grootte (in pixels) van een aaneengesloten cluster om mee te tellen.
# Losse vlekjes kleiner dan dit worden weggegooid VOOR de lijn-fit, zodat een
# paar afdwalende oranje pixels de lijn niet meer scheeftrekken. Verhoog dit
# als er nog kleine vlekjes meedoen; verlaag het als echte lijnstukken wegvallen.
MIN_CLUSTER_AREA = 150

# --- Stuur parameters ---
# De stuursterkte schaalt mee met het hoekverschil tussen de twee lijnen.
# Klein hoekverschil -> zacht sturen, groot hoekverschil -> tot STEER_MAX.
STEER_MIN = 0.45          # minimale stuurwaarde zodra er een geldige bocht is
STEER_MAX = 0.75          # maximale stuurwaarde (bij groot hoekverschil)
STEER_LIMIT = 0.8         # HARDE servo-limiet! Nooit verder, anders kapot.

# Hoekverschil (in graden) waarbij we STEER_MAX bereiken. Een verschil van
# >= ANGLE_FULL graden tussen de twee lijnen geeft vol sturen; daaronder
# schaalt het lineair tussen STEER_MIN en STEER_MAX.
ANGLE_FULL = 40.0

# Hoekverschil (in graden) waaronder we het als "geen duidelijke bocht" zien
# (de twee lijnen lopen dan bijna parallel). Anti-trillen / deadzone.
ANGLE_DEADZONE = 5.0

# Teken-omkering: als links en rechts omgedraaid zijn, zet dit op -1.
# (De juiste waarde hangt af van hoe de lijnen in jouw camerabeeld liggen;
#  controleer met de preview en draai dit om als het de verkeerde kant op stuurt.)
TURN_SIGN = 1.0

# --- Throttle parameters ---
THROTTLE_DRIVE = 1.0        # rechtdoor / geen bocht
THROTTLE_TURN = 0.5         # tijdens sturen langzamer

# --- Bocht-afronding ---
# Als de kleuren uit beeld verdwijnen zit de auto vaak nog midden in de bocht.
# Dan stuurt hij nog TURN_SLEEP seconden door in de laatst geziene richting
# voordat hij teruggaat naar rechtdoor.
# LET OP: tijdens deze sleep blokkeert de control-thread; de auto verwerkt dan
# even geen nieuwe frames. Houd de waarde daarom klein (orde 0.1 - 0.5 s).
TURN_SLEEP = 0.3

# ==========================
# HSV kleurwaarden (opgemeten met hsv_kalibratie_dubbel.py, Blur=1)
# OpenCV-schaal H 0-179, S/V 0-255
# ==========================
# Donkerblauw
lower_blue_a = np.array([93, 27, 109])
upper_blue_a = np.array([166, 100, 140])

# Oranje - drie bereiken (A OR B OR C) wegens fisheye-kleurverschuiving
lower_orange_a = np.array([168, 77, 144])
upper_orange_a = np.array([179, 107, 187])

lower_orange_b = np.array([168, 0, 147])
upper_orange_b = np.array([179, 48, 168])

lower_orange_c = np.array([58, 0, 144])
upper_orange_c = np.array([83, 82, 151])

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
# Helper - kleine clusters uit een masker filteren
# ==========================
def filter_small_clusters(mask, min_area):
    """
    Houdt alleen aaneengesloten clusters (componenten) die groter dan of
    gelijk aan min_area pixels zijn. Kleinere vlekjes worden zwart gemaakt.

    Zo tellen losse afdwalende pixels niet meer mee bij de lijn-fit, terwijl
    een echt lijnstuk (groot, aaneengesloten) gewoon blijft staan.
    """
    # connectedComponentsWithStats geeft per component een label + statistieken.
    # stats[label, cv2.CC_STAT_AREA] = aantal pixels in dat component.
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    # Nieuw, leeg masker opbouwen met alleen de grote componenten.
    # Label 0 is altijd de achtergrond, die slaan we over.
    cleaned = np.zeros_like(mask)
    for label in range(1, num):
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == label] = 255

    return cleaned

# ==========================
# Helper - lijn fitten door de pixels van een masker + hoek bepalen
# ==========================
def line_angle(mask):
    """
    Fit een rechte lijn door alle witte pixels van het masker en bepaalt
    de hoek ten opzichte van de horizontale onderrand.

    Returns (angle_deg, count, fit):
      angle_deg : hoek van de lijn in graden, genormaliseerd naar [0, 180).
                  0 = horizontaal, 90 = verticaal. None als te weinig pixels.
      count     : aantal pixels in het masker.
      fit       : (vx, vy, x0, y0) van cv2.fitLine voor het tekenen in preview,
                  of None.

    Een lijn heeft geen richting/kant: 10 graden en 190 graden zijn dezelfde
    lijn. Daarom normaliseren we naar het bereik [0, 180).
    """
    count = cv2.countNonZero(mask)

    if count < MIN_PIXELS:
        return None, count, None

    # Coordinaten van alle witte pixels ophalen.
    # findNonZero geeft (N, 1, 2) met (x, y) per punt.
    pts = cv2.findNonZero(mask)

    if pts is None or len(pts) < 2:
        return None, count, None

    # Lijn fitten (kleinste-kwadraten, robuust met DIST_L2).
    vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).flatten()

    # Hoek t.o.v. de horizontale as via atan2. Resultaat in graden.
    angle = np.degrees(np.arctan2(vy, vx))

    # Normaliseren naar [0, 180) want de lijn heeft geen kant.
    angle = angle % 180.0

    return angle, count, (vx, vy, x0, y0)

# ==========================
# Helper - kleinste hoekverschil tussen twee lijn-hoeken (met teken)
# ==========================
def signed_angle_diff(a_blue, a_orange):
    """
    Verschil tussen de blauwe en oranje lijnhoek, in graden, in (-90, 90].

    Omdat lijnhoeken in [0, 180) liggen en een lijn geen kant heeft, is het
    grootste zinvolle verschil 90 graden. We vouwen het verschil daarom terug
    naar (-90, 90]. Het TEKEN vertelt welke kant de wig opent (links/rechts).
    """
    diff = a_blue - a_orange

    # Terugvouwen naar (-90, 90]
    while diff > 90.0:
        diff -= 180.0
    while diff <= -90.0:
        diff += 180.0

    return diff

# ==========================
# Thread 2 - Beeldverwerking + sturen
# ==========================
def control_thread():

    # Laatst geziene stuurwaarde onthouden voor de bocht-afronding.
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
        mask_blue = cv2.inRange(hsv, lower_blue_a, upper_blue_a)

        # Oranje uit drie bereiken (A OR B OR C). bitwise_or neemt steeds maar
        # twee maskers, dus in twee stappen.
        mask_orange_a = cv2.inRange(hsv, lower_orange_a, upper_orange_a)
        mask_orange_b = cv2.inRange(hsv, lower_orange_b, upper_orange_b)
        mask_orange_c = cv2.inRange(hsv, lower_orange_c, upper_orange_c)
        mask_orange = cv2.bitwise_or(mask_orange_a, mask_orange_b)
        mask_orange = cv2.bitwise_or(mask_orange, mask_orange_c)

        # ==========================
        # ROI: bovenste helft negeren (alleen onderste helft telt mee)
        # ==========================
        roi_top = h // 2
        mask_blue[0:roi_top, :] = 0
        mask_orange[0:roi_top, :] = 0

        # ==========================
        # Kleine clusters wegfilteren: losse vlekjes kleiner dan
        # MIN_CLUSTER_AREA verdwijnen, zodat ze de lijn-fit niet scheeftrekken.
        # ==========================
        mask_blue = filter_small_clusters(mask_blue, MIN_CLUSTER_AREA)
        mask_orange = filter_small_clusters(mask_orange, MIN_CLUSTER_AREA)

        # ==========================
        # Lijn fitten + hoek bepalen per kleur
        # ==========================
        blue_angle, blue_count, blue_fit = line_angle(mask_blue)
        orange_angle, orange_count, orange_fit = line_angle(mask_orange)

        # ==========================
        # STUUR LOGICA - hoek tussen de twee lijnen (wig)
        # ==========================
        steering = 0.0
        throttle = THROTTLE_DRIVE
        action_label = "GEEN -> RECHTDOOR"
        diff = 0.0

        valid_turn = False

        # Beide lijnen moeten gevonden zijn voor een geldige hoekmeting
        if blue_angle is not None and orange_angle is not None:

            # Verschil tussen de twee lijnhoeken, met teken (welke kant opent de wig)
            diff = signed_angle_diff(blue_angle, orange_angle)

            # Alleen sturen als het verschil buiten de deadzone valt
            # (anders lopen de lijnen bijna parallel = geen duidelijke bocht)
            if abs(diff) >= ANGLE_DEADZONE:

                valid_turn = True

                # Sterkte schaalt lineair met het hoekverschil:
                #   |diff| = ANGLE_DEADZONE -> STEER_MIN
                #   |diff| >= ANGLE_FULL    -> STEER_MAX
                span = max(ANGLE_FULL - ANGLE_DEADZONE, 0.001)
                frac = (abs(diff) - ANGLE_DEADZONE) / span
                frac = max(0.0, min(1.0, frac))
                magnitude = STEER_MIN + frac * (STEER_MAX - STEER_MIN)

                # Richting: teken van diff, eventueel omgedraaid met TURN_SIGN
                direction = 1.0 if diff > 0 else -1.0
                steering = TURN_SIGN * direction * magnitude
                throttle = THROTTLE_TURN

                if steering > 0:
                    action_label = f"WIG -> BOCHT RECHTS ({diff:+.0f} graden)"
                else:
                    action_label = f"WIG -> BOCHT LINKS ({diff:+.0f} graden)"

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
        # ==========================
        if not valid_turn and last_steer != 0.0:

            finish_steer = max(-STEER_LIMIT, min(STEER_LIMIT, last_steer))
            car.steering = finish_steer
            motor_kit.motor3.throttle = THROTTLE_TURN
            action_label = "BOCHT AFRONDEN (sleep)"

            time.sleep(TURN_SLEEP)

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

            # Gedetecteerde pixels markeren
            frame[mask_blue > 0] = (255, 0, 0)
            frame[mask_orange > 0] = (0, 165, 255)

            # Gefitte lijnen doortrekken over de volle breedte
            def draw_fit(fit, color, label, count, angle):
                if fit is None:
                    return
                vx, vy, x0, y0 = fit
                # Lijn ver doortrekken in beide richtingen
                t = 2000
                p1 = (int(x0 - vx * t), int(y0 - vy * t))
                p2 = (int(x0 + vx * t), int(y0 + vy * t))
                cv2.line(frame, p1, p2, color, 2)
                cv2.putText(
                    frame, f"{label} {angle:.0f}deg ({count})",
                    (10, int(y0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
                )

            draw_fit(blue_fit, (255, 0, 0), "BLAUW", blue_count,
                     blue_angle if blue_angle is not None else 0)
            draw_fit(orange_fit, (0, 165, 255), "ORANJE", orange_count,
                     orange_angle if orange_angle is not None else 0)

            # Actie + stuurinfo bovenaan
            cv2.putText(
                frame, action_label, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
            )
            cv2.putText(
                frame,
                f"diff={diff:+.1f}deg stuur={steering:+.2f} gas={throttle:.2f} "
                f"(limiet +/-{STEER_LIMIT})",
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
            )

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

            if not display_queue.empty():
                preview = display_queue.get()
                cv2.imshow("Racer view", preview)

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                stop_event.set()
                break

        else:
            time.sleep(0.1)

except KeyboardInterrupt:

    print("Programma stoppen...")
    stop_event.set()

finally:

    stop_event.set()
    t1.join()
    t2.join()

    try:
        motor_kit.motor3.throttle = 0.0
        car.steering = 0.0
    except:
        pass

    cap.release()

    if SHOW_PREVIEW:
        cv2.destroyAllWindows()

    print("Motoren gestopt")
