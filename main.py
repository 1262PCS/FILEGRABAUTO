
import cv2
import mediapipe as mp
import math
import time
import pyautogui
from pathlib import Path


# ==========================================
# 1. CONFIGURATION
# ==========================================

MODEL_PATH = Path(__file__).resolve().parent / "hand_landmarker.task"

# Pinch detection thresholds
GRAB_THRESHOLD = 0.35
RELEASE_THRESHOLD = 0.50

# Gesture confirmation times (seconds)
PINCH_CONFIRM_TIME = 0.15
RELEASE_CONFIRM_TIME = 0.15

# Improved cursor smoothing
SMOOTHING_ALPHA = 0.35
DEAD_ZONE = 5
MAX_STEP = 100

# Camera calibration
CAMERA_MARGIN = 0.15
SCREEN_PADDING = 10

# Mouse safety
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0

SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()


# ==========================================
# 2. MEDIAPIPE CONFIGURATION
# ==========================================

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=str(MODEL_PATH)
    ),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1
)

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]


# ==========================================
# 3. PINCH DISTANCE
# ==========================================

def calculate_pinch_distance(hand):
    thumb = hand[4]
    index = hand[8]

    finger_distance = math.hypot(
        thumb.x - index.x,
        thumb.y - index.y
    )

    wrist = hand[0]
    middle_knuckle = hand[9]

    palm_size = math.hypot(
        wrist.x - middle_knuckle.x,
        wrist.y - middle_knuckle.y
    )

    if palm_size < 0.0001:
        return float("inf")

    return finger_distance / palm_size


# ==========================================
# 4. CAMERA TO SCREEN MAPPING
# ==========================================

def map_to_screen(finger_x, finger_y):
    """
    Map normalized camera coordinates
    to Windows screen coordinates.
    """

    x = (finger_x - CAMERA_MARGIN) / (
        1 - 2 * CAMERA_MARGIN
    )

    y = (finger_y - CAMERA_MARGIN) / (
        1 - 2 * CAMERA_MARGIN
    )

    x = max(0.0, min(1.0, x))
    y = max(0.0, min(1.0, y))

    screen_x = SCREEN_PADDING + x * (
        SCREEN_WIDTH - 2 * SCREEN_PADDING
    )

    screen_y = SCREEN_PADDING + y * (
        SCREEN_HEIGHT - 2 * SCREEN_PADDING
    )

    return screen_x, screen_y


# ==========================================
# 5. IMPROVED CURSOR SMOOTHING
# ==========================================

def smooth_cursor(
    target_x,
    target_y,
    previous_x,
    previous_y
):
    """
    Smooth cursor movement while maintaining
    responsiveness.
    """

    if previous_x is None or previous_y is None:
        return target_x, target_y

    dx = target_x - previous_x
    dy = target_y - previous_y

    distance = math.hypot(dx, dy)

    # Ignore tiny movements
    if distance < DEAD_ZONE:
        return previous_x, previous_y

    # Limit sudden jumps
    if distance > MAX_STEP:
        scale = MAX_STEP / distance
        dx *= scale
        dy *= scale

    smooth_x = previous_x + SMOOTHING_ALPHA * dx
    smooth_y = previous_y + SMOOTHING_ALPHA * dy

    return smooth_x, smooth_y


# ==========================================
# 6. DRAW HAND LANDMARKS
# ==========================================

def draw_hand(frame, hand):
    height, width, _ = frame.shape
    points = []

    for landmark in hand:
        x = int(landmark.x * width)
        y = int(landmark.y * height)

        points.append((x, y))

        cv2.circle(
            frame,
            (x, y),
            5,
            (0, 255, 0),
            -1
        )

    for start, end in HAND_CONNECTIONS:
        cv2.line(
            frame,
            points[start],
            points[end],
            (255, 0, 0),
            2
        )


# ==========================================
# 7. STATUS DISPLAY
# ==========================================

def draw_status(
    frame,
    gesture,
    distance,
    cursor_enabled,
    mouse_is_down,
    cursor_position
):
    height, width, _ = frame.shape

    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (10, 10),
        (min(width - 10, 450), 220),
        (30, 30, 30),
        -1
    )

    cv2.addWeighted(
        overlay,
        0.7,
        frame,
        0.3,
        0,
        frame
    )

    gesture_color = (
        (0, 255, 0)
        if mouse_is_down
        else (255, 255, 255)
    )

    cv2.putText(
        frame,
        gesture,
        (25, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        gesture_color,
        2
    )

    distance_text = (
        f"Pinch distance: {distance:.2f}"
        if distance is not None
        else "Pinch distance: --"
    )

    cv2.putText(
        frame,
        distance_text,
        (25, 85),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cursor_text = (
        "Cursor: ON"
        if cursor_enabled
        else "Cursor: OFF"
    )

    cv2.putText(
        frame,
        cursor_text,
        (25, 125),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0)
        if cursor_enabled
        else (0, 165, 255),
        2
    )

    mouse_text = (
        "Mouse: DOWN"
        if mouse_is_down
        else "Mouse: UP"
    )

    cv2.putText(
        frame,
        mouse_text,
        (25, 165),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0)
        if mouse_is_down
        else (255, 255, 255),
        2
    )

    position_text = (
        f"Screen: {cursor_position}"
        if cursor_position is not None
        else "Screen: --"
    )

    cv2.putText(
        frame,
        position_text,
        (25, 205),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "M: Toggle cursor | Q: Quit",
        (15, height - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )


# ==========================================
# 8. MAIN APPLICATION
# ==========================================

def main():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"MediaPipe model not found: {MODEL_PATH}"
        )

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        raise RuntimeError("Could not open webcam")

    # Gesture state
    is_holding = False

    # Mouse state
    mouse_is_down = False

    # Cursor control starts OFF
    cursor_enabled = False

    # Smoothed coordinates
    smooth_x = None
    smooth_y = None

    # Gesture confirmation timers
    pinch_started_at = None
    release_started_at = None

    # MediaPipe timestamps
    start_time = time.monotonic()
    last_timestamp_ms = -1

    print("=" * 45)
    print("GESTURE FILE TRANSFER - PHASE 2.3")
    print("=" * 45)
    print("M = Enable / Disable cursor")
    print("Q = Quit")
    print("Pinch = Hold left mouse button")
    print("Release = Release left mouse button")
    print("Tracking landmark: Index knuckle (5)")
    print("=" * 45)

    try:
        with HandLandmarker.create_from_options(
            options
        ) as landmarker:

            while True:
                success, frame = camera.read()

                if not success:
                    print("Failed to capture frame")
                    break

                frame = cv2.flip(frame, 1)

                height, width, _ = frame.shape

                # Convert BGR to RGB
                rgb_frame = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )

                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb_frame
                )

                # Increasing timestamps
                timestamp_ms = max(
                    int(
                        (time.monotonic() - start_time)
                        * 1000
                    ),
                    last_timestamp_ms + 1
                )

                last_timestamp_ms = timestamp_ms

                result = landmarker.detect_for_video(
                    mp_image,
                    timestamp_ms
                )

                gesture = "NO HAND DETECTED"
                distance = None
                cursor_position = None

                # ==================================
                # HAND DETECTED
                # ==================================

                if result.hand_landmarks:
                    hand = result.hand_landmarks[0]

                    draw_hand(frame, hand)

                    distance = calculate_pinch_distance(
                        hand
                    )

                    # Gesture hysteresis
                    if (
                        not is_holding
                        and distance < GRAB_THRESHOLD
                    ):
                        is_holding = True

                    elif (
                        is_holding
                        and distance > RELEASE_THRESHOLD
                    ):
                        is_holding = False

                    gesture = (
                        "GRABBING"
                        if is_holding
                        else "RELEASED"
                    )

                    # ==================================
                    # INDEX KNUCKLE TRACKING
                    # ==================================

                    # Landmark 5 is more stable during pinch
                    pointer = hand[5]

                    target_x, target_y = map_to_screen(
                        pointer.x,
                        pointer.y
                    )

                    smooth_x, smooth_y = smooth_cursor(
                        target_x,
                        target_y,
                        smooth_x,
                        smooth_y
                    )

                    cursor_x = int(smooth_x)
                    cursor_y = int(smooth_y)

                    cursor_position = (
                        cursor_x,
                        cursor_y
                    )

                    # Draw yellow circle on tracked knuckle
                    cv2.circle(
                        frame,
                        (
                            int(pointer.x * width),
                            int(pointer.y * height)
                        ),
                        12,
                        (0, 255, 255),
                        3
                    )

                    # ==================================
                    # WINDOWS MOUSE CONTROL
                    # ==================================

                    if cursor_enabled:
                        try:
                            pyautogui.moveTo(
                                cursor_x,
                                cursor_y,
                                duration=0
                            )

                            now = time.monotonic()

                            # Pinch -> mouseDown
                            if is_holding:
                                release_started_at = None

                                if pinch_started_at is None:
                                    pinch_started_at = now

                                if (
                                    not mouse_is_down
                                    and now - pinch_started_at
                                    >= PINCH_CONFIRM_TIME
                                ):
                                    pyautogui.mouseDown(
                                        button="left"
                                    )

                                    mouse_is_down = True
                                    print("Mouse DOWN")

                            # Release -> mouseUp
                            else:
                                pinch_started_at = None

                                if release_started_at is None:
                                    release_started_at = now

                                if (
                                    mouse_is_down
                                    and now - release_started_at
                                    >= RELEASE_CONFIRM_TIME
                                ):
                                    pyautogui.mouseUp(
                                        button="left"
                                    )

                                    mouse_is_down = False
                                    print("Mouse UP")

                        except pyautogui.FailSafeException:
                            print("Emergency stop triggered.")
                            break

                # ==================================
                # HAND LOST
                # ==================================

                else:
                    is_holding = False

                    smooth_x = None
                    smooth_y = None

                    pinch_started_at = None
                    release_started_at = None

                    if mouse_is_down:
                        pyautogui.mouseUp(button="left")
                        mouse_is_down = False
                        print("Mouse UP - hand lost")

                # ==================================
                # DISPLAY
                # ==================================

                draw_status(
                    frame,
                    gesture,
                    distance,
                    cursor_enabled,
                    mouse_is_down,
                    cursor_position
                )

                cv2.imshow(
                    "Gesture File Transfer - Phase 2.3",
                    frame
                )

                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    break

                elif key == ord("m"):
                    cursor_enabled = not cursor_enabled

                    smooth_x = None
                    smooth_y = None

                    pinch_started_at = None
                    release_started_at = None

                    if mouse_is_down:
                        pyautogui.mouseUp(button="left")
                        mouse_is_down = False

                    print(
                        "Cursor control:",
                        "ON"
                        if cursor_enabled
                        else "OFF"
                    )

    except KeyboardInterrupt:
        print("Stopped by user.")

    except pyautogui.FailSafeException:
        print("PyAutoGUI emergency stop.")

    finally:
        if mouse_is_down:
            try:
                pyautogui.mouseUp(button="left")
            except pyautogui.FailSafeException:
                print(
                    "Move physical mouse away from corner "
                    "to allow mouse release."
                )

        camera.release()
        cv2.destroyAllWindows()

        print("Application closed safely.")


if __name__ == "__main__":
    main()
