"""GestureFileTransfer: stabilized pinch-to-drag mouse control."""
import math
import time
from pathlib import Path

import cv2
import mediapipe as mp
import pyautogui

MODEL_PATH = Path(__file__).resolve().parent / "hand_landmarker.task"
GRAB_THRESHOLD = 0.35
RELEASE_THRESHOLD = 0.50
PINCH_CONFIRM_TIME = 0.15
RELEASE_CONFIRM_TIME = 0.20
CAMERA_MARGIN = 0.15
SCREEN_PADDING = 10

# Cursor tuning
STILL_ZONE = 7           # Ignore camera jitter smaller than 7 screen pixels
SLOW_ALPHA = 0.16        # More filtering for fine movements
FAST_ALPHA = 0.55        # Less filtering for larger movements
FAST_DISTANCE = 65      # Switch to fast response above this distance
DRAG_START_DISTANCE = 18  # Hand must move this far after mouseDown
MAX_TARGET_JUMP = 240    # Ignore single-frame tracking glitches

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0
SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
OPTIONS = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1,
)
CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]


def pinch_distance(hand):
    thumb, index = hand[4], hand[8]
    wrist, knuckle = hand[0], hand[9]
    palm = math.hypot(wrist.x - knuckle.x, wrist.y - knuckle.y)
    if palm < 0.0001:
        return float("inf")
    return math.hypot(thumb.x - index.x, thumb.y - index.y) / palm


def map_to_screen(x, y):
    x = max(0.0, min(1.0, (x - CAMERA_MARGIN) / (1 - 2 * CAMERA_MARGIN)))
    y = max(0.0, min(1.0, (y - CAMERA_MARGIN) / (1 - 2 * CAMERA_MARGIN)))
    return (
        SCREEN_PADDING + x * (SCREEN_WIDTH - 2 * SCREEN_PADDING),
        SCREEN_PADDING + y * (SCREEN_HEIGHT - 2 * SCREEN_PADDING),
    )


def smooth_cursor(target, previous):
    if previous is None:
        return target
    dx, dy = target[0] - previous[0], target[1] - previous[1]
    distance = math.hypot(dx, dy)
    if distance < STILL_ZONE:
        return previous
    if distance > MAX_TARGET_JUMP:
        # Cap sudden tracking glitches without freezing ordinary fast motion.
        scale = MAX_TARGET_JUMP / distance
        dx, dy = dx * scale, dy * scale
        distance = MAX_TARGET_JUMP
    blend = min(1.0, distance / FAST_DISTANCE)
    alpha = SLOW_ALPHA + (FAST_ALPHA - SLOW_ALPHA) * blend
    return previous[0] + alpha * dx, previous[1] + alpha * dy


def draw_hand(frame, hand):
    height, width = frame.shape[:2]
    points = [(int(p.x * width), int(p.y * height)) for p in hand]
    for start, end in CONNECTIONS:
        cv2.line(frame, points[start], points[end], (255, 0, 0), 2)
    for point in points:
        cv2.circle(frame, point, 4, (0, 255, 0), -1)
    cv2.circle(frame, points[5], 11, (0, 255, 255), 2)


def draw_status(frame, state, enabled, mouse_down, distance, position):
    height, width = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (8, 8), (min(width - 8, 510), 208), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)
    lines = [
        f"Gesture: {state}",
        f"Cursor: {'ON' if enabled else 'OFF'}",
        f"Mouse: {'DOWN' if mouse_down else 'UP'}",
        f"Pinch distance: {distance:.2f}" if distance is not None else "Pinch distance: --",
        f"Screen: {position}" if position is not None else "Screen: --",
        "M: Toggle cursor  |  Q: Quit",
    ]
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (20, 38 + i * 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.56, (255, 255, 255), 2)


def main():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Missing model: {MODEL_PATH}")
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise RuntimeError("Cannot open webcam")

    cursor_enabled = False
    mouse_down = False
    smooth_position = None
    pinch_since = None
    release_since = None
    pinch_anchor = None
    drag_started = False
    state = "NO HAND"
    start_time = time.monotonic()
    last_timestamp = -1

    print("GestureFileTransfer - stabilized mouse drag")
    print("M = toggle cursor; Q = quit")
    try:
        with HandLandmarker.create_from_options(OPTIONS) as detector:
            while True:
                ok, frame = camera.read()
                if not ok:
                    print("Camera frame unavailable")
                    break
                frame = cv2.flip(frame, 1)
                now = time.monotonic()
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                timestamp = max(int((now - start_time) * 1000), last_timestamp + 1)
                last_timestamp = timestamp
                result = detector.detect_for_video(image, timestamp)
                distance = None
                position = None

                if result.hand_landmarks:
                    hand = result.hand_landmarks[0]
                    draw_hand(frame, hand)
                    distance = pinch_distance(hand)
                    pointer = hand[5]
                    target = map_to_screen(pointer.x, pointer.y)
                    smooth_position = smooth_cursor(target, smooth_position)

                    if cursor_enabled:
                        if not mouse_down:
                            if distance < GRAB_THRESHOLD:
                                if pinch_since is None:
                                    pinch_since = now
                                    # Lock exactly where the cursor was when pinch began.
                                    pinch_anchor = pyautogui.position()
                                state = "STABILIZING GRAB"
                                if now - pinch_since >= PINCH_CONFIRM_TIME:
                                    pyautogui.mouseDown(button="left")
                                    mouse_down = True
                                    drag_started = False
                                    release_since = None
                                    state = "HOLDING - MOVE TO DRAG"
                                    print("Mouse DOWN")
                            else:
                                pinch_since = None
                                pinch_anchor = None
                                state = "READY"
                                pyautogui.moveTo(round(smooth_position[0]),
                                                 round(smooth_position[1]), duration=0)
                        else:
                            # Wait until the hand moves enough to count as a drag.
                            if not drag_started:
                                moved = math.hypot(
                                    smooth_position[0] - pinch_anchor[0],
                                    smooth_position[1] - pinch_anchor[1],
                                )
                                if moved >= DRAG_START_DISTANCE:
                                    drag_started = True
                            if drag_started:
                                pyautogui.moveTo(round(smooth_position[0]),
                                                 round(smooth_position[1]), duration=0)
                                state = "DRAGGING"
                            else:
                                state = "HOLDING - MOVE TO DRAG"

                            if distance > RELEASE_THRESHOLD:
                                if release_since is None:
                                    release_since = now
                                if now - release_since >= RELEASE_CONFIRM_TIME:
                                    pyautogui.mouseUp(button="left")
                                    mouse_down = False
                                    drag_started = False
                                    pinch_since = None
                                    pinch_anchor = None
                                    release_since = None
                                    state = "RELEASED"
                                    print("Mouse UP")
                            else:
                                release_since = None
                    else:
                        state = "CURSOR OFF"
                else:
                    state = "NO HAND"
                    smooth_position = None
                    pinch_since = None
                    pinch_anchor = None
                    release_since = None
                    drag_started = False
                    if mouse_down:
                        pyautogui.mouseUp(button="left")
                        mouse_down = False
                        print("Mouse UP - hand lost")

                if smooth_position is not None:
                    position = (round(smooth_position[0]), round(smooth_position[1]))
                draw_status(frame, state, cursor_enabled, mouse_down, distance, position)
                cv2.imshow("GestureFileTransfer - Stabilized", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("m"):
                    if mouse_down:
                        pyautogui.mouseUp(button="left")
                        mouse_down = False
                    cursor_enabled = not cursor_enabled
                    smooth_position = None
                    pinch_since = None
                    pinch_anchor = None
                    release_since = None
                    drag_started = False
                    print("Cursor:", "ON" if cursor_enabled else "OFF")
    except KeyboardInterrupt:
        print("Stopped")
    except pyautogui.FailSafeException:
        print("PyAutoGUI failsafe activated")
    finally:
        if mouse_down:
            try:
                pyautogui.mouseUp(button="left")
            except pyautogui.FailSafeException:
                print("Move physical mouse away from screen corner to release it")
        camera.release()
        cv2.destroyAllWindows()
        print("Closed")


if __name__ == "__main__":
    main()
