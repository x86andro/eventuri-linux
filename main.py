import os
import time
import numpy as np
import threading
from mouse import Mouse, button_states
from capture import get_frame, get_region, SCREEN_WIDTH, SCREEN_HEIGHT, BOX_SIZE, CAPTURE_MARGIN
from detection import detect_fake_full_body, detect_head
import cv2
from config import config
os.environ["QT_QPA_PLATFORM"] = "xcb"

import queue

_aimbot_running = False
_aimbot_thread = None

_display_thread = None
_display_running = False
_display_lock = threading.Lock()
_display_frame = None
_display_window_name = "Aimbot: Detection"

debug_windows_open = False

try:
    import mss
except ImportError:
    mss = None

def _display_loop():
    global _display_running, debug_windows_open, _display_frame

    try:
        while _display_running:
            frame = None
            with _display_lock:
                if _display_frame is not None:
                    frame = _display_frame

            if config.debug and frame is not None:
                try:
                    if not debug_windows_open:
                        cv2.namedWindow(_display_window_name, cv2.WINDOW_NORMAL)
                        debug_windows_open = True
                    cv2.imshow(_display_window_name, frame)
                except cv2.error as e:
                    print(f"[DISPLAY] OpenCV error during imshow: {e}")
                    config.debug = False

                try:
                    cv2.waitKey(1)
                except cv2.error as e:
                    print(f"[DISPLAY] OpenCV error during waitKey: {e}")
                    config.debug = False
            else:
                if debug_windows_open:
                    try:
                        cv2.destroyWindow(_display_window_name)
                        cv2.waitKey(1)
                    except cv2.error as e:
                        print(f"[DISPLAY] OpenCV error during destroyWindow: {e}")
                    debug_windows_open = False

            time.sleep(0.01)

    finally:
        if debug_windows_open:
            try:
                cv2.destroyWindow(_display_window_name)
                cv2.waitKey(1)
                debug_windows_open = False
            except cv2.error:
                pass

def start_display_thread():
    global _display_thread, _display_running
    if _display_thread is None:
        _display_running = True
        _display_thread = threading.Thread(target=_display_loop, daemon=True)
        _display_thread.start()

def adaptive_speed(dx, dy, min_speed=2, max_speed=24):
    distance = np.hypot(dx, dy)
    speed = int(min_speed + (max_speed - min_speed) * min(1.0, distance / 60))
    if distance == 0:
        return 0, 0
    step_dx = int(dx * speed / distance)
    step_dy = int(dy * speed / distance)
    return step_dx, step_dy

def aimbot_loop():
    global _aimbot_running, _display_frame
    mouse = Mouse()
    last_button_state = False
    last_shot_time = 0
    start_display_thread()

    while _aimbot_running:
        region = get_region()
        frame = get_frame()
        if frame is None or frame.shape[0] == 0 or frame.shape[1] == 0:
            print("[WARN] Frame is empty or invalid. Region:", region)
            last_button_state = False
            time.sleep(0.1)
            continue

        from detection import detect_blobs_all_colors, merge_blobs_by_distance, merge_blobs_full_body_clusters, visualize_merged_blobs
        blobs, debug_img = detect_blobs_all_colors(frame, debug=config.debug)

        if getattr(config, 'full_body_merge', False):
            blobs = merge_blobs_full_body_clusters(blobs, color=config.target_color, distance_threshold=60)
        else:
            blobs = merge_blobs_by_distance(blobs, distance_threshold=40)

        if config.debug and debug_img is not None:
            debug_img = visualize_merged_blobs(debug_img, blobs, draw_rect=True)

        display = debug_img if debug_img is not None else frame.copy()

        button_index = config.mouse_button
        if blobs and button_states[button_index]:
            target_blob = blobs[0]
            cx, cy = target_blob['center']
            dx = cx - (frame.shape[1] // 2)
            dy = cy - (frame.shape[0] // 2)
            try:
                if config.mode == "bezier":
                    segments = max(3, min(20, int(np.hypot(dx, dy) // 5)))
                    mouse.move_bezier(dx, dy, segments, int(dx * 0.5), int(dy * 0.5))
                elif config.mode == "normal":
                    mouse.move(dx, dy)
                elif config.mode == "auto":
                    mouse.move_auto(dx, dy, 40)
                else:
                    segments = max(3, min(20, int(np.hypot(dx, dy) // 5)))
                    mouse.move_bezier(dx, dy, segments, int(dx * 0.5), int(dy * 0.5))
            except Exception as e:
                print(f"[ERROR] Mouse action failed: {e}")

        if config.debug:
            with _display_lock:
                _display_frame = display
        else:
            with _display_lock:
                _display_frame = None

        last_button_state = button_states[button_index]
        time.sleep(0.001)

def start_aimbot():
    global _aimbot_running, _aimbot_thread
    if not _aimbot_running:
        _aimbot_running = True
        _aimbot_thread = threading.Thread(target=aimbot_loop, daemon=True)
        _aimbot_thread.start()

def stop_aimbot():
    global _aimbot_running, _aimbot_thread, _display_running, _display_thread
    _aimbot_running = False
    _display_running = False

    if _aimbot_thread is not None:
        _aimbot_thread.join(timeout=1.5)
        if _aimbot_thread.is_alive():
            print("[WARNING] Aimbot thread did not exit cleanly.")
        _aimbot_thread = None

    if _display_thread is not None:
        _display_thread.join(timeout=1.5)
        if _display_thread.is_alive():
            print("[WARNING] Display thread did not exit cleanly.")
        _display_thread = None

if __name__ == "__main__":
    import gui
    gui.EventuriGUI().mainloop()
