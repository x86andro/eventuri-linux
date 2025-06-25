import time
import numpy as np
import threading
from mouse import Mouse, button_states
from capture import get_frame, get_region, SCREEN_WIDTH, SCREEN_HEIGHT, BOX_SIZE, CAPTURE_MARGIN
from detection import detect_fake_full_body, detect_head
import cv2
from config import config

_aimbot_running = False
_aimbot_thread = None
debug_windows_open = False

try:
    import mss
except ImportError:
    mss = None

def adaptive_speed(dx, dy, min_speed=2, max_speed=24):
    distance = np.hypot(dx, dy)
    speed = int(min_speed + (max_speed - min_speed) * min(1.0, distance / 60))
    if distance == 0:
        return 0, 0
    step_dx = int(dx * speed / distance)
    step_dy = int(dy * speed / distance)
    return step_dx, step_dy

def aimbot_loop():
    global _aimbot_running, debug_windows_open
    mouse = Mouse()
    last_button_state = False
    last_shot_time = 0

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
        # Option to merge all blobs into a full body region (per body, not global)
        if getattr(config, 'full_body_merge', False):
            blobs = merge_blobs_full_body_clusters(blobs, color=config.target_color, distance_threshold=60)
        else:
            blobs = merge_blobs_by_distance(blobs, distance_threshold=40)
        print(f"[DEBUG] Blobs detected (after merge): {len(blobs)}")
        for blob in blobs:
            print(f"  Color: {blob['color']}, Center: {blob['center']}, Area: {blob['area']}")
        # Visualize merged blobs on debug image
        if config.debug and debug_img is not None:
            debug_img = visualize_merged_blobs(debug_img, blobs, draw_rect=True)

        display = debug_img if debug_img is not None else frame.copy()
        button_index = config.mouse_button
        if blobs and button_states[button_index]:
            target_blob = blobs[0]
            cx, cy = target_blob['center']
            dx = cx - (frame.shape[1] // 2)
            dy = cy - (frame.shape[0] // 2)
            print(f"[AIMBOT] Moving mouse to blob center dx={dx}, dy={dy}")
            try:
                if config.mode == "bezier":
                    # Use the new km.move(x, y, segments, ctrl_x, ctrl_y) command for full bezier
                    segments = max(3, min(20, int(np.hypot(dx, dy) // 5)))  # Example: segment count based on distance
                    ctrl_x = int(dx * 0.5)  # Control point at 50% of dx
                    ctrl_y = int(dy * 0.5)  # Control point at 50% of dy
                    mouse.move_bezier(dx, dy, segments, ctrl_x, ctrl_y)
                elif config.mode == "normal":
                    mouse.move(dx, dy)
                elif config.mode == "auto":
                    ms = 40
                    mouse.move_auto(dx, dy, ms)
                else:
                    # Default to bezier for any other mode
                    segments = max(3, min(20, int(np.hypot(dx, dy) // 5)))
                    ctrl_x = int(dx * 0.5)
                    ctrl_y = int(dy * 0.5)
                    mouse.move_bezier(dx, dy, segments, ctrl_x, ctrl_y)
            except Exception as e:
                print(f"[ERROR] Mouse action failed: {e}")
        else:
            if not blobs:
                print("[INFO] No blobs detected.")
            if not button_states[button_index]:
                print(f"[INFO] Mouse button {button_index} not pressed.")

        if config.debug:
            cv2.imshow("Aimbot: Detection", display)
            debug_windows_open = True
        else:
            if debug_windows_open:
                cv2.destroyWindow("Aimbot: Detection")
                debug_windows_open = False

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            _aimbot_running = False
            break

        last_button_state = button_states[button_index]
        time.sleep(0.001)

def start_aimbot():
    global _aimbot_running, _aimbot_thread
    if not _aimbot_running:
        _aimbot_running = True
        _aimbot_thread = threading.Thread(target=aimbot_loop, daemon=True)
        _aimbot_thread.start()

def stop_aimbot():
    global _aimbot_running, _aimbot_thread
    _aimbot_running = False
    if _aimbot_thread is not None:
        _aimbot_thread.join()
        _aimbot_thread = None

if __name__ == "__main__":
    import gui
    gui.EventuriGUI().mainloop()

