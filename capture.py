import mss
import numpy as np
import tkinter as tk

def get_screen_size():
    # Hardcode to 1920x1080 as requested
    return 1920, 1080
SCREEN_WIDTH, SCREEN_HEIGHT = get_screen_size()
BOX_SIZE = 100  # FOV size
CAPTURE_MARGIN = 0  # No extra margin for strict center capture

def get_region():
    # Capture exactly the center of the screen, no margin, for 1920x1080
    x1 = SCREEN_WIDTH // 2 - BOX_SIZE // 2
    y1 = SCREEN_HEIGHT // 2 - BOX_SIZE // 2
    x2 = x1 + BOX_SIZE
    y2 = y1 + BOX_SIZE
    return (x1, y1, x2, y2)

def get_frame():
    region = get_region()
    with mss.mss() as sct:
        monitor = {"top": region[1], "left": region[0], "width": region[2] - region[0], "height": region[3] - region[1]}
        img = np.array(sct.grab(monitor))[:, :, :3]
        return img
