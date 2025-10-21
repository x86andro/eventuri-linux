import subprocess
import numpy as np
from PIL import Image
from io import BytesIO
import shutil
import os

if not shutil.which("grim"):
    raise RuntimeError("grim is not installed.")

def get_screen_size():
    try:
        output = subprocess.run(
            ["swaymsg", "-t", "get_outputs"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=True
        )
        import json
        data = json.loads(output.stdout)
        for out in data:
            if out.get("active"):
                return out["current_mode"]["width"], out["current_mode"]["height"]
    except Exception:
        return 1920, 1080

SCREEN_WIDTH, SCREEN_HEIGHT = get_screen_size()
BOX_SIZE = 200
CAPTURE_MARGIN = 0

def get_region(fov=None):
    global BOX_SIZE
    if fov is None:
        fov = BOX_SIZE
    cx, cy = SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2
    x = cx - fov // 2
    y = cy - fov // 2
    return (x, y, fov, fov)

def get_frame(fov=None):
    x, y, w, h = get_region(fov)
    geometry = f"{x},{y} {w}x{h}"

    try:
        proc = subprocess.run(
            ["grim", "-g", geometry, "-"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )
        img = Image.open(BytesIO(proc.stdout)).convert("RGB")
        return np.array(img)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Screen capture failed: {e.stderr.decode().strip()}")
