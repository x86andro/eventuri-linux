import cv2
import numpy as np
from config import config, DEFAULT_CONFIG
COLOR_RANGES = config.color_ranges

def mask_frame(frame, color):
    # Faster blur, less kernel size for speed
    blurred = cv2.GaussianBlur(frame, (3, 3), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    lower = np.array(COLOR_RANGES[color]['lower'], dtype=np.uint8)
    upper = np.array(COLOR_RANGES[color]['upper'], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.erode(mask, None, iterations=1)
    mask = cv2.dilate(mask, None, iterations=2)
    return mask

def detect_fake_full_body(
    frame, color='purple', body_height=48, body_width=22, min_width=8, max_width_ratio=0.45, debug=False
):
    mask = mask_frame(frame, color)
    h, w = mask.shape
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Find contours for all blobs
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_target = None
    best_area = 0
    best_rect = None
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 30:
            continue
        x, y, w_box, h_box = cv2.boundingRect(cnt)
        aspect = h_box / w_box if w_box > 0 else 0
        if w_box < min_width or w_box > mask.shape[1] * max_width_ratio:
            continue
        if aspect < 1.0 or aspect > 4.0:
            continue
        if area > best_area:
            best_area = area
            best_rect = (x, y, w_box, h_box)
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                best_target = (cx, cy)
            else:
                best_target = (x + w_box // 2, y + h_box // 2)

    debug_images = None
    if debug:
        debug_images = {
            "mask": mask,
            "closed": closed,
        }
        if best_rect:
            debug_img = frame.copy()
            x, y, w_box, h_box = best_rect
            cv2.rectangle(debug_img, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)
            debug_images["rect"] = debug_img

    return best_target, mask, debug_images

def detect_blobs_all_colors(frame, debug=False):
    # Detect blobs for all configured colors
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    blobs = []
    debug_img = frame.copy() if debug else None
    min_area = getattr(config, 'min_blob_area', 10)
    for color, range_ in COLOR_RANGES.items():
        lower = np.array(range_['lower'], dtype=np.uint8)
        upper = np.array(range_['upper'], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
        mask = cv2.erode(mask, None, iterations=1)
        mask = cv2.dilate(mask, None, iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = x + w // 2, y + h // 2
            blobs.append({
                'color': color,
                'rect': (x, y, w, h),
                'center': (cx, cy),
                'area': area
            })
            if debug and debug_img is not None:
                cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.drawMarker(debug_img, (cx, cy), (0, 255, 255), markerType=cv2.MARKER_CROSS, markerSize=12, thickness=2)
                cv2.putText(debug_img, color, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
    return blobs, debug_img

def merge_blobs_by_distance(blobs, distance_threshold=40):
    """
    Merge blobs whose centers are within distance_threshold of each other.
    Returns a new list of merged blobs (with averaged center and summed area).
    """
    if not blobs:
        return []
    merged = []
    used = set()
    for i, blob in enumerate(blobs):
        if i in used:
            continue
        group = [blob]
        used.add(i)
        cx1, cy1 = blob['center']
        for j, other in enumerate(blobs):
            if j == i or j in used:
                continue
            cx2, cy2 = other['center']
            if np.hypot(cx1 - cx2, cy1 - cy2) < distance_threshold:
                group.append(other)
                used.add(j)
        if len(group) == 1:
            merged.append(blob)
        else:
            # Average center, sum area, keep color of largest blob
            total_area = sum(b['area'] for b in group)
            avg_cx = int(sum(b['center'][0] * b['area'] for b in group) / total_area)
            avg_cy = int(sum(b['center'][1] * b['area'] for b in group) / total_area)
            largest = max(group, key=lambda b: b['area'])
            merged.append({
                'color': largest['color'],
                'rect': largest['rect'],
                'center': (avg_cx, avg_cy),
                'area': total_area
            })
    return merged

def merge_blobs_full_body(blobs, color=None):
    """
    Merge all blobs of the same color into a single bounding box and center.
    If color is specified, only merge blobs of that color.
    Returns a list of merged blobs (one per color).
    """
    if not blobs:
        return []
    merged = []
    color_groups = {}
    for blob in blobs:
        c = blob['color']
        if color and c != color:
            continue
        color_groups.setdefault(c, []).append(blob)
    for c, group in color_groups.items():
        if not group:
            continue
        # Merge all rects
        xs = [b['rect'][0] for b in group]
        ys = [b['rect'][1] for b in group]
        ws = [b['rect'][0] + b['rect'][2] for b in group]
        hs = [b['rect'][1] + b['rect'][3] for b in group]
        x1, y1 = min(xs), min(ys)
        x2, y2 = max(ws), max(hs)
        w, h = x2 - x1, y2 - y1
        # Center is center of merged box
        cx, cy = x1 + w // 2, y1 + h // 2
        total_area = sum(b['area'] for b in group)
        merged.append({
            'color': c,
            'rect': (x1, y1, w, h),
            'center': (cx, cy),
            'area': total_area
        })
    return merged

def cluster_blobs_by_distance(blobs, distance_threshold=60):
    """
    Cluster blobs by proximity. Returns a list of clusters (each a list of blobs).
    """
    clusters = []
    used = set()
    for i, blob in enumerate(blobs):
        if i in used:
            continue
        cluster = [blob]
        used.add(i)
        cx1, cy1 = blob['center']
        for j, other in enumerate(blobs):
            if j == i or j in used:
                continue
            cx2, cy2 = other['center']
            if np.hypot(cx1 - cx2, cy1 - cy2) < distance_threshold:
                cluster.append(other)
                used.add(j)
        clusters.append(cluster)
    return clusters

def merge_blobs_full_body_clusters(blobs, color=None, distance_threshold=60):
    """
    Cluster blobs by distance, then merge each cluster into a convex hull and center.
    Returns a list of merged blobs (one per detected body), with hull points.
    """
    if not blobs:
        return []
    # Filter by color if needed
    if color:
        blobs = [b for b in blobs if b['color'] == color]
    clusters = cluster_blobs_by_distance(blobs, distance_threshold=distance_threshold)
    merged = []
    for group in clusters:
        if not group:
            continue
        # Collect all contour points from all blobs in the group
        points = []
        for b in group:
            x, y, w, h = b['rect']
            # Rectangle corners as points
            points.extend([(x, y), (x+w, y), (x, y+h), (x+w, y+h)])
        points = np.array(points)
        hull = cv2.convexHull(points)
        M = cv2.moments(hull)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = np.mean(hull[:,0,0]), np.mean(hull[:,0,1])
        x, y, w, h = cv2.boundingRect(hull)
        total_area = sum(b['area'] for b in group)
        merged.append({
            'color': group[0]['color'],
            'rect': (x, y, w, h),
            'center': (cx, cy),
            'area': total_area,
            'hull': hull
        })
    return merged

def visualize_merged_blobs(debug_img, merged_blobs, draw_rect=True):
    # Draw merged blobs as magenta convex hulls and label them
    for blob in merged_blobs:
        cx, cy = blob['center']
        area = blob['area']
        radius = max(8, int(np.sqrt(area / np.pi)))
        if 'hull' in blob:
            cv2.polylines(debug_img, [blob['hull']], isClosed=True, color=(255, 0, 255), thickness=2)
        elif draw_rect:
            x, y, w, h = blob['rect']
            cv2.rectangle(debug_img, (x, y), (x + w, y + h), (255, 0, 255), 2)
        cv2.circle(debug_img, (cx, cy), radius, (255, 0, 255), 2)
        cv2.drawMarker(debug_img, (cx, cy), (255, 0, 255), markerType=cv2.MARKER_STAR, markerSize=16, thickness=2)
        cv2.putText(debug_img, 'MERGED', (cx + 5, cy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255), 1)
    return debug_img

def detect_head(frame, color='purple', min_area=10, max_area=400, debug=False):
    mask = mask_frame(frame, color)
    h, w = mask.shape
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_head = None
    best_score = 0
    best_rect = None
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue
        x, y, w_box, h_box = cv2.boundingRect(cnt)
        aspect = w_box / h_box if h_box > 0 else 0
        if aspect < 0.6 or aspect > 1.4:
            continue
        score = area - 2*y
        if score > best_score:
            best_score = score
            best_rect = (x, y, w_box, h_box)
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                best_head = (cx, cy)
            else:
                best_head = (x + w_box // 2, y + h_box // 2)
    debug_images = None
    if debug:
        debug_images = {
            "mask": mask,
            "closed": closed,
        }
        if best_rect:
            debug_img = frame.copy()
            x, y, w_box, h_box = best_rect
            cv2.rectangle(debug_img, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)
            debug_images["rect"] = debug_img
    return best_head, mask, debug_images

