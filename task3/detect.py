"""
detect.py — Orange cylinder detection via HSV colour segmentation.

Workflow:
  1. Undistort the input image using calibration parameters.
  2. Convert to HSV and threshold for orange pixels (two hue sub-ranges to
     catch orange that wraps around hue=0 in OpenCV's 0-179 scale).
  3. Morphologically clean the mask, then find contours.
  4. Filter contours by area and aspect ratio to keep likely cylinders.
  5. Merge vertically stacked boxes that belong to the same cylinder — the
     sign (non-orange) splits each cylinder into an upper and lower blob.
  6. The sign crop is the gap region between the two merged sub-boxes.
"""

import cv2
import numpy as np
from pathlib import Path


# ── Colour thresholds ────────────────────────────────────────────────────────
# Orange in OpenCV HSV (H: 0-179, S: 0-255, V: 0-255).
# We cover two sub-ranges so that deep orange / reddish-orange is caught.
ORANGE_LOWER_1 = np.array([0, 50, 50],  dtype=np.uint8)
ORANGE_UPPER_1 = np.array([25, 255, 255], dtype=np.uint8)
ORANGE_LOWER_2 = np.array([165, 120, 80],  dtype=np.uint8)
ORANGE_UPPER_2 = np.array([179, 255, 255], dtype=np.uint8)

# Contour filtering
MIN_AREA   = 400    # px² — discard tiny noise blobs
MAX_AREA   = 80_000 # px² — discard full-frame blobs
MIN_ASPECT = 0.15   # width/height — very wide blobs are probably not cylinders
MAX_ASPECT = 2.5    # width/height — also discard very wide blobs

# Merging: two boxes are the same cylinder if their horizontal centres are
# within this many pixels of each other and they are vertically close/touching.
MERGE_X_TOL   = 60    # px — max horizontal centre distance to consider same cylinder
MERGE_Y_GAP   = 80    # px — max vertical gap between boxes to attempt merge

SIGN_MIN_PX   = 20    # skip sign crops smaller than this in either dimension


def detect_cylinders(image: np.ndarray,
                     K: np.ndarray,
                     dist: np.ndarray,
                     undistort: bool = True
                     ) -> tuple[list[tuple[int,int,int,int]], list[np.ndarray]]:
    """
    Detect orange cylinders in *image* and return their bounding boxes plus
    the corresponding sign-region crops.

    Parameters
    ----------
    image     : BGR image (H x W x 3, uint8).
    K         : 3x3 camera intrinsic matrix.
    dist      : distortion coefficient vector (5 values).
    undistort : if True, undistort the image before processing.

    Returns
    -------
    boxes      : list of (x, y, w, h) rectangles — one per detected cylinder.
    sign_crops : list of BGR crops — one per detected cylinder (may be empty
                 if the crop would be too small).
    """
    if undistort:
        h, w = image.shape[:2]
        new_K, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1, (w, h))
        image = cv2.undistort(image, K, dist, None, new_K)

    # ── HSV thresholding ─────────────────────────────────────────────────────
    hsv  = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = (cv2.inRange(hsv, ORANGE_LOWER_1, ORANGE_UPPER_1) |
            cv2.inRange(hsv, ORANGE_LOWER_2, ORANGE_UPPER_2))

    # ── Morphological clean-up ───────────────────────────────────────────────
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=2)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # ── Contour detection ────────────────────────────────────────────────────
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    raw_boxes : list[tuple[int,int,int,int]] = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (MIN_AREA <= area <= MAX_AREA):
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        aspect = w / max(h, 1)
        if not (MIN_ASPECT <= aspect <= MAX_ASPECT):
            continue
        raw_boxes.append((x, y, w, h))

    # ── Merge vertically stacked boxes from the same cylinder ────────────────
    # The sign (non-orange) splits each cylinder into an upper and lower blob.
    # We group boxes by horizontal proximity then merge each group into one
    # combined box; the sign crop is the gap between the sub-boxes.
    merged_boxes, sign_crops = _merge_boxes(image, raw_boxes)
    return merged_boxes, sign_crops


def _merge_boxes(image: np.ndarray,
                 boxes: list[tuple[int,int,int,int]]
                 ) -> tuple[list[tuple[int,int,int,int]], list[np.ndarray]]:
    """
    Group raw boxes into cylinders by horizontal proximity, merge each group
    into one spanning box, and extract the sign crop from the gap between the
    upper and lower sub-boxes.

    For a single unmerged box (no partner found) the sign crop falls back to
    the middle third of the box.
    """
    if not boxes:
        return [], []

    used   = [False] * len(boxes)
    merged : list[tuple[int,int,int,int]] = []
    crops  : list[np.ndarray]             = []

    # Sort top-to-bottom so the first box in each group is always the upper one
    boxes = sorted(boxes, key=lambda b: b[1])

    for i, (xi, yi, wi, hi) in enumerate(boxes):
        if used[i]:
            continue
        cx_i = xi + wi // 2

        # Find the best partner: same horizontal column, directly below, close
        best_j   = -1
        best_gap = MERGE_Y_GAP + 1

        for j, (xj, yj, wj, hj) in enumerate(boxes):
            if j <= i or used[j]:
                continue
            cx_j = xj + wj // 2
            if abs(cx_i - cx_j) > MERGE_X_TOL:
                continue          # different column
            gap = yj - (yi + hi)  # vertical gap between bottom of i and top of j
            if 0 <= gap <= MERGE_Y_GAP and gap < best_gap:
                best_gap = gap
                best_j   = j

        if best_j >= 0:
            # ── Merge the two boxes ──────────────────────────────────────────
            xj, yj, wj, hj = boxes[best_j]
            mx = min(xi, xj)
            my = yi
            mw = max(xi + wi, xj + wj) - mx
            mh = (yj + hj) - yi
            merged.append((mx, my, mw, mh))
            used[i] = used[best_j] = True

            # Sign crop = the gap between bottom of upper box and top of lower
            gap_y0 = yi + hi
            gap_y1 = yj
            crop   = image[gap_y0:gap_y1, mx:mx + mw]
        else:
            # ── No partner — use middle third of the single box ──────────────
            merged.append((xi, yi, wi, hi))
            used[i] = True
            gap_y0  = yi + hi // 3
            gap_y1  = yi + (2 * hi) // 3
            crop    = image[gap_y0:gap_y1, xi:xi + wi]

        if crop.shape[0] >= SIGN_MIN_PX and crop.shape[1] >= SIGN_MIN_PX:
            crops.append(crop)
        else:
            crops.append(np.zeros((SIGN_MIN_PX, SIGN_MIN_PX, 3), dtype=np.uint8))

    return merged, crops


def draw_detections(image:  np.ndarray,
                    boxes:  list[tuple[int,int,int,int]],
                    dists:  list[float | None] | None = None,
                    labels: list[str  | None] | None = None
                    ) -> np.ndarray:
    """
    Draw bounding boxes (and optionally distance / label text) on a copy of
    *image*.  Returns the annotated image.

    Parameters
    ----------
    image  : BGR image.
    boxes  : list of (x, y, w, h) bounding boxes.
    dists  : optional list of distance values (metres) aligned with *boxes*.
    labels : optional list of classification label strings aligned with *boxes*.
    """
    vis = image.copy()

    for i, (x, y, w, h) in enumerate(boxes):
        # Cylinder box — green
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Sign region indicator — blue (middle third of merged box)
        sign_y0 = y + h // 3
        sign_y1 = y + (2 * h) // 3
        cv2.rectangle(vis, (x, sign_y0), (x + w, sign_y1), (255, 100, 0), 1)

        # Annotation text
        parts = []
        if dists is not None and i < len(dists) and dists[i] is not None:
            parts.append(f"{dists[i]:.2f}m")
        if labels is not None and i < len(labels) and labels[i] is not None:
            parts.append(labels[i])

        if parts:
            text = "  ".join(parts)
            cv2.putText(vis, text,
                        (x, max(y - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 255), 1, cv2.LINE_AA)

    return vis


def save_detection_image(image:    np.ndarray,
                         boxes:    list[tuple[int,int,int,int]],
                         out_path: str | Path,
                         dists:    list[float | None] | None = None,
                         labels:   list[str  | None] | None = None
                         ) -> None:
    """
    Annotate *image* with detections and save to *out_path*.
    """
    vis = draw_detections(image, boxes, dists=dists, labels=labels)
    cv2.imwrite(str(out_path), vis)
