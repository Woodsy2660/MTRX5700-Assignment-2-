"""
detect.py — Orange cylinder detection via HSV colour segmentation.

Workflow:
  1. Undistort the input image using calibration parameters.
  2. Convert to HSV and threshold for orange pixels (two hue sub-ranges to
     catch orange that wraps around hue=0 in OpenCV's 0-179 scale).
  3. Morphologically clean the mask, then find contours.
  4. Filter contours by area and aspect ratio to keep likely cylinders.
  5. Return axis-aligned bounding boxes and corresponding sign crops
     (upper portion of each box, where the traffic sign lives).
"""

import cv2
import numpy as np
from pathlib import Path


# ── Colour thresholds ────────────────────────────────────────────────────────
# Orange in OpenCV HSV (H: 0-179, S: 0-255, V: 0-255).
# We cover two sub-ranges so that deep orange / reddish-orange is caught.
ORANGE_LOWER_1 = np.array([0,  120, 80],  dtype=np.uint8)
ORANGE_UPPER_1 = np.array([18, 255, 255], dtype=np.uint8)
ORANGE_LOWER_2 = np.array([165, 120, 80],  dtype=np.uint8)
ORANGE_UPPER_2 = np.array([179, 255, 255], dtype=np.uint8)

# Contour filtering
MIN_AREA   = 400    # px² — discard tiny noise blobs
MAX_AREA   = 80_000 # px² — discard full-frame blobs
MIN_ASPECT = 0.15   # width/height — very wide blobs are probably not cylinders
MAX_ASPECT = 2.5    # width/height — also discard very wide blobs

# Sign is in the top fraction of the cylinder bounding box
SIGN_FRACTION = 0.55   # use top 55 % of box height as the sign region
SIGN_MIN_PX   = 20     # skip sign crops smaller than this in either dimension


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

    boxes      : list[tuple[int,int,int,int]] = []
    sign_crops : list[np.ndarray]             = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (MIN_AREA <= area <= MAX_AREA):
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        aspect = w / max(h, 1)
        if not (MIN_ASPECT <= aspect <= MAX_ASPECT):
            continue

        boxes.append((x, y, w, h))

        # ── Sign crop — top SIGN_FRACTION of the bounding box ────────────────
        sign_h = max(int(h * SIGN_FRACTION), 1)
        crop   = image[y : y + sign_h, x : x + w]
        if crop.shape[0] >= SIGN_MIN_PX and crop.shape[1] >= SIGN_MIN_PX:
            sign_crops.append(crop)
        else:
            sign_crops.append(np.zeros((SIGN_MIN_PX, SIGN_MIN_PX, 3),
                                       dtype=np.uint8))

    return boxes, sign_crops


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

        # Sign region indicator — blue
        sign_h = int(h * SIGN_FRACTION)
        cv2.rectangle(vis, (x, y), (x + w, y + sign_h), (255, 100, 0), 1)

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
