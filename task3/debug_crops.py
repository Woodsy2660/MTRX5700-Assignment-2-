"""
debug_crops.py — saves side-by-side images showing the full cylinder bounding
box alongside the current sign crop, so you can visually determine the correct
SIGN_START / SIGN_END values in detect.py.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import cv2
import numpy as np
from calibration import K, dist
from detect import detect_cylinders

IMG_DIR = Path.home() / "extracted_data/extracted/traffic_signs_multiple_2/images/images"
OUT_DIR = Path(__file__).parent / "output" / "debug_crops"
OUT_DIR.mkdir(parents=True, exist_ok=True)

imgs   = sorted(IMG_DIR.glob("*.png"))
# Sample every 10th image across the sequence
sample = imgs[::10][:20]

saved = 0
for img_path in sample:
    image = cv2.imread(str(img_path))
    if image is None:
        continue

    boxes, crops = detect_cylinders(image, K, dist)

    for j, (box, crop) in enumerate(zip(boxes, crops)):
        x, y, w, h = box

        # Full cylinder crop with crop region marked
        full = image[y:y+h, x:x+w].copy()
        # Show the middle-third region (where gap crop lands for merged boxes)
        y0 = h // 3
        y1 = (2 * h) // 3
        cv2.rectangle(full, (0, y0), (w-1, y1), (0, 255, 0), 2)
        cv2.putText(full, "sign region",
                    (4, y0 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

        # Resize both to same height for side-by-side
        target_h = 200
        def resize_h(img, th):
            if img.shape[0] == 0 or img.shape[1] == 0:
                return np.zeros((th, th, 3), dtype=np.uint8)
            scale = th / img.shape[0]
            return cv2.resize(img, (max(1, int(img.shape[1]*scale)), th))

        left  = resize_h(full, target_h)
        right = resize_h(crop, target_h)

        # Label them
        cv2.putText(left,  "FULL BOX (green=crop region)",
                    (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,255), 1)
        cv2.putText(right, "CURRENT CROP (sent to model)",
                    (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,255), 1)

        combined = np.hstack([left, np.ones((target_h, 4, 3), dtype=np.uint8)*128, right])
        out_path = OUT_DIR / f"{img_path.stem}_cyl{j}.jpg"
        cv2.imwrite(str(out_path), combined)
        saved += 1

print(f"Saved {saved} debug images to {OUT_DIR.resolve()}")
print("Right panel = sign crop sent to model. Left panel = full merged box with region marked.")
print("If crops look wrong, adjust MERGE_X_TOL / MERGE_Y_GAP in detect.py.")
