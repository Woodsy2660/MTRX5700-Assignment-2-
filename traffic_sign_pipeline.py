"""
traffic_sign_pipeline.py
========================
Full pipeline for Task 3:
  1. Extract images from .mcap ROS2 bags (no ROS2 installation required)
  2. Detect orange cylinders via HSV colour segmentation (detect.py)
  3. Classify each traffic sign crop with best_model.pth (ResNet18)
  4. Save annotated images with green bounding boxes + label text
  5. Compare predictions against expected results and report accuracy

Usage
-----
    python traffic_sign_pipeline.py

Install dependencies first:
    pip install mcap mcap-ros2-support opencv-python torch torchvision pillow numpy

Directory layout expected
-------------------------
MTRX5700-Assignment-2-/
    best_model.pth
    network.py
    traffic_sign_pipeline.py   <- this file
    task3/
        detect.py
        calibration.py
    Task3_trafficsign_dataset/
        traffic_sign_left_1/
            traffic_sign_left_1_0.mcap
            metadata.yaml
        traffic_sign_left_2/ ...
        ...

Output
------
    MTRX5700-Assignment-2-/extracted_images/<bag_name>/   raw .png frames
    MTRX5700-Assignment-2-/annotated_images/<bag_name>/   annotated .jpg
    MTRX5700-Assignment-2-/debug_images/<bag_name>/       HSV mask + crops
    MTRX5700-Assignment-2-/results.txt
    MTRX5700-Assignment-2-/results.csv
"""

import sys
import csv
from pathlib import Path

# -- Path setup — must happen before local imports
REPO_ROOT = Path(__file__).resolve().parent
TASK3_DIR = REPO_ROOT / "task3"
sys.path.insert(0, str(TASK3_DIR))
sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

from detect      import detect_cylinders
from calibration import K, dist as dist_coeffs
from network     import ResNet18


# =============================================================================
# Configuration
# =============================================================================

DATA_ROOT       = REPO_ROOT / "Task3_trafficsign_dataset"
EXTRACTED_DIR   = REPO_ROOT / "extracted_images"
ANNOTATED_DIR   = REPO_ROOT / "annotated_images"
DEBUG_DIR       = REPO_ROOT / "debug_images"       # HSV masks + probability panels
MODEL_PATH      = REPO_ROOT / "best_model.pth"
RESULTS_TXT     = REPO_ROOT / "results.txt"
RESULTS_CSV     = REPO_ROOT / "results.csv"

IMAGE_TOPIC         = "/camera/image_raw"
MAX_SIGNS           = 3
CONFIDENCE_THRESH   = 0.50   # lower this if classifier never fires

# How many evenly-spaced debug frames to save per bag (0 = disabled)
DEBUG_FRAMES_PER_BAG = 20

CLASS_NAMES = {
    0: "Stop",
    1: "Turn right",
    2: "Turn left",
    3: "Ahead only",
    4: "Roundabout mandatory",
}
NUM_CLASSES = len(CLASS_NAMES)

TRANSFORM = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])


# =============================================================================
# Expected results ground truth
# =============================================================================

EXPECTED = {
    "traffic_sign_left_1": [
        (None, ["Turn left"]),
    ],
    "traffic_sign_left_2": [
        (None, ["Turn left"]),
    ],
    "traffic_sign_right_1": [
        (None, ["Turn right"]),
    ],
    "traffic_sign_right_2": [
        (588,  ["Turn right"]),
        (None, []),
    ],
    "traffic_signs_multiple_1": [
        (386,  ["Turn left", "Turn left", "Turn right"]),
        (142,  ["Turn left", "Turn left"]),
        (None, ["Turn left"]),
    ],
    "traffic_signs_multiple_2": [
        (148,  ["Turn left", "Turn right"]),
        (128,  ["Turn left", "Turn left", "Turn right"]),
        (37,   ["Turn left", "Turn right"]),
        (2,    ["Turn left", "Turn left", "Turn right"]),
        (76,   ["Turn left", "Turn left"]),
        (17,   ["Turn left"]),
        (5,    []),
        (38,   ["Turn right"]),
        (30,   []),
        (None, ["Turn left"]),
    ],
    "traffic_signs_multiple_3": [
        (208,  ["Turn left", "Turn left"]),
        (130,  ["Turn left"]),
        (79,   ["Turn left", "Turn right"]),
        (196,  ["Turn right"]),
        (None, []),
    ],
}

BAG_NAMES = list(EXPECTED.keys())


# =============================================================================
# Step 1 — Image extraction from .mcap bags
# =============================================================================

def _decode_ros_image(msg) -> np.ndarray | None:
    enc  = (msg.encoding or "").lower().strip()
    h, w = msg.height, msg.width
    step = msg.step
    data = bytes(msg.data)
    try:
        if enc == "rgb8":
            arr = np.frombuffer(data, dtype=np.uint8).reshape(h, step)[:, :w*3].reshape(h, w, 3)
            return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        elif enc == "bgr8":
            arr = np.frombuffer(data, dtype=np.uint8).reshape(h, step)[:, :w*3].reshape(h, w, 3)
            return arr.copy()
        elif enc == "mono8":
            arr = np.frombuffer(data, dtype=np.uint8).reshape(h, step)[:, :w]
            return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        elif enc in ("mono16", "16uc1"):
            arr = np.frombuffer(data, dtype=np.uint16).reshape(h, w)
            return cv2.cvtColor(cv2.convertScaleAbs(arr, alpha=255.0/65535.0),
                                cv2.COLOR_GRAY2BGR)
        else:
            arr = np.frombuffer(data, dtype=np.uint8).reshape(h, step)[:, :w*3].reshape(h, w, 3)
            return arr.copy()
    except Exception as exc:
        print(f"      [WARN] Decode failed (enc={enc}): {exc}")
        return None


def extract_images(bag_name: str) -> Path:
    try:
        from mcap_ros2.reader import read_ros2_messages
    except ImportError:
        print("  [ERROR] Run:  pip install mcap mcap-ros2-support")
        sys.exit(1)

    bag_dir = DATA_ROOT / bag_name
    img_dir = EXTRACTED_DIR / bag_name
    img_dir.mkdir(parents=True, exist_ok=True)

    existing = sorted(img_dir.glob("*.png"))
    if existing:
        print(f"  [{bag_name}] {len(existing)} images already extracted — skipping.")
        return img_dir

    mcap_files = sorted(bag_dir.glob("*.mcap"))
    if not mcap_files:
        print(f"  [WARN] No .mcap files in {bag_dir}")
        return img_dir

    count = 0
    for mcap_path in mcap_files:
        print(f"  [{bag_name}] Extracting {mcap_path.name} ...")
        try:
            for msg in read_ros2_messages(str(mcap_path), topics=[IMAGE_TOPIC]):
                frame = _decode_ros_image(msg.ros_msg)
                if frame is None:
                    continue
                cv2.imwrite(str(img_dir / f"{count:06d}.png"), frame)
                count += 1
                if count % 200 == 0:
                    print(f"      {count} frames extracted ...")
        except Exception as exc:
            print(f"  [ERROR] {mcap_path.name}: {exc}")

    print(f"  [{bag_name}] Extracted {count} images -> {img_dir}")
    return img_dir


# =============================================================================
# Step 2+3 — Model loading and classification
# =============================================================================

def load_model(model_path: Path, device: torch.device):
    model = ResNet18(num_classes=NUM_CLASSES)
    checkpoint = torch.load(str(model_path), map_location=device)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print(f"Model loaded from {model_path}  (device: {device})")
    return model


def classify_crop(crop: np.ndarray, model, device: torch.device
                  ) -> tuple[str | None, float, dict]:
    """
    Returns (label, top_confidence, all_class_probs_dict).
    label is None when top confidence < CONFIDENCE_THRESH.
    """
    empty = {v: 0.0 for v in CLASS_NAMES.values()}
    if crop is None or crop.size == 0 or crop.shape[0] < 4 or crop.shape[1] < 4:
        return None, 0.0, empty

    pil    = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    tensor = TRANSFORM(pil).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1)[0].cpu().numpy()

    all_probs = {CLASS_NAMES[i]: float(probs[i]) for i in range(NUM_CLASSES)}
    best_idx  = int(np.argmax(probs))
    conf      = float(probs[best_idx])
    label     = CLASS_NAMES[best_idx] if conf >= CONFIDENCE_THRESH else None

    return label, conf, all_probs


# =============================================================================
# Debug frame: original | HSV mask  +  per-class probability panel
# =============================================================================

def _orange_mask(image: np.ndarray) -> np.ndarray:
    """Reproduce the same HSV orange mask used in detect.py."""
    hsv  = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    m1   = cv2.inRange(hsv, np.array([0,   120, 80], np.uint8),
                            np.array([18,  255, 255], np.uint8))
    m2   = cv2.inRange(hsv, np.array([165, 120, 80], np.uint8),
                            np.array([179, 255, 255], np.uint8))
    mask = m1 | m2
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kern, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kern, iterations=2)
    return mask


def save_debug_frame(image:         np.ndarray,
                     boxes:         list,
                     sign_crops:    list,
                     all_probs_list: list[dict],
                     labels:        list,
                     confs:         list,
                     out_path:      Path) -> None:
    """
    Saves a composite debug image:

      TOP ROW:  [annotated original]  |  [orange HSV mask]
      BOTTOM:   text panel — one block per detected cylinder showing
                every class name and its softmax probability as a bar,
                with the winning class highlighted in green.

    If no cylinders were detected the bottom panel says so and
    tells you to look at the HSV mask on the right to diagnose.
    """
    # -- Top row: annotated original + HSV mask side by side
    vis = image.copy()
    for (x, y, w, h), label, conf in zip(boxes, labels, confs):
        colour = (0, 255, 0) if label else (0, 0, 200)
        cv2.rectangle(vis, (x, y), (x+w, y+h), colour, 2)
        text = f"{label} {conf:.0%}" if label else f"no sign {conf:.0%}"
        cv2.putText(vis, text, (x, max(y-8, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0),    3, cv2.LINE_AA)
        cv2.putText(vis, text, (x, max(y-8, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

    mask_bgr = cv2.cvtColor(_orange_mask(image), cv2.COLOR_GRAY2BGR)
    top_row  = np.hstack([vis, mask_bgr])

    # -- Bottom panel: per-class probabilities for every detected cylinder
    row_h   = 22
    n_crops = max(len(sign_crops), 1)
    # each crop block: 1 header + NUM_CLASSES rows + 1 spacer
    panel_h = (2 + NUM_CLASSES + 1) * row_h * n_crops + 40
    panel   = np.full((panel_h, top_row.shape[1], 3), 30, dtype=np.uint8)

    y_cur = 24
    font  = cv2.FONT_HERSHEY_SIMPLEX

    if not sign_crops:
        msg = ("No cylinders detected.  "
               "Inspect the HSV mask (right side above) — "
               "if the orange posts are not white in the mask, "
               "the HSV thresholds need tuning.")
        cv2.putText(panel, msg, (10, y_cur), font, 0.48,
                    (60, 120, 255), 1, cv2.LINE_AA)
    else:
        for ci, (crop, probs_dict, label, conf) in enumerate(
                zip(sign_crops, all_probs_list, labels, confs)):
            # Header
            winner   = label or "NONE  (below confidence threshold)"
            hdr_col  = (80, 220, 255)
            header   = (f"Cylinder {ci+1}  |  "
                        f"Top prediction: {winner}  ({conf:.1%})  "
                        f"[threshold={CONFIDENCE_THRESH:.0%}]")
            cv2.putText(panel, header, (10, y_cur), font, 0.48, hdr_col, 1, cv2.LINE_AA)
            y_cur += row_h

            # One line per class, sorted highest -> lowest probability
            for cls_name, prob in sorted(probs_dict.items(), key=lambda x: -x[1]):
                bar_len = int(prob * 35)
                bar     = "|" * bar_len
                line    = f"  {cls_name:<28}  {prob:5.1%}  {bar}"
                # Green for the winning class, white otherwise
                colour  = (50, 220, 80) if cls_name == label else (190, 190, 190)
                # Dim grey if this class is very unlikely
                if prob < 0.05:
                    colour = (90, 90, 90)
                cv2.putText(panel, line, (10, y_cur), font, 0.42, colour, 1, cv2.LINE_AA)
                y_cur += row_h

            y_cur += row_h  # spacer between crops

    combined = np.vstack([top_row, panel])
    cv2.imwrite(str(out_path), combined)


# =============================================================================
# Step 4 — Production annotated images
# =============================================================================

def annotate_and_save(image:    np.ndarray,
                      boxes:    list,
                      labels:   list,
                      confs:    list,
                      out_path: Path) -> None:
    vis = image.copy()
    for (x, y, w, h), label, conf in zip(boxes, labels, confs):
        colour = (0, 255, 0) if label else (0, 0, 200)
        cv2.rectangle(vis, (x, y), (x+w, y+h), colour, 2)
        text = f"{label}  {conf:.0%}" if label else f"low conf  {conf:.0%}"
        cv2.putText(vis, text, (x, max(y-8, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0),    3, cv2.LINE_AA)
        cv2.putText(vis, text, (x, max(y-8, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(out_path), vis)


# =============================================================================
# Step 5 — Expected label lookup
# =============================================================================

def get_expected_labels(bag_name: str, frame_idx: int) -> list[str]:
    cursor = 0
    for n_frames, labels in EXPECTED.get(bag_name, []):
        if n_frames is None or frame_idx < cursor + n_frames:
            return sorted(labels)
        cursor += n_frames
    return []


# =============================================================================
# Per-bag processing
# =============================================================================

def process_bag(bag_name: str, model, device: torch.device) -> dict:
    img_dir   = extract_images(bag_name)
    img_files = sorted(img_dir.glob("*.png"))
    if not img_files:
        print(f"  [WARN] No images for {bag_name} — skipping.")
        return _empty_stats(bag_name)

    ann_dir = ANNOTATED_DIR / bag_name
    dbg_dir = DEBUG_DIR / bag_name
    ann_dir.mkdir(parents=True, exist_ok=True)
    dbg_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  [{bag_name}] Classifying {len(img_files)} frames ...")

    # Evenly spaced frame indices that get a debug image
    debug_indices = set(
        np.linspace(0, len(img_files)-1, DEBUG_FRAMES_PER_BAG, dtype=int).tolist()
    ) if DEBUG_FRAMES_PER_BAG > 0 else set()

    n_frames       = 0
    correct_frames = 0
    frame_results  = []

    for frame_idx, img_path in enumerate(img_files):
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"    [WARN] Cannot read {img_path.name}")
            continue

        # Detect orange cylinders
        boxes, sign_crops = detect_cylinders(image, K, dist_coeffs)
        boxes      = boxes[:MAX_SIGNS]
        sign_crops = sign_crops[:MAX_SIGNS]

        # Classify each crop
        pred_labels:    list[str | None] = []
        confs:          list[float]      = []
        all_probs_list: list[dict]       = []

        for crop in sign_crops:
            label, conf, all_probs = classify_crop(crop, model, device)
            pred_labels.append(label)
            confs.append(conf)
            all_probs_list.append(all_probs)

        pred_set = sorted(l for l in pred_labels if l is not None)

        # Save annotated output
        annotate_and_save(image, boxes, pred_labels, confs,
                          ann_dir / f"{frame_idx:06d}.jpg")

        # Save debug frame for selected indices
        if frame_idx in debug_indices:
            save_debug_frame(image, boxes, sign_crops, all_probs_list,
                             pred_labels, confs,
                             dbg_dir / f"debug_{frame_idx:06d}.jpg")

        # Accuracy
        exp_set = get_expected_labels(bag_name, frame_idx)
        match   = (pred_set == exp_set)
        n_frames       += 1
        correct_frames += int(match)
        frame_results.append((frame_idx, pred_set, exp_set, match))

        # Console progress every 100 frames — print full probability breakdown
        if frame_idx % 100 == 0:
            status = "OK  " if match else "FAIL"
            print(f"    [{frame_idx:5d}/{len(img_files)}]  "
                  f"pred={pred_set}  exp={exp_set}  {status}")
            if sign_crops:
                for ci, (lbl, conf, probs) in enumerate(
                        zip(pred_labels, confs, all_probs_list)):
                    ranked   = sorted(probs.items(), key=lambda x: -x[1])
                    prob_str = "  |  ".join(f"{n}: {p:.1%}" for n, p in ranked)
                    print(f"        crop {ci+1}: {prob_str}")
                    print(f"               --> final label: "
                          f"{lbl or 'NONE (below threshold)'} ({conf:.1%})")
            else:
                print("        --> no cylinders detected "
                      "(HSV mask found nothing — check debug_images/)")

    accuracy = correct_frames / n_frames if n_frames > 0 else 0.0
    print(f"\n  [{bag_name}] Done — accuracy: {accuracy*100:.1f}%  "
          f"({correct_frames}/{n_frames})")
    print(f"  Debug images: {dbg_dir}")

    return {
        "bag"           : bag_name,
        "n_frames"      : n_frames,
        "correct_frames": correct_frames,
        "accuracy"      : accuracy,
        "frame_results" : frame_results,
    }


def _empty_stats(bag_name: str) -> dict:
    return {
        "bag": bag_name, "n_frames": 0,
        "correct_frames": 0, "accuracy": 0.0, "frame_results": [],
    }


# =============================================================================
# Summary and results output
# =============================================================================

def print_summary(all_stats: list[dict]) -> None:
    hdr = f"{'Bag':<36}  {'Frames':>7}  {'Correct':>8}  {'Accuracy':>9}"
    sep = "-" * len(hdr)
    print(f"\n{sep}\n{hdr}\n{sep}")
    for s in all_stats:
        print(f"{s['bag']:<36}  {s['n_frames']:>7}  "
              f"{s['correct_frames']:>8}  {s['accuracy']*100:>8.1f}%")
    print(sep)
    total_f = sum(s["n_frames"]       for s in all_stats)
    total_c = sum(s["correct_frames"] for s in all_stats)
    overall = total_c / total_f * 100 if total_f > 0 else 0.0
    print(f"{'OVERALL':<36}  {total_f:>7}  {total_c:>8}  {overall:>8.1f}%")
    print(f"{sep}\n")


def save_results(all_stats: list[dict]) -> None:
    with open(RESULTS_TXT, "w") as f:
        f.write("Traffic Sign Classification Results\n")
        f.write("=" * 62 + "\n\n")
        f.write(f"{'Bag':<36}  {'Frames':>7}  {'Correct':>8}  {'Accuracy':>9}\n")
        f.write("-" * 62 + "\n")
        for s in all_stats:
            f.write(f"{s['bag']:<36}  {s['n_frames']:>7}  "
                    f"{s['correct_frames']:>8}  {s['accuracy']*100:>8.1f}%\n")
        total_f = sum(s["n_frames"]       for s in all_stats)
        total_c = sum(s["correct_frames"] for s in all_stats)
        overall = total_c / total_f * 100 if total_f > 0 else 0.0
        f.write("-" * 62 + "\n")
        f.write(f"{'OVERALL':<36}  {total_f:>7}  {total_c:>8}  {overall:>8.1f}%\n")
        f.write("\n\n=== Per-Frame Detail ===\n")
        for s in all_stats:
            f.write(f"\n--- {s['bag']} ---\n")
            for frame_idx, pred, exp, match in s["frame_results"]:
                f.write(f"  frame {frame_idx:6d}  pred={pred}  "
                        f"exp={exp}  {'PASS' if match else 'FAIL'}\n")

    with open(RESULTS_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bag", "frame", "predicted", "expected", "match"])
        for s in all_stats:
            for frame_idx, pred, exp, match in s["frame_results"]:
                w.writerow([s["bag"], frame_idx,
                             "; ".join(pred), "; ".join(exp),
                             "PASS" if match else "FAIL"])

    print(f"Results written to:\n  {RESULTS_TXT}\n  {RESULTS_CSV}")


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if not MODEL_PATH.exists():
        print(f"[ERROR] Model not found at {MODEL_PATH}")
        sys.exit(1)
    model = load_model(MODEL_PATH, device)

    EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)
    ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)

    all_stats: list[dict] = []
    for bag_name in BAG_NAMES:
        if not (DATA_ROOT / bag_name).exists():
            print(f"\n[SKIP] {bag_name} — not found at {DATA_ROOT / bag_name}")
            all_stats.append(_empty_stats(bag_name))
            continue
        all_stats.append(process_bag(bag_name, model, device))

    print_summary(all_stats)
    save_results(all_stats)