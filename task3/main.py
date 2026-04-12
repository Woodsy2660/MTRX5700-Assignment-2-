"""
main.py — Task 3 pipeline: detect orange cylinders, estimate LiDAR distance,
          and (placeholder) classify traffic signs.

Usage:
    python task3/main.py

Output images are saved to task3/output/<bag_name>/.
A summary table is printed to stdout when all bags are processed.
"""

import sys
import os
from pathlib import Path

# Allow imports from the task3 package regardless of working directory
sys.path.insert(0, str(Path(__file__).parent))

import cv2
import numpy as np

from calibration import K, dist, cam_T_lidar, z_offset
from detect     import detect_cylinders, save_detection_image
from distance   import load_lidar, estimate_distances

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_ROOT = Path.home() / "extracted_data" / "extracted"
OUTPUT_DIR = Path(__file__).parent / "output"

BAG_NAMES = [
    "traffic_sign_left_1",
    "traffic_sign_left_2",
    "traffic_sign_right_1",
    "traffic_sign_right_2",
    "traffic_signs_multiple_1",
    "traffic_signs_multiple_2",
    "traffic_signs_multiple_3",
]

# ── Detection parameters ─────────────────────────────────────────────────────
# A frame counts as a "detection success" if at least this many cylinders
# were found.  Single-cylinder bags use 1; multiple-cylinder bags may use 2.
MIN_CYLINDERS_FOR_SUCCESS = 1

# Save at most this many annotated images per bag (to avoid filling disk)
MAX_SAVED_IMAGES = 10


# ── Placeholder: sign classification ─────────────────────────────────────────
def classify_sign(crop: np.ndarray) -> str | None:
    """
    Placeholder for traffic sign classification.

    TODO: Load a trained PyTorch model and replace this function body with
          a real inference call, e.g.:

              transform = ...
              tensor    = transform(crop).unsqueeze(0).to(device)
              with torch.no_grad():
                  logits = model(tensor)
              return CLASS_NAMES[logits.argmax().item()]

    Returns None until the model is wired up.
    """
    return None   # ← replace with model inference


# ── Per-bag processing ────────────────────────────────────────────────────────
def process_bag(bag_name: str) -> dict:
    """
    Run the full pipeline on one bag and return a statistics dictionary.

    Returns
    -------
    stats : {
        'bag'          : str,
        'n_frames'     : int,   total image/lidar pairs processed
        'det_success'  : int,   frames with >= MIN_CYLINDERS_FOR_SUCCESS detections
        'det_rate'     : float, detection success rate (0–1)
        'all_dists'    : list[float], all valid per-cylinder mean distances
        'mean_dist'    : float | None,
        'std_dist'     : float | None,
        'cls_success'  : int,   correctly classified signs (always 0 until model)
        'cls_attempts' : int,   total sign crops passed to classifier
    }
    """
    img_dir   = DATA_ROOT / bag_name / "images" / "images"
    lidar_dir = DATA_ROOT / bag_name / "lidar"
    out_dir   = OUTPUT_DIR / bag_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect sorted file lists
    img_files   = sorted(img_dir.glob("*.png"))
    lidar_files = sorted(lidar_dir.glob("*.pcd"))

    if not img_files:
        print(f"  [WARN] No images found in {img_dir}")
        return _empty_stats(bag_name)
    if not lidar_files:
        print(f"  [WARN] No LiDAR scans found in {lidar_dir}")
        return _empty_stats(bag_name)

    # Pair images and lidar scans by index.
    # There are typically ~3x more images than scans; we sample images evenly
    # so that each lidar scan is paired with one image.
    n_pairs  = min(len(img_files), len(lidar_files))
    img_step = max(len(img_files) // n_pairs, 1)

    sampled_imgs = [img_files[i * img_step] for i in range(n_pairs)]
    paired       = list(zip(sampled_imgs, lidar_files[:n_pairs]))

    print(f"  Bag: {bag_name} — {len(paired)} pairs "
          f"(from {len(img_files)} imgs, {len(lidar_files)} scans)")

    n_frames      = 0
    det_success   = 0
    all_dists     : list[float] = []
    cls_attempts  = 0
    cls_success   = 0
    saved_count   = 0

    for idx, (img_path, pcd_path) in enumerate(paired):

        # ── Load image ───────────────────────────────────────────────────────
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"    [WARN] Could not read {img_path.name}, skipping.")
            continue

        # ── Load LiDAR scan ──────────────────────────────────────────────────
        try:
            lidar_xy = load_lidar(pcd_path)
        except Exception as e:
            print(f"    [WARN] Could not load {pcd_path.name}: {e}, skipping.")
            continue

        n_frames += 1

        # ── Detect cylinders ─────────────────────────────────────────────────
        boxes, sign_crops = detect_cylinders(image, K, dist)

        if len(boxes) >= MIN_CYLINDERS_FOR_SUCCESS:
            det_success += 1

        if not boxes:
            # Nothing to do for this frame
            if idx % 50 == 0:
                print(f"    [{idx:4d}/{len(paired)}] No detections.")
            continue

        # ── Estimate distances ───────────────────────────────────────────────
        h, w = image.shape[:2]
        dist_results = estimate_distances(
            lidar_xy, boxes, K, cam_T_lidar, z_offset,
            img_shape=(h, w)
        )

        frame_dists : list[float | None] = []
        for dr in dist_results:
            if dr['mean_dist'] is not None:
                all_dists.append(dr['mean_dist'])
                frame_dists.append(dr['mean_dist'])
            else:
                frame_dists.append(None)

        # ── Classify signs ───────────────────────────────────────────────────
        frame_labels : list[str | None] = []
        for crop in sign_crops:
            cls_attempts += 1
            label = classify_sign(crop)
            frame_labels.append(label)
            if label is not None:
                cls_success += 1

        # ── Progress print ───────────────────────────────────────────────────
        if idx % 50 == 0:
            dist_str = (f"dists={[f'{d:.2f}' if d else 'N/A' for d in frame_dists]}"
                        if frame_dists else "no dists")
            print(f"    [{idx:4d}/{len(paired)}] {len(boxes)} cylinder(s)  {dist_str}")

        # ── Save annotated image (first MAX_SAVED_IMAGES only) ───────────────
        if saved_count < MAX_SAVED_IMAGES:
            out_path = out_dir / f"detection_{idx:04d}.jpg"
            save_detection_image(image, boxes, out_path,
                                 dists=frame_dists,
                                 labels=frame_labels)
            saved_count += 1

    # ── Bag-level statistics ─────────────────────────────────────────────────
    det_rate  = det_success / n_frames if n_frames > 0 else 0.0
    mean_dist = float(np.mean(all_dists))  if all_dists else None
    std_dist  = float(np.std(all_dists))   if all_dists else None

    return {
        'bag'         : bag_name,
        'n_frames'    : n_frames,
        'det_success' : det_success,
        'det_rate'    : det_rate,
        'all_dists'   : all_dists,
        'mean_dist'   : mean_dist,
        'std_dist'    : std_dist,
        'cls_success' : cls_success,
        'cls_attempts': cls_attempts,
    }


def _empty_stats(bag_name: str) -> dict:
    return {
        'bag': bag_name, 'n_frames': 0, 'det_success': 0,
        'det_rate': 0.0, 'all_dists': [], 'mean_dist': None,
        'std_dist': None, 'cls_success': 0, 'cls_attempts': 0,
    }


# ── Summary table ─────────────────────────────────────────────────────────────
def print_summary(all_stats: list[dict]) -> None:
    hdr = (f"{'Bag':<32} {'Frames':>7} {'Det%':>7} "
           f"{'Mean dist':>10} {'Std dist':>9} "
           f"{'Cls%':>7}")
    sep = "-" * len(hdr)
    print("\n" + sep)
    print(hdr)
    print(sep)
    for s in all_stats:
        det_pct  = f"{s['det_rate']*100:.1f}%"
        mean_str = f"{s['mean_dist']:.3f} m" if s['mean_dist'] is not None else "  N/A   "
        std_str  = f"{s['std_dist']:.3f} m"  if s['std_dist']  is not None else "  N/A  "
        cls_pct  = (f"{s['cls_success']/s['cls_attempts']*100:.1f}%"
                    if s['cls_attempts'] > 0 else "  N/A")
        print(f"{s['bag']:<32} {s['n_frames']:>7} {det_pct:>7} "
              f"{mean_str:>10} {std_str:>9} {cls_pct:>7}")
    print(sep)

    # Aggregate across all bags
    total_frames = sum(s['n_frames']    for s in all_stats)
    total_det    = sum(s['det_success'] for s in all_stats)
    all_d        = [d for s in all_stats for d in s['all_dists']]
    total_cls_a  = sum(s['cls_attempts'] for s in all_stats)
    total_cls_s  = sum(s['cls_success']  for s in all_stats)

    overall_det_pct  = f"{total_det/total_frames*100:.1f}%" if total_frames else "N/A"
    overall_mean     = f"{np.mean(all_d):.3f} m" if all_d else "  N/A   "
    overall_std      = f"{np.std(all_d):.3f} m"  if all_d else "  N/A  "
    overall_cls_pct  = (f"{total_cls_s/total_cls_a*100:.1f}%"
                        if total_cls_a > 0 else "  N/A")

    print(f"{'OVERALL':<32} {total_frames:>7} {overall_det_pct:>7} "
          f"{overall_mean:>10} {overall_std:>9} {overall_cls_pct:>7}")
    print(sep + "\n")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR.resolve()}\n")

    all_stats = []
    for bag in BAG_NAMES:
        bag_path = DATA_ROOT / bag
        if not bag_path.exists():
            print(f"[SKIP] {bag} — directory not found at {bag_path}")
            all_stats.append(_empty_stats(bag))
            continue
        stats = process_bag(bag)
        all_stats.append(stats)

        # Per-bag summary line
        md = f"{stats['mean_dist']:.3f} m" if stats['mean_dist'] is not None else "N/A"
        sd = f"{stats['std_dist']:.3f} m"  if stats['std_dist']  is not None else "N/A"
        print(f"  → det rate: {stats['det_rate']*100:.1f}%  "
              f"mean dist: {md}  std dist: {sd}\n")

    print_summary(all_stats)
