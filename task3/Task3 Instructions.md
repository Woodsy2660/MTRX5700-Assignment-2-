# Task 3 Pipeline — Usage Guide

A pipeline for detecting orange cylinders in camera frames, estimating their distance using LiDAR point clouds, and (optionally) classifying traffic signs.

## Prerequisites

### Directory Structure

The pipeline expects extracted ROS bag data at the following path:

```
~/extracted_data/extracted/
├── traffic_sign_left_1/
│   ├── images/images/*.png
│   └── lidar/*.pcd
├── traffic_sign_left_2/
│   ├── images/images/*.png
│   └── lidar/*.pcd
└── ... (other bags)
```

The following bag names are processed by default:

| Bag Name |
|---|
| `traffic_sign_left_1` |
| `traffic_sign_left_2` |
| `traffic_sign_right_1` |
| `traffic_sign_right_2` |
| `traffic_signs_multiple_1` |
| `traffic_signs_multiple_2` |
| `traffic_signs_multiple_3` |

### Python Dependencies

Ensure the following packages are installed:

```bash
pip install opencv-python numpy
```

> If using the sign classification feature, also install PyTorch and any model-specific dependencies.

---

## Running the Pipeline

From the **project root**, run:

```bash
python task3/main.py
```
---

## Output

### Annotated Images

Saved to:

```
task3/output/<bag_name>/detection_XXXX.jpg
```

Each image shows detected cylinder bounding boxes, estimated distances, and (when available) sign classification labels. A maximum of **10 images per bag** are saved.

### Console Summary Table

After all bags are processed, a summary table is printed:

```
Bag                              Frames    Det%   Mean dist   Std dist    Cls%
--------------------------------------------------------------------------------
traffic_sign_left_1                  42   95.2%    3.141 m    0.452 m     N/A
...
OVERALL                             294   88.4%    3.056 m    0.511 m     N/A
```

| Column | Description |
|---|---|
| `Frames` | Total image–LiDAR pairs processed |
| `Det%` | Percentage of frames with ≥ 1 cylinder detected |
| `Mean dist` | Mean estimated distance across all detected cylinders |
| `Std dist` | Standard deviation of estimated distances |
| `Cls%` | Sign classification accuracy (requires a trained model) |

---

## Configuration

Key parameters are defined at the top of `main.py`:

| Parameter | Default | Description |
|---|---|---|
| `DATA_ROOT` | `~/extracted_data/extracted` | Root directory for bag data |
| `OUTPUT_DIR` | `task3/output/` | Directory for saved output images |
| `MIN_CYLINDERS_FOR_SUCCESS` | `1` | Minimum detections to count a frame as a success |
| `MAX_SAVED_IMAGES` | `10` | Maximum annotated images saved per bag |

To add or remove bags, edit the `BAG_NAMES` list:

```python
BAG_NAMES = [
    "traffic_sign_left_1",
    "my_custom_bag",   # ← add new bags here
    ...
]
```

## Module Overview

| Module | Responsibility |
|---|---|
| `main.py` | Pipeline orchestration, summary reporting |
| `calibration.py` | Camera intrinsics (`K`, `dist`), extrinsics (`cam_T_lidar`), z-offset |
| `detect.py` | Orange cylinder detection, bounding box extraction, image saving |
| `distance.py` | LiDAR `.pcd` loading, point projection, distance estimation |
