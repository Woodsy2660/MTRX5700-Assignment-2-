"""
ros_node.py — ROS2 node for Task 3: cylinder detection, distance estimation,
              and (placeholder) traffic sign classification.

Run while playing a ROS bag:
    Terminal 1:  ros2 bag play <path_to_bag>
    Terminal 2:  python3 task3/ros_node.py

Topics subscribed:
    /camera/image_raw   — sensor_msgs/Image
    /pointcloud2d       — sensor_msgs/PointCloud2

The node time-synchronises both topics, runs the full pipeline on each matched
pair, and prints a per-bag summary table when shut down (Ctrl-C).
"""

import sys
import signal
from pathlib import Path

# Allow imports from the task3 package regardless of working directory
sys.path.insert(0, str(Path(__file__).parent))

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from cv_bridge import CvBridge
import message_filters
import sensor_msgs_py.point_cloud2 as pc2

from calibration import K, dist, cam_T_lidar, z_offset
from detect     import detect_cylinders, save_detection_image
from distance   import estimate_distances


# ── Classification placeholder ────────────────────────────────────────────────
def classify_sign(crop: np.ndarray) -> str | None:
    """
    Placeholder for traffic sign classification.

    TODO: replace with PyTorch model inference, e.g.:
        tensor = transform(crop).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(tensor)
        return CLASS_NAMES[logits.argmax().item()]
    """
    return None


# ── ROS2 Node ─────────────────────────────────────────────────────────────────
class CylinderDetectorNode(Node):

    # Save at most this many annotated images per run (to avoid filling disk)
    MAX_SAVED = 20

    def __init__(self):
        super().__init__('cylinder_detector')

        self.bridge = CvBridge()

        # ── Synchronised subscribers ─────────────────────────────────────────
        img_sub   = message_filters.Subscriber(self, Image,       '/camera/image_raw')
        lidar_sub = message_filters.Subscriber(self, PointCloud2, '/pointcloud2d')

        # Allow up to 0.1 s difference between matched stamps
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [img_sub, lidar_sub], queue_size=10, slop=0.1)
        self.sync.registerCallback(self.callback)

        # ── Statistics ───────────────────────────────────────────────────────
        self.n_frames     = 0
        self.det_success  = 0   # frames with >= 1 cylinder detected
        self.all_dists    : list[float] = []
        self.cls_attempts = 0
        self.cls_success  = 0
        self.saved_count  = 0

        # Output directory for annotated images
        self.out_dir = Path(__file__).parent / "output" / "ros"
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.get_logger().info(
            "CylinderDetectorNode started — waiting for synchronised "
            "/camera/image_raw + /pointcloud2d …")

    # ── Callback ──────────────────────────────────────────────────────────────
    def callback(self, img_msg: Image, pc_msg: PointCloud2) -> None:
        """Called once per time-synchronised image + lidar pair."""

        # ── Convert image ────────────────────────────────────────────────────
        try:
            image = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().warn(f"Image conversion failed: {e}")
            return

        # ── Convert PointCloud2 → (N, 2) xy array ───────────────────────────
        try:
            pts = pc2.read_points_numpy(pc_msg, field_names=('x', 'y'),
                                        skip_nans=True)
            # read_points_numpy may return a structured array — flatten to (N,2)
            if pts.dtype.names:
                lidar_xy = np.column_stack([pts['x'], pts['y']])
            else:
                lidar_xy = pts[:, :2]
            # Discard zero-range points
            ranges   = np.linalg.norm(lidar_xy, axis=1)
            lidar_xy = lidar_xy[ranges > 0.05]
        except Exception as e:
            self.get_logger().warn(f"PointCloud2 conversion failed: {e}")
            return

        self.n_frames += 1

        # ── Detect cylinders ─────────────────────────────────────────────────
        boxes, sign_crops = detect_cylinders(image, K, dist)

        if len(boxes) >= 1:
            self.det_success += 1

        if not boxes:
            if self.n_frames % 20 == 0:
                self.get_logger().info(
                    f"[{self.n_frames:4d}] No cylinders detected.")
            return

        # ── Estimate distances ───────────────────────────────────────────────
        h, w = image.shape[:2]
        dist_results = estimate_distances(
            lidar_xy, boxes, K, cam_T_lidar, z_offset, img_shape=(h, w))

        frame_dists : list[float | None] = []
        for dr in dist_results:
            if dr['mean_dist'] is not None:
                self.all_dists.append(dr['mean_dist'])
                frame_dists.append(dr['mean_dist'])
            else:
                frame_dists.append(None)

        # ── Classify signs ───────────────────────────────────────────────────
        frame_labels : list[str | None] = []
        for crop in sign_crops:
            self.cls_attempts += 1
            label = classify_sign(crop)
            frame_labels.append(label)
            if label is not None:
                self.cls_success += 1

        # ── Log progress ─────────────────────────────────────────────────────
        dist_str = [f"{d:.2f}m" if d is not None else "N/A"
                    for d in frame_dists]
        self.get_logger().info(
            f"[{self.n_frames:4d}] {len(boxes)} cylinder(s)  "
            f"dists={dist_str}  labels={frame_labels}")

        # ── Save annotated image ─────────────────────────────────────────────
        if self.saved_count < self.MAX_SAVED:
            out_path = self.out_dir / f"frame_{self.n_frames:05d}.jpg"
            save_detection_image(image, boxes, out_path,
                                 dists=frame_dists, labels=frame_labels)
            self.saved_count += 1

    # ── Summary on shutdown ───────────────────────────────────────────────────
    def print_summary(self) -> None:
        print("\n" + "=" * 55)
        print("  TASK 3 — RUN SUMMARY")
        print("=" * 55)
        print(f"  Frames processed   : {self.n_frames}")

        det_rate = (self.det_success / self.n_frames * 100
                    if self.n_frames > 0 else 0.0)
        print(f"  Detection rate     : {det_rate:.1f}%  "
              f"({self.det_success}/{self.n_frames} frames)")

        if self.all_dists:
            print(f"  Mean distance      : {np.mean(self.all_dists):.3f} m")
            print(f"  Std  distance      : {np.std(self.all_dists):.3f} m")
        else:
            print("  Mean/Std distance  : N/A")

        if self.cls_attempts > 0:
            cls_rate = self.cls_success / self.cls_attempts * 100
            print(f"  Classification rate: {cls_rate:.1f}%  "
                  f"({self.cls_success}/{self.cls_attempts})")
        else:
            print("  Classification rate: N/A (model not wired up yet)")

        print(f"  Annotated images   : {self.out_dir.resolve()}")
        print("=" * 55 + "\n")


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    rclpy.init()
    node = CylinderDetectorNode()

    # Print summary on Ctrl-C
    def shutdown(sig, frame):
        node.print_summary()
        node.destroy_node()
        rclpy.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.print_summary()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
