"""
distance.py — LiDAR-based distance estimation for detected cylinders.

The sensor setup provides a 2D LiDAR scan (all points in a horizontal plane).
We are given a 2D SE(2) rigid-body transform `cam_T_lidar` that maps points
from the lidar frame into the camera's coordinate system (in 2D).

Coordinate conventions
----------------------
LiDAR frame  : x = forward, y = left,  z = up  (z ≈ 0 for 2D scan)
Camera frame : X = right,   Y = down,  Z = depth (forward along optical axis)

Mapping used here
-----------------
Given a lidar 2D point (x_L, y_L), apply the SE(2) transform to get
(x_C2D, y_C2D).  We then interpret this as a 3D camera-frame point:

    cam_X = x_C2D          (lateral / rightward)
    cam_Y = z_offset        (constant height difference lidar→camera)
    cam_Z = y_C2D           (depth into the scene)

This is then projected through the camera intrinsic matrix K to get a
pixel coordinate (u, v), which we use to assign LiDAR points to bounding boxes.

Distance to a cylinder is taken as the Euclidean 2D lidar range √(x²+y²) for
all points whose projection falls inside the bounding box, averaged over the
vertical strip containing the cylinder.
"""

import numpy as np
import open3d as o3d
from pathlib import Path


def load_lidar(pcd_path: str | Path) -> np.ndarray:
    """
    Load a .pcd file and return an (N, 2) array of (x, y) lidar points.
    Points with zero range are discarded.
    """
    pcd    = o3d.io.read_point_cloud(str(pcd_path))
    pts    = np.asarray(pcd.points)          # shape (N, 3)

    # Keep only points with non-zero range
    ranges = np.linalg.norm(pts[:, :2], axis=1)
    valid  = ranges > 0.05                   # discard origin / invalid returns
    return pts[valid, :2]                    # (N, 2) — x, y only


def _build_projection_matrix(cam_T_lidar: np.ndarray,
                              z_offset: float,
                              K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Build the camera-frame [R | t] that maps a 3D lidar-frame point
    (x_L, y_L, 0) to a 3D camera-frame point (cam_X, cam_Y, cam_Z).

    Coordinate conventions
    ----------------------
    LiDAR frame  : x = forward, y = left  (ground-plane 2D scan, z≈0)
    Camera frame : X = right, Y = down, Z = depth (forward)

    Derivation
    ----------
    The SE(2) matrix cam_T_lidar maps (x_L, y_L) → (x_C2D, y_C2D) where both
    frames share the same ground-plane convention (forward = x, left = y).
    Then:
        cam_Z =  x_C2D   (lidar forward  → camera depth)
        cam_X = -y_C2D   (lidar left     → negative camera right)
        cam_Y =  z_offset (constant height difference)

    Returns
    -------
    R_full : (3, 3) rotation part  — cam_pt = R_full @ lidar_3D + t_full
    t_full : (3,)   translation part
    """
    R2 = cam_T_lidar[:2, :2]   # 2x2 rotation
    t2 = cam_T_lidar[:2,  2]   # 2-vector translation

    # Row order: cam_X, cam_Y, cam_Z
    #   cam_X = -( R2[1,0]*x_L + R2[1,1]*y_L + t2[1] )
    #   cam_Y = z_offset   (constant, handled in t_full)
    #   cam_Z =    R2[0,0]*x_L + R2[0,1]*y_L + t2[0]
    R_full = np.array([
        [-R2[1, 0], -R2[1, 1], 0.0],   # cam X  ← -y_C2D
        [ 0.0,       0.0,      0.0],   # cam Y  ← constant
        [ R2[0, 0],  R2[0, 1], 0.0],   # cam Z  ←  x_C2D
    ])
    t_full = np.array([-t2[1], z_offset, t2[0]])

    return R_full, t_full


def project_lidar_to_image(lidar_xy: np.ndarray,
                            K:          np.ndarray,
                            cam_T_lidar: np.ndarray,
                            z_offset:    float
                            ) -> tuple[np.ndarray, np.ndarray]:
    """
    Project (N, 2) lidar points into image coordinates.

    Returns
    -------
    uv     : (M, 2) float array of (u, v) pixel coordinates for valid points.
    valid  : (N,)   bool mask — True for points that project in front of camera.
    ranges : (N,)   float array — Euclidean 2D range of each lidar point (m).
    """
    N = lidar_xy.shape[0]
    R_full, t_full = _build_projection_matrix(cam_T_lidar, z_offset, K)

    # Extend to homogeneous 3D in lidar frame: [x, y, 0]
    lidar_3D = np.hstack([lidar_xy, np.zeros((N, 1))])   # (N, 3)

    # Transform to camera frame
    cam_pts = (R_full @ lidar_3D.T).T + t_full           # (N, 3)

    # Only keep points in front of the camera (positive depth)
    valid = cam_pts[:, 2] > 0.01

    # Project through K: [u, v, 1] = K @ cam_pt / cam_pt[2]
    cam_valid = cam_pts[valid]                            # (M, 3)
    uvw       = (K @ cam_valid.T).T                      # (M, 3)
    uv        = uvw[:, :2] / uvw[:, 2:3]                 # (M, 2)

    # 2D lidar range for all points
    ranges = np.linalg.norm(lidar_xy, axis=1)            # (N,)

    return uv, valid, ranges


def estimate_distances(lidar_xy:    np.ndarray,
                       boxes:       list[tuple[int,int,int,int]],
                       K:           np.ndarray,
                       cam_T_lidar: np.ndarray,
                       z_offset:    float,
                       img_shape:   tuple[int,int] | None = None
                       ) -> list[dict]:
    """
    For each bounding box in *boxes*, find which projected lidar points fall
    inside it and compute range statistics.

    Parameters
    ----------
    lidar_xy    : (N, 2) array of lidar x, y points.
    boxes       : list of (x, y, w, h) bounding boxes in pixel coordinates.
    K           : 3x3 camera intrinsic matrix.
    cam_T_lidar : 3x3 SE(2) lidar→camera transform.
    z_offset    : vertical offset (metres) between lidar plane and camera.
    img_shape   : optional (height, width) to discard out-of-bounds projections.

    Returns
    -------
    results : list of dicts, one per box:
              {
                'mean_dist': float | None,
                'std_dist' : float | None,
                'n_points' : int,
              }
    """
    if lidar_xy.shape[0] == 0:
        return [{'mean_dist': None, 'std_dist': None, 'n_points': 0}
                for _ in boxes]

    uv, valid_mask, ranges = project_lidar_to_image(
        lidar_xy, K, cam_T_lidar, z_offset)

    # Indices of valid projected points in the original array
    valid_idx = np.where(valid_mask)[0]   # maps back to lidar_xy / ranges

    results = []
    for (x, y, w, h) in boxes:
        # Find projected points inside this box
        in_box = (
            (uv[:, 0] >= x) & (uv[:, 0] < x + w) &
            (uv[:, 1] >= y) & (uv[:, 1] < y + h)
        )

        if img_shape is not None:
            H, W = img_shape
            in_bounds = (
                (uv[:, 0] >= 0) & (uv[:, 0] < W) &
                (uv[:, 1] >= 0) & (uv[:, 1] < H)
            )
            in_box = in_box & in_bounds

        box_valid_idx = valid_idx[in_box]   # original indices of matched pts

        if box_valid_idx.size == 0:
            results.append({'mean_dist': None, 'std_dist': None,
                            'n_points': 0})
            continue

        box_ranges = ranges[box_valid_idx]
        results.append({
            'mean_dist': float(np.mean(box_ranges)),
            'std_dist' : float(np.std(box_ranges)),
            'n_points' : int(box_valid_idx.size),
        })

    return results
