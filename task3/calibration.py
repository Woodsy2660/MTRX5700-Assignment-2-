import numpy as np

# Camera intrinsic matrix
K = np.array([
    [483.04449161,   0.,         307.23210259],
    [  0.,         485.72258244, 260.49519651],
    [  0.,           0.,           1.        ]
])

# Distortion coefficients
dist = np.array([0.15327381, -0.32630592, 0.01920961, -0.0040156, 0.21145595])

# Camera to LiDAR 2D transform (x, y, yaw) as 3x3 SE(2) homogeneous matrix
# Transforms points FROM lidar frame TO camera frame (in 2D)
cam_T_lidar = np.array([
    [ 0.99911652,  0.04202605, -0.13067443],
    [-0.04202605,  0.99911652,  0.01125667],
    [ 0.,          0.,          1.        ]
])

# Assumed vertical (z) offset between lidar plane and camera optical centre (metres)
z_offset = 0.05
