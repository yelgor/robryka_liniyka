""" 
Module for geometric interpretation of the CV pipeline output.
 
Responsibilities:
- back-projecting a pixel coordinate into a 3-D ray in the camera frame;
- intersecting that ray with the horizontal ground plane Z_world = 0;
- returning the target position in the world coordinate frame.
"""

from typing import Optional, Tuple
import numpy as np



def project_to_geometry(
    support_point_px: Optional[Tuple[int, int]],
    K: np.ndarray,
    R: np.ndarray,
    t_vec: np.ndarray,
) -> Optional[Tuple[float, float, float]]:
    """
    Converts a pixel support point into 3-D world coordinates,
    assuming the target lies on the ground plane Z_world = 0.
 
    :param support_point_px: pixel coordinates (u, v) of the support point.
    :param K:     3*3 camera intrinsic matrix.
    :param R:     3*3 rotation matrix (world -> camera): P_c = R @ P_w + t_vec
    :param t_vec: translation vector (3,) — origin of the world frame
                  expressed in the camera frame.
    :return: (X, Y, Z) in metres in the world frame, or None.
    """
    if support_point_px is None:
        return None

    u, v = support_point_px

    pixel_homogeneous = np.array([u, v, 1.0])
    intrinsic_matrix_inv = np.linalg.inv(K)
    ray_direction_in_camera_frame = intrinsic_matrix_inv @ pixel_homogeneous

    rotation_transposed = R.T
    ray_direction_in_world_frame = rotation_transposed @ ray_direction_in_camera_frame
    camera_origin_in_world_frame = -(rotation_transposed @ t_vec)

    ray_world_z_component = float(ray_direction_in_world_frame[2])
    if abs(ray_world_z_component) < 1e-9:
        return None

    ray_parameter_at_ground_plane = -float(camera_origin_in_world_frame[2]) / ray_world_z_component
    if ray_parameter_at_ground_plane < 0:
        return None

    intersection_point_in_world = camera_origin_in_world_frame + ray_parameter_at_ground_plane * ray_direction_in_world_frame
    return (float(intersection_point_in_world[0]), float(intersection_point_in_world[1]), float(intersection_point_in_world[2]))
 





def build_camera_matrix(
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> np.ndarray:
    return np.array([
        [fx,  0, cx],
        [ 0, fy, cy],
        [ 0,  0,  1],
    ], dtype=np.float64)
 
 
def build_extrinsics(
    cam_pos_world: Tuple[float, float, float],
    R: np.ndarray,
) -> np.ndarray:
    """
    Computes the translation vector t_vec from the known camera position
    in the world frame and the rotation matrix R.
 
    Relation:  t_vec = -R · cam_pos_world
 
    :param cam_pos_world: (X, Y, Z) — camera position in the world frame.
    :param R: 3×3 rotation matrix mapping P_world -> P_camera.
    :return: t_vec, np.ndarray of shape (3,).
    """
    pos = np.array(cam_pos_world, dtype=np.float64)
    return -(R @ pos)
 
