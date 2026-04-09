import cv2
import numpy as np
import time
import os

from src.cv.pipeline import CVPipeline
from src.models.frame_packet import FramePacket


def main():
    camera_intrinsic_matrix = np.array([
        [800, 0, 320],
        [0, 800, 240],
        [0, 0, 1]
    ], dtype=float)


    # HELPER FUNCTIONS
    def rotation_matrix_x(angle_rad):
        return np.array([
            [1, 0, 0],
            [0, np.cos(angle_rad), -np.sin(angle_rad)],
            [0, np.sin(angle_rad),  np.cos(angle_rad)],
        ])

    def rotation_matrix_z(angle_rad):
        return np.array([
            [np.cos(angle_rad), -np.sin(angle_rad), 0],
            [np.sin(angle_rad),  np.cos(angle_rad), 0],
            [0, 0, 1],
        ])

    # it's easier to calculate world_to_camera_rotation_matrix as
    # multiplication of of two rotation matrices in different axiis
    Rx = rotation_matrix_x(np.deg2rad(50))
    Rz = rotation_matrix_z(np.deg2rad(20))

    world_to_camera_rotation_matrix = Rz @ Rx

    camera_translation_in_world = np.array([0.0, 0.0, 0.12])

    pipeline = CVPipeline(
        camera_intrinsic_matrix,
        world_to_camera_rotation_matrix,
        camera_translation_in_world
    )

    # --- Load image ---
    project_root_directory = os.path.dirname(os.path.dirname(__file__))
    input_image_path = os.path.join(
        project_root_directory,
        "assets",
        "photo_2026-04-07_23-54-14.jpg"
    )

    input_image_bgr = cv2.imread(input_image_path)
    if input_image_bgr is None:
        raise RuntimeError("Image not found")

    input_image_rgb = cv2.cvtColor(input_image_bgr, cv2.COLOR_BGR2RGB)

    frame_packet = FramePacket(
        frame_id=0,
        image=input_image_rgb,
        timestamp=time.time()
    )

    detection_result = pipeline.process(frame_packet)

    print("target_found:", detection_result.target_found)
    print("bbox:", detection_result.bbox)
    print("world_coords:", detection_result.world_coords)

    debug_image_bgr = cv2.cvtColor(detection_result.debug_image, cv2.COLOR_RGB2BGR)
    cv2.imshow("debug", debug_image_bgr)
    cv2.waitKey(0)


if __name__ == "__main__":
    main()