"""
Main entry point of the CV subsystem.

Stage order:
    1. preprocess    — normalise the frame format
    2. segmentation  — binary mask + bounding box of the object
    3. support_point — single representative pixel coordinate
    4. geometry      — 3-D world coordinates via ray-plane intersection (Z = 0)
"""

import cv2
import numpy as np

from src.cv.preprocess import preprocess_image
from src.cv.segmentation import segment_object
from src.cv.support_point import find_support_point
from src.cv.geometry import project_to_geometry
from src.models.frame_packet import FramePacket
from src.models.detection_result import DetectionResult


class CVPipeline:
    """
    Full CV pipeline: from a raw camera frame to 3-D coordinates of the cube.

    Camera parameters are supplied once at construction time and reused
    for every frame.
    """

    def __init__(
        self,
        intrinsic_matrix: np.ndarray,
        rotation_matrix: np.ndarray,
        translation_vector: np.ndarray,
    ) -> None:
        """
        :param intrinsic_matrix:   3*3 camera intrinsic matrix K.
        :param rotation_matrix:    3*3 rotation matrix R (world -> camera):
                                   P_camera = R @ P_world + t
        :param translation_vector: translation vector (3,) — world origin
                                   expressed in the camera frame.
        """
        self._intrinsic_matrix = intrinsic_matrix
        self._rotation_matrix = rotation_matrix
        self._translation_vector = translation_vector

    def process(self, frame_packet: FramePacket) -> DetectionResult:
        """
        Processes a single frame and returns a DetectionResult.

        Result fields:
            target_found     — True if the cube was detected
            support_point_px — support point in pixels (cx, cy)
            world_coords     — 3-D position in metres (X, Y, Z), Z is always 0
            bbox             — bounding box (x, y, w, h)
            mask             — binary segmentation mask
            debug_image      — frame with all detections drawn on top
        """

        # preprocess: ensure the frame is in the expected format.
        normalised_image = preprocess_image(frame_packet.image)

        # segmentation: locate the object and build its mask.
        object_mask, bounding_box = segment_object(normalised_image)

        target_found = bounding_box is not None

        # Stage 3 — support point: pick one representative pixel for the object.
        # Only runs when segmentation found something.
        support_point_in_pixels = None
        if target_found:
            support_point_in_pixels = find_support_point(object_mask, bounding_box)

        # Stage 4 — geometry: convert the pixel coordinate to metres on Z = 0.
        target_position_in_world = None
        if support_point_in_pixels is not None:
            target_position_in_world = project_to_geometry(
                support_point_in_pixels,
                K=self._intrinsic_matrix,
                R=self._rotation_matrix,
                t_vec=self._translation_vector,
            )

        # Debug visualisation — draw all results on a copy of the frame.
        debug_image = normalised_image.copy()

        # Overlay the segmentation mask.
        if object_mask is not None:
            mask_as_rgb = cv2.cvtColor(object_mask, cv2.COLOR_GRAY2RGB)
            debug_image = cv2.addWeighted(debug_image, 1.0, mask_as_rgb, 0.35, 0.0)

        # draw the bounding box.
        if bounding_box is not None:
            box_x, box_y, box_w, box_h = bounding_box
            cv2.rectangle(debug_image, (box_x, box_y), (box_x + box_w, box_y + box_h), (0, 255, 0), 2)

        # draw the support point.
        if support_point_in_pixels is not None:
            centre_x, centre_y = support_point_in_pixels
            cv2.circle(debug_image, (centre_x, centre_y), 6, (0, 0, 255), -1)
            cv2.circle(debug_image, (centre_x, centre_y), 8, (255, 255, 255), 1)

        # draw target position
        if target_position_in_world is not None and support_point_in_pixels is not None:
            world_x, world_y, world_z = target_position_in_world
            coordinate_label = f"({world_x:.3f}, {world_y:.3f}, {world_z:.3f}) m"
            centre_x, centre_y = support_point_in_pixels
            cv2.putText(
                debug_image, coordinate_label,
                (centre_x + 10, centre_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (255, 0, ), 1, cv2.LINE_AA,
            )

        return DetectionResult(
            frame_id=frame_packet.frame_id,
            timestamp=frame_packet.timestamp,
            target_found=target_found,
            support_point_px=support_point_in_pixels,
            world_coords=target_position_in_world,
            bbox=bounding_box,
            mask=object_mask,
            debug_image=debug_image,
        )
