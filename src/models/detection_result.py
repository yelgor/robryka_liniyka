"""
detection_result.py
Contains results structure
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np


@dataclass
class DetectionResult:
    # Frame identifier (links result to a specific FramePacket).
    frame_id: int

    # Timestamp of the processed frame.
    timestamp: float

    # Whether the target object was detected.
    target_found: bool

    # Representative object point in pixel coordinates (cx, cy).
    support_point_px: Optional[Tuple[int, int]] = None

    # Object position in world coordinates (meters), Z = 0.
    world_coords: Optional[Tuple[float, float, float]] = None

    # Bounding box (x, y, w, h).
    bbox: Optional[Tuple[int, int, int, int]] = None

    # Binary segmentation mask.
    mask: Optional[np.ndarray] = None

    # Debug image with overlays (bbox, mask, point, labels).
    debug_image: Optional[np.ndarray] = None