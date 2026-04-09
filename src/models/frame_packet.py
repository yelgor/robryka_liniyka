"""
frame_packet.py
It contains input data for CV-module,
is created by camera module and transfered to CV pipeline.
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class FramePacket:
    # Unique frame ID (to understand which frame is in process)
    frame_id: int

    # Image that will be processed
    image: np.ndarray

    # timestamp of image
    # needed to control frames freshness
    timestamp: float
