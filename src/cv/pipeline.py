"""
pipeline.py

Головний вхідний модуль CV-частини.
Приймає FramePacket, викликає етапи обробки та повертає DetectionResult.
"""

import cv2

from src.cv.preprocess import preprocess_image
from src.cv.segmentation import segment_object
from src.models.frame_packet import FramePacket
from src.models.detection_result import DetectionResult


class CVPipeline:
    def process(self, packet: FramePacket) -> DetectionResult:
        """
        Головна точка входу в CV-пайплайн.

        На цьому етапі виконується:
        - preprocess
        - segmentation
        """

        preprocessed_image = preprocess_image(packet.image)
        mask, bbox = segment_object(preprocessed_image)

        debug_image = preprocessed_image.copy()
        target_found = bbox is not None

        if mask is not None:
            overlay = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
            debug_image = cv2.addWeighted(debug_image, 1.0, overlay, 0.35, 0.0)

        if bbox is not None:
            x, y, w, h = bbox
            cv2.rectangle(debug_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        return DetectionResult(
            frame_id=packet.frame_id,
            timestamp=packet.timestamp,
            target_found=target_found,
            bbox=bbox,
            mask=mask,
            debug_image=debug_image,
        )
