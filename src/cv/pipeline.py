"""
pipeline.py

Головний вхідний модуль CV-частини.
Приймає FramePacket, викликає етапи обробки та повертає DetectionResult.
"""

from src.models.frame_packet import FramePacket
from src.models.detection_result import DetectionResult


class CVPipeline:
    def process(self, packet: FramePacket) -> DetectionResult:
        """
        Головна точка входу в CV-пайплайн.

        На цьому етапі реалізація є заглушкою.
        Далі тут буде виклик:
        - preprocess
        - segmentation
        - support point detection
        - geometry
        """

        return DetectionResult(
            frame_id=packet.frame_id,
            timestamp=packet.timestamp,
            target_found=False
        )