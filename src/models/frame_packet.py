"""
frame_packet.py
Містить структуру вхідних даних для CV-модуля.
Створюється на стороні камери робота та передається далі в pipeline CV-модуля.
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class FramePacket:
    # Унікальний номер кадру.
    # Потрібен, щоб розуміти, який саме кадр зараз обробляється.
    frame_id: int

    # Саме зображення, з яким далі працює CV-модуль.
    image: np.ndarray

    # Час отримання кадру.
    # Потрібен для контролю свіжості кадру та відладки таймінгів.
    timestamp: float