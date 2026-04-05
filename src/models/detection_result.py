"""
detection_result.py
Містить структуру результату обробки одного кадру CV-модулем.
Створюється після обробки FramePacket та передається далі в модулі геометрії, логіки або керування.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np


@dataclass
class DetectionResult:
    # Номер кадру, для якого отримано цей результат.
    # Потрібен, щоб пов’язати результат обробки з конкретним FramePacket.
    frame_id: int

    # Час кадру, по якому був отриманий результат.
    # Зазвичай копіюється з FramePacket.timestamp.
    timestamp: float

    # Прапорець наявності цільового об’єкта.
    # True  -> об’єкт знайдено
    # False -> об’єкт не знайдено
    target_found: bool

    # Опорна точка об’єкта в пікселях.
    # Це основна точка, з якою далі працює геометричний модуль.
    # Якщо об’єкт не знайдено, значення None.
    support_point_px: Optional[Tuple[int, int]] = None

    # Обмежувальний прямокутник об’єкта у форматі (x, y, w, h).
    # Використовується для відладки, візуалізації та простих евристик.
    bbox: Optional[Tuple[int, int, int, int]] = None

    # Бінарна маска об’єкта.
    # Потрібна, якщо CV-модуль явно виділяє область цільового об’єкта.
    mask: Optional[np.ndarray] = None

    # Відладочне зображення з накладеними результатами обробки:
    # прямокутником, маскою, точкою, підписами тощо.
    debug_image: Optional[np.ndarray] = None