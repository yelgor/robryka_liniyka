"""
segmentation.py

Модуль сегментації цільового об'єкта.

Має відповідати за:
- виділення області об'єкта;
- побудову маски;
- пошук bounding box;
- підготовку даних для вибору опорної точки.
"""

from typing import Optional, Tuple
import numpy as np


def segment_object(image: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int, int, int]]]:
    """
    Заглушка сегментації.

    :return:
        mask  - маска об'єкта або None
        bbox  - обмежувальний прямокутник або None
    """

    return None, None