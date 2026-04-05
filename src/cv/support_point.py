"""
support_point.py

Модуль вибору опорної точки об'єкта.

Має відповідати за:
- аналіз маски або bounding box;
- вибір точки, з якою далі працює геометричний модуль.
"""

from typing import Optional, Tuple
import numpy as np


def find_support_point(
    mask: Optional[np.ndarray],
    bbox: Optional[Tuple[int, int, int, int]]
) -> Optional[Tuple[int, int]]:
    """
    Заглушка вибору опорної точки.

    :return:
        Координати точки в пікселях або None.
    """

    return None