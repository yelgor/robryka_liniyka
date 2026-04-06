"""
preprocess.py

Модуль попередньої обробки зображення.

Має відповідати за:
- приведення зображення до потрібного формату;
- згладжування;
- шумопридушення;
- підготовку кадру до сегментації.
"""

import numpy as np


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Мінімальна підготовка RGB-кадру до сегментації.

    На цьому етапі тільки перевіряється і приводиться формат кадру,
    без додаткових фільтрів, щоб не псувати границі для edge-сегментації.
    """

    if image is None:
        raise ValueError("image must not be None")

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must have shape HxWx3")

    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    if not image.flags.c_contiguous:
        image = np.ascontiguousarray(image)

    return image

