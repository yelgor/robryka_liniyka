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
import cv2
import numpy as np


def segment_object(
    image: np.ndarray,
    lower_hsv: Optional[Tuple[int, int, int]] = None,
    upper_hsv: Optional[Tuple[int, int, int]] = None,
    min_area: int = 250,
) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int, int, int]]]:
    """
    Виділяє найбільш імовірний об'єкт на RGB-кадрі.

    :return:
        mask  - маска об'єкта або None
        bbox  - обмежувальний прямокутник або None
    """

    if image is None:
        raise ValueError("image must not be None")

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must have shape HxWx3")

    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    if not image.flags.c_contiguous:
        image = np.ascontiguousarray(image)

    height, width = image.shape[:2]
    effective_min_area = max(min_area, int(0.0004 * height * width))

    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    if lower_hsv is not None and upper_hsv is not None:
        mask = cv2.inRange(
            hsv,
            np.array(lower_hsv, dtype=np.uint8),
            np.array(upper_hsv, dtype=np.uint8),
        )
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = cv2.magnitude(grad_x, grad_y)
        grad_u8 = cv2.normalize(grad_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        grad_thr, _ = cv2.threshold(grad_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        grad_mask = (grad_u8 >= max(int(grad_thr) - 1, 13)).astype(np.uint8) * 255
        median_intensity = float(np.median(gray))
        low_thr = int(max(17, 0.68 * median_intensity))
        high_thr = int(min(240, max(low_thr + 20, 1.38 * median_intensity)))
        canny_mask = cv2.Canny(gray, low_thr, high_thr)
        mask = cv2.bitwise_or(grad_mask, canny_mask)

        kernel = np.ones((5, 5), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None

    border_margin = max(1, int(0.008 * min(width, height)))
    best_contour = None
    best_area = 0.0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < effective_min_area:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        if x <= border_margin or y <= border_margin or (x + w) >= (width - border_margin) or (y + h) >= (height - border_margin):
            continue

        aspect_ratio = w / float(h)
        if not (0.15 <= aspect_ratio <= 6.0):
            continue

        if area > best_area:
            best_area = area
            best_contour = contour

    if best_contour is None:
        return None, None

    x, y, w, h = cv2.boundingRect(best_contour)
    object_mask = np.zeros_like(mask)
    cv2.drawContours(object_mask, [best_contour], -1, 255, thickness=cv2.FILLED)

    return object_mask, (int(x), int(y), int(w), int(h))
