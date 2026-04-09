"""
support_point.py

Module for selecting the support point of a detected object.

The support point is the centroid of the mask (when available),
or the geometric centre of the bounding box otherwise.
The mask centroid is more accurate because it accounts for the actual
shape of the object, not just the rectangle surrounding it.
"""

from typing import Optional, Tuple
import numpy as np


def find_support_point(
    mask: Optional[np.ndarray],
    bounding_box: Optional[Tuple[int, int, int, int]]
) -> Optional[Tuple[int, int]]:
    """
    Computes the support point of the object in pixel coordinates.

    Priority:
    1. If a mask is available — returns the mask centroid (weighted centre of mass).
       The centroid is computed via image moments, which is the standard way
       to find the "centre of gravity" of a binary region.
    2. If there is no mask but a bounding box is present — returns the geometric
       centre of the bounding box.
    3. If neither is available — returns None.

    :param mask: binary object mask (np.uint8, values 0/255) or None.
    :param bounding_box: bounding rectangle in (x, y, w, h) format or None.
    :return: support point coordinates (cx, cy) in pixels, or None.
    """

    if mask is not None:
        # if mask presented - calc "center of brightness"
        m = _compute_centroid(mask)
        if m is not None:
            return m

    # otherwise center of box
    if bounding_box is not None:
        x, y, w, h = bounding_box
        cx = x + w // 2
        cy = y + h // 2
        return (cx, cy)

    return None


def _compute_centroid(mask: np.ndarray) -> Optional[Tuple[int, int]]:
    """
    Computes the "center of brightnessmass" of a binary mask via spatial moments.

    :param mask: np.ndarray uint8, where 255 = object, 0 = background.
    :return: (cx, cy) or None if the mask is empty.
    """
    binary = (mask > 0).astype(np.float64)

    m00 = binary.sum()
    if m00 < 1.0:
        return None # Empty mask — nothing to compute.

    # np.indices returns arrays of row (y) and column (x) coordinates.
    rows, cols = np.indices(binary.shape)   # rows ~ y,  cols ~ x

    m10 = float((cols * binary).sum())
    m01 = float((rows * binary).sum())

    cx = int(round(m10 / m00))
    cy = int(round(m01 / m00))
    return (cx, cy)
