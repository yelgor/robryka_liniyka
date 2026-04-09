"""
geometry.py

Модуль геометричної інтерпретації результату CV.

Має відповідати за:
- перехід від піксельної точки до геометричної точки;
- подальшу реконструкцію положення цілі.
"""

from typing import Optional, Tuple


#TODO: CALIBRATION IS REQUIRED
FX = 600.0  # focal length x
FY = 600.0  # focal length y
CX = 320.0  # optical center x
CY = 240.0  # optical center y
CAMERA_HEIGHT_M = 0.5  # camera height above the table


def project_to_geometry(support_point_px: Optional[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
    """
    Geometric tranformation func.
    TODO: Not yet implemented
    """
    if support_point_px is None:
        return None
    
    u, v = support_point_px
    Z = CAMERA_HEIGHT_M
    X = (u - CX) * Z / FX
