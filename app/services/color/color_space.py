"""Color space conversion functions."""

import numpy as np
from app.core.constants import WHITE_POINTS

def srgb_to_xyz(rgb: np.ndarray) -> np.ndarray:
    """Convert sRGB to XYZ color space (D65 illuminant)."""
    x = rgb.astype(float) / 255.0
    mask = x > 0.04045
    x[mask] = ((x[mask] + 0.055) / 1.055) ** 2.4
    x[~mask] = x[~mask] / 12.92
    x *= 100.0
    M = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]])
    return x @ M.T

def xyz_to_lab(xyz: np.ndarray, wp_name: str = "D65") -> np.ndarray:
    """Convert XYZ to Lab color space."""
    wp = WHITE_POINTS[wp_name]
    xr = xyz[...,0] / wp[0]
    yr = xyz[...,1] / wp[1]
    zr = xyz[...,2] / wp[2]
    delta = 6/29
    
    def f(t):
        return np.where(t > delta**3, np.cbrt(t), (t/(3*delta**2) + 4/29))
    
    fx, fy, fz = f(xr), f(yr), f(zr)
    L = 116*fy - 16
    a = 500*(fx - fy)
    b = 200*(fy - fz)
    return np.stack([L, a, b], axis=-1)

def rgb_to_cmyk(rgb: np.ndarray) -> np.ndarray:
    """Convert RGB to CMYK color space."""
    r, g, b = rgb[...,0]/255.0, rgb[...,1]/255.0, rgb[...,2]/255.0
    k = 1 - np.max(np.stack([r,g,b], axis=-1), axis=-1)
    denom = 1 - k + 1e-8
    c = (1 - r - k) / denom
    m = (1 - g - k) / denom
    y = (1 - b - k) / denom
    return np.stack([c,m,y,k], axis=-1)

