"""Whiteness and yellowness index calculations."""

import numpy as np

def cie_whiteness_tint(xyz: np.ndarray, illuminant: str = 'D65') -> tuple:
    """
    CIE Whiteness and Tint (ISO 11475) for illuminant D65 with 10° observer.
    
    Args:
        xyz: XYZ tristimulus values
        illuminant: Illuminant name (default: D65)
    
    Returns:
        Tuple of (whiteness, tint)
    """
    X, Y, Z = xyz[...,0], xyz[...,1], xyz[...,2]
    
    # Chromaticity coordinates
    sum_XYZ = np.maximum(X + Y + Z, 1e-8)
    x = X / sum_XYZ
    y = Y / sum_XYZ
    
    # CIE Whiteness (D65, 10°) - ISO 11475
    # Reference white point for D65/10°: xn=0.3138, yn=0.3310
    xn, yn = 0.3138, 0.3310
    W = Y + 800 * (xn - x) + 1700 * (yn - y)
    
    # Tint
    T = 900 * (xn - x) - 650 * (yn - y)
    
    return W, T

def astm_e313_yellowness(xyz: np.ndarray) -> np.ndarray:
    """
    ASTM E313 Yellowness Index.
    
    Args:
        xyz: XYZ tristimulus values
    
    Returns:
        Yellowness index value
    """
    X, Y, Z = xyz[...,0], xyz[...,1], xyz[...,2]
    
    # Coefficients for D65/10° (newer standard)
    C_x = 1.3013
    C_z = 1.1498
    
    YI = 100 * (C_x * X - C_z * Z) / np.maximum(Y, 1e-8)
    
    return YI

