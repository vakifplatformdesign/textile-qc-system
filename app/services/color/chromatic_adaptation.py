"""Chromatic adaptation functions."""

import numpy as np
from app.core.constants import M_BRADFORD, M_BRADFORD_INV

def adapt_white_xyz(xyz: np.ndarray, src_wp: np.ndarray, dst_wp: np.ndarray) -> np.ndarray:
    """
    Adapt XYZ tristimulus values from source to destination white point using Bradford transform.
    
    Args:
        xyz: XYZ tristimulus values
        src_wp: Source white point
        dst_wp: Destination white point
    
    Returns:
        Adapted XYZ values
    """
    src_lms = (M_BRADFORD @ xyz.reshape(-1,3).T).T
    src_wp_lms = M_BRADFORD @ src_wp
    dst_wp_lms = M_BRADFORD @ dst_wp
    D = (dst_wp_lms / src_wp_lms)
    dst_lms = (src_lms * D)
    out = (M_BRADFORD_INV @ dst_lms.T).T
    return out.reshape(xyz.shape)

