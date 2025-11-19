"""Helper utility functions."""

import os
import numpy as np
from datetime import datetime, timedelta
from app.core.config import TIMEZONE_OFFSET_HOURS

def get_local_time():
    """Get current time with Turkey timezone offset."""
    return datetime.now() + timedelta(hours=TIMEZONE_OFFSET_HOURS)

def grid_points(h: int, w: int, n: int = 5) -> list:
    """Generate grid of sample points."""
    ys = np.linspace(0.2, 0.8, n)
    xs = np.linspace(0.2, 0.8, n)
    pts = [(int(y*h), int(x*w)) for y, x in zip(ys, xs)]
    return pts[:n]

def ensure_dir(p: str) -> str:
    """Create directory if it doesn't exist."""
    os.makedirs(p, exist_ok=True)
    return p

def pick_logo() -> str:
    """Pick the best available logo file."""
    from app.core.config import PRIMARY_LOGO, FALLBACK_LOGOS
    if os.path.exists(PRIMARY_LOGO):
        return PRIMARY_LOGO
    for p in FALLBACK_LOGOS:
        if os.path.exists(p):
            return p
    return None

def fmt_pct(x: float) -> str:
    """Format as percentage."""
    return f"{x:.1f}%"

def fmt2(x: float) -> str:
    """Format to 2 decimal places."""
    return f"{x:.2f}"

def fmt1(x: float) -> str:
    """Format to 1 decimal place."""
    return f"{x:.1f}"

def determine_status(value: float, pass_threshold: float, 
                    conditional_threshold: float, lower_is_better: bool = True) -> str:
    """
    Unified status determination function.
    
    Args:
        value: The metric value to evaluate
        pass_threshold: Threshold for PASS status
        conditional_threshold: Threshold for CONDITIONAL status
        lower_is_better: If True, lower values are better (e.g., ΔE). 
                        If False, higher is better (e.g., SSIM)
    
    Returns:
        str: "PASS", "CONDITIONAL", or "FAIL"
    """
    if lower_is_better:
        if value < pass_threshold:
            return "PASS"
        elif value <= conditional_threshold:
            return "CONDITIONAL"
        else:
            return "FAIL"
    else:
        if value > pass_threshold:
            return "PASS"
        elif value > conditional_threshold:
            return "CONDITIONAL"
        else:
            return "FAIL"

