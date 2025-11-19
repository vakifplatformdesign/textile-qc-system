"""Image input/output utilities."""

import os
import numpy as np
from PIL import Image
from typing import Tuple

def validate_image_file(path: str) -> bool:
    """Validate that the file exists and is a valid image format."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image file not found: {path}")

    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    ext = os.path.splitext(path)[1].lower()
    if ext not in valid_extensions:
        raise ValueError(f"Unsupported image format: {ext}. Supported formats: {', '.join(valid_extensions)}")

    return True

def validate_image_dimensions(img: np.ndarray, min_size: int = 100, max_size: int = 10000) -> bool:
    """Validate image dimensions are within acceptable range."""
    h, w = img.shape[:2]
    if h < min_size or w < min_size:
        raise ValueError(f"Image too small: {w}x{h}. Minimum size: {min_size}x{min_size}")
    if h > max_size or w > max_size:
        raise ValueError(f"Image too large: {w}x{h}. Maximum size: {max_size}x{max_size}")
    return True

def read_rgb(path: str) -> np.ndarray:
    """Read RGB image with validation."""
    try:
        validate_image_file(path)
        img = Image.open(path).convert("RGB")
        arr = np.array(img)
        validate_image_dimensions(arr)
        return arr
    except Exception as e:
        raise RuntimeError(f"Failed to read image {path}: {str(e)}")

def save_image(img: np.ndarray, path: str) -> None:
    """Save numpy array as image."""
    Image.fromarray(img).save(path, "PNG")

