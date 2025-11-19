"""Image processing utilities."""

import numpy as np
import cv2
from typing import Tuple
from app.core.settings import QCSettings

def to_same_size(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Resize two images to the same size (minimum of both)."""
    h = min(a.shape[0], b.shape[0])
    w = min(a.shape[1], b.shape[1])
    a2 = cv2.resize(a, (w, h), interpolation=cv2.INTER_AREA)
    b2 = cv2.resize(b, (w, h), interpolation=cv2.INTER_AREA)
    return a2, b2

def apply_mask_to_image(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Apply a binary mask to an image (supports grayscale and color)."""
    if len(img.shape) == 3:
        mask_3ch = np.stack([mask, mask, mask], axis=-1)
        return np.where(mask_3ch > 0, img, 0)
    else:
        return np.where(mask > 0, img, 0)

def apply_circular_crop(img: np.ndarray, center_x: int, center_y: int, diameter: int) -> np.ndarray:
    """Apply circular crop to image, masking outside as black."""
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    radius = diameter // 2
    cv2.circle(mask, (center_x, center_y), radius, 255, -1)
    return apply_mask_to_image(img, mask)

def apply_rectangular_crop(img: np.ndarray, center_x: int, center_y: int, 
                          width: int, height: int) -> np.ndarray:
    """Apply rectangular crop to image, masking outside as black."""
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # Calculate rectangle corners
    x1 = max(0, center_x - width // 2)
    y1 = max(0, center_y - height // 2)
    x2 = min(w, center_x + width // 2)
    y2 = min(h, center_y + height // 2)
    
    # Fill rectangle
    mask[y1:y2, x1:x2] = 255
    return apply_mask_to_image(img, mask)

def apply_crop(img: np.ndarray, settings: QCSettings) -> np.ndarray:
    """Apply crop based on settings (circle or rectangle)."""
    if not settings.use_crop:
        return img
    
    if settings.crop_shape == "circle":
        return apply_circular_crop(img, settings.crop_center_x, 
                                   settings.crop_center_y, settings.crop_diameter)
    else:  # rectangle
        return apply_rectangular_crop(img, settings.crop_center_x, settings.crop_center_y,
                                      settings.crop_width, settings.crop_height)

def draw_circle_on_image(img: np.ndarray, center_x: int, center_y: int, 
                        diameter: int, color: Tuple = (255, 0, 0), thickness: int = 3) -> np.ndarray:
    """Draw a circle on image for visualization."""
    img_copy = img.copy()
    radius = diameter // 2
    cv2.circle(img_copy, (center_x, center_y), radius, color, thickness)
    # Draw crosshair at center
    cross_size = 15
    cv2.line(img_copy, (center_x - cross_size, center_y), 
             (center_x + cross_size, center_y), color, thickness)
    cv2.line(img_copy, (center_x, center_y - cross_size), 
             (center_x, center_y + cross_size), color, thickness)
    return img_copy

def draw_rectangle_on_image(img: np.ndarray, center_x: int, center_y: int, 
                           width: int, height: int, color: Tuple = (255, 0, 0), 
                           thickness: int = 3) -> np.ndarray:
    """Draw a rectangle on image for visualization."""
    img_copy = img.copy()
    x1 = center_x - width // 2
    y1 = center_y - height // 2
    x2 = center_x + width // 2
    y2 = center_y + height // 2
    cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, thickness)
    # Draw crosshair at center
    cross_size = 15
    cv2.line(img_copy, (center_x - cross_size, center_y), 
             (center_x + cross_size, center_y), color, thickness)
    cv2.line(img_copy, (center_x, center_y - cross_size), 
             (center_x, center_y + cross_size), color, thickness)
    return img_copy

def overlay_regions(img: np.ndarray, pts: list, radius: int = 12) -> np.ndarray:
    """Draw circles on image at specified points."""
    from PIL import Image, ImageDraw
    pil = Image.fromarray(img.copy())
    drw = ImageDraw.Draw(pil)
    for (y, x) in pts:
        drw.ellipse([(x-radius, y-radius), (x+radius, y+radius)], 
                    outline=(255,0,0), width=3)
    return np.array(pil)

