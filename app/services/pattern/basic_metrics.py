"""Basic pattern analysis metrics."""
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
from skimage.color import rgb2gray

def ssim_percent(ref_rgb: np.ndarray, test_rgb: np.ndarray) -> float:
    """Calculate SSIM percentage."""
    gr1 = rgb2gray(ref_rgb)
    gr2 = rgb2gray(test_rgb)
    return float(ssim(gr1, gr2, data_range=1.0) * 100.0)

def symmetry_score(gray: np.ndarray) -> float:
    """Calculate symmetry score."""
    h, w = gray.shape
    left = gray[:, :w//2]
    right = np.fliplr(gray[:, w - w//2:])
    top = gray[:h//2, :]
    bottom = np.flipud(gray[h - h//2:, :])
    sh = ssim(left, right, data_range=1.0)
    sv = ssim(top, bottom, data_range=1.0)
    return float((sh + sv)/2 * 100)

def repeat_period_estimate(gray: np.ndarray) -> tuple:
    """Estimate repeat period using FFT."""
    f = np.fft.fftshift(np.fft.fft2(gray))
    mag = np.log(np.abs(f) + 1e-8)
    cy, cx = np.array(mag.shape)//2
    window = 10
    mag[cy-window:cy+window, cx-window:cx+window] = 0
    y_idx, x_idx = np.unravel_index(np.argmax(mag), mag.shape)
    fy = abs(y_idx - cy) / gray.shape[0]
    fx = abs(x_idx - cx) / gray.shape[1]
    px = int(round(1/fx)) if fx > 1e-4 else 0
    py = int(round(1/fy)) if fy > 1e-4 else 0
    return px, py

def edge_definition(gray: np.ndarray) -> float:
    """Calculate edge definition score."""
    lap = cv2.Laplacian((gray*255).astype(np.uint8), cv2.CV_64F)
    var = np.var(lap)
    return float(min(100.0, var / 50.0))

