# ===========================
# Textile QC: Colab Edition — Enhanced & Optimized
# ===========================
# - Upload two images when prompted (Reference & Test)
# - Performs Color + Pattern analyses with advanced texture analysis
# - Generates comprehensive A4 PDF reports with rich visualizations
# - Shows ONLY a "Download Report" button (no inline outputs)
#
# RECENT IMPROVEMENTS (v1.1.0):
# - Fixed critical mathematical errors in ΔE94, whiteness formulas
# - Added comprehensive input validation and error handling
# - Refactored duplicate code (reduced from 5478 to optimized lines)
# - Added proper logging system replacing print statements
# - Improved status determination with unified function
# - Added CSV export functionality for data analysis
# - Enhanced documentation with docstrings
# - Optimized image processing with better error recovery
#
# Author: AI Textile QC System
# Version: 1.1.0
# ===========================

# ----------------------------
# Imports
# ----------------------------
import io, os, base64, math, textwrap, tempfile, uuid
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
import cv2
from PIL import Image, ImageDraw
import matplotlib
matplotlib.use("Agg")  # Important: no inline backend
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.color import rgb2gray
from google.colab import files as colab_files
from IPython.display import display, HTML, clear_output

# ReportLab for PDF
!pip -q install reportlab >/dev/null
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfgen import canvas
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Image as RLImage, Table, TableStyle,
                                Spacer, PageBreak, Flowable, KeepTogether)
from reportlab.lib.enums import TA_LEFT, TA_RIGHT, TA_CENTER

# ipywidgets for interactive UI
!pip -q install ipywidgets >/dev/null
import ipywidgets as widgets
from ipywidgets import Layout, HBox, VBox, Button, IntText, FloatText, Text, HTML as HTMLWidget, Output

# Advanced analysis libraries
!pip -q install PyWavelets >/dev/null
import pywt
from scipy import signal, ndimage
from scipy.stats import chi2
from scipy.spatial.distance import euclidean
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops, hog
from skimage.filters import gabor_kernel, threshold_otsu
from skimage.morphology import disk, white_tophat, black_tophat, opening, closing
from skimage.measure import label, regionprops
from skimage.util import img_as_ubyte
import warnings
warnings.filterwarnings('ignore')
import logging
from functools import lru_cache

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Cache for expensive computations
_computation_cache = {}

# ----------------------------
# Config / Theme
# ----------------------------
SOFTWARE_VERSION = "1.1.0"
COMPANY_NAME = "Textile Engineering Solutions"
COMPANY_SUBTITLE = "Professional Color Analysis Solutions"
REPORT_TITLE = "Color Analysis Report"
PAGE_SIZE = A4
MARGIN_L = 50  # Margins adjusted for 3mm frame spacing
MARGIN_R = 50
MARGIN_T = 50
MARGIN_B = 50  # Increased bottom margin for safe distance from footer
DPI = 300
TIMEZONE_OFFSET_HOURS = 3  # Turkey timezone offset (UTC+3)
FRAME_MARGIN = 9  # 3mm frame margin (approximately 9 points)

# Colors
BLUE1 = colors.HexColor("#2980B9")
BLUE2 = colors.HexColor("#3498DB")
GREEN = colors.HexColor("#27AE60")
RED   = colors.HexColor("#E74C3C")
ORANGE= colors.HexColor("#F39C12")
NEUTRAL_DARK = colors.HexColor("#2C3E50")
NEUTRAL = colors.HexColor("#7F8C8D")
NEUTRAL_L = colors.HexColor("#BDC3C7")

STATUS_COLORS = {"PASS": GREEN, "FAIL": RED, "CONDITIONAL": ORANGE}

# Logo file (primary logo file to use)
PRIMARY_LOGO = "llogo_square_with_name_1024x1024.png"
FALLBACK_LOGOS = ["logo_square_with_name_1024x1024.png", "logo_square_no_name_1024x1024.png"]

# ----------------------------
# Settings Data Class
# ----------------------------
@dataclass
class QCSettings:
    """Quality Control Settings for textile analysis"""
    # Operator info
    operator_name: str = "Operator"

    # Color thresholds
    delta_e_threshold: float = 2.0
    delta_e_conditional: float = 3.5
    lab_l_threshold: float = 1.0
    lab_ab_threshold: float = 1.0
    lab_overall_threshold: float = 2.0

    # Pattern thresholds
    ssim_pass_threshold: float = 0.95
    ssim_conditional_threshold: float = 0.90

    # Scoring parameters
    color_score_multiplier: float = 20.0
    uniformity_std_multiplier: float = 10.0

    # Quality decision thresholds
    color_score_threshold: float = 70.0
    pattern_score_threshold: float = 90.0
    overall_score_threshold: float = 70.0

    # Region selection
    use_crop: bool = False
    crop_shape: str = "circle"  # "circle" or "rectangle"
    crop_center_x: int = 0
    crop_center_y: int = 0
    crop_diameter: int = 500  # For circle
    crop_width: int = 500  # For rectangle
    crop_height: int = 500  # For rectangle

    # Number of sample points
    num_sample_points: int = 5

    # ===== ADVANCED TEXTURE/PATTERN PARAMETERS =====
    # FFT parameters
    fft_enable_notch: bool = False
    fft_num_peaks: int = 5

    # Gabor parameters
    gabor_frequencies: list = field(default_factory=lambda: [0.1, 0.2, 0.3])
    gabor_frequencies_str: str = "0.1, 0.2, 0.3"  # UI input
    gabor_num_orientations: int = 8

    # GLCM parameters
    glcm_distances: list = field(default_factory=lambda: [1, 3, 5])
    glcm_distances_str: str = "1, 3, 5"  # UI input
    glcm_angles: list = field(default_factory=lambda: [0, 45, 90, 135])
    glcm_angles_str: str = "0, 45, 90, 135"  # UI input

    # LBP parameters
    lbp_points: int = 24
    lbp_radius: int = 3

    # Wavelet parameters
    wavelet_type: str = 'db4'
    wavelet_levels: int = 3

    # Defect detection parameters
    defect_min_area: int = 50
    saliency_strength: float = 1.0
    morph_kernel_size: int = 5

    # ===== COLOR/SPECTROPHOTOMETER PARAMETERS =====
    # Observer angle
    observer_angle: str = "2"  # "2" or "10" degrees

    # Geometry mode
    geometry_mode: str = "d/8 SCI"  # "d/8 SCI", "d/8 SCE", "45/0"

    # Color difference methods
    use_delta_e_cmc: bool = True
    cmc_l_c_ratio: str = "2:1"  # "2:1" or "1:1"

    # Whiteness/Yellowness thresholds
    whiteness_min: float = 40.0
    yellowness_max: float = 10.0

    # Metamerism illuminants
    metamerism_illuminants: list = field(default_factory=lambda: ["D65", "D50", "TL84", "A", "F2", "CWF"])

    # Spectral data
    spectral_ref_path: str = ""
    spectral_sample_path: str = ""
    spectral_enable: bool = False
    spectral_ref_wavelengths: np.ndarray = field(default_factory=lambda: np.array([]))
    spectral_ref_reflectance: np.ndarray = field(default_factory=lambda: np.array([]))
    spectral_sample_wavelengths: np.ndarray = field(default_factory=lambda: np.array([]))
    spectral_sample_reflectance: np.ndarray = field(default_factory=lambda: np.array([]))

    # UV control (note only)
    uv_control_note: str = "UV control not available for RGB images"

    # ===== PATTERN REPETITION PARAMETERS =====
    pattern_min_area: int = 100  # Minimum pattern area in pixels
    pattern_max_area: int = 5000  # Maximum pattern area in pixels
    pattern_similarity_threshold: float = 0.85  # For template matching
    blob_min_circularity: float = 0.5
    blob_min_convexity: float = 0.8
    keypoint_detector: str = "ORB"  # "SIFT", "ORB", "AKAZE"
    grid_cell_size: int = 50  # For spatial distribution analysis
    pattern_count_tolerance: int = 5  # Acceptable difference in count
    pattern_match_threshold: float = 0.7  # For keypoint matching

    # ===== REPORT SECTIONS CONTROL =====
    # Main sections
    enable_analysis_settings: bool = False  # Disabled by default, can be enabled by user
    enable_color_unit: bool = True
    enable_pattern_unit: bool = True
    enable_pattern_repetition: bool = True  # New pattern repetition analysis
    enable_spectrophotometer: bool = True

    # Color Unit sub-sections
    enable_color_input_images: bool = True
    enable_color_measurements: bool = True
    enable_color_difference: bool = True
    enable_color_statistical: bool = True
    enable_color_spectral_proxy: bool = True
    enable_color_visual_diff: bool = True
    enable_color_lab_detailed: bool = True
    enable_color_lab_viz: bool = True
    enable_color_quality_assessment: bool = True
    enable_color_scoring: bool = True
    enable_color_recommendations: bool = True

    # Pattern Unit sub-sections
    enable_pattern_ssim: bool = True
    enable_pattern_symmetry: bool = True
    enable_pattern_edge: bool = True
    enable_pattern_repeat: bool = True
    enable_pattern_advanced: bool = True

    # Pattern Repetition Unit sub-sections
    enable_pattern_rep_summary: bool = True
    enable_pattern_rep_count: bool = True
    enable_pattern_rep_blob: bool = True
    enable_pattern_rep_keypoint: bool = True
    enable_pattern_rep_autocorr: bool = True
    enable_pattern_rep_spatial: bool = True
    enable_pattern_rep_integrity: bool = True
    enable_pattern_rep_catalog: bool = True

    # Spectrophotometer sub-sections
    enable_spectro_config: bool = True
    enable_spectro_color_diff_methods: bool = True
    enable_spectro_whiteness: bool = True
    enable_spectro_metamerism: bool = True
    enable_spectro_spectral_data: bool = True
    enable_spectro_calibration: bool = True

def get_local_time():
    """Get current time with Turkey timezone offset"""
    return datetime.now() + timedelta(hours=TIMEZONE_OFFSET_HOURS)

# ----------------------------
# 0) Helper: Colab uploads
# ----------------------------
def upload_two_images():
    print("👉 Please upload the REFERENCE image first, then the TEST image.")
    uploaded = colab_files.upload()
    if len(uploaded) == 0:
        raise RuntimeError("No files uploaded.")
    names = list(uploaded.keys())
    if len(names) == 1:
        print("Now upload the TEST image.")
        uploaded2 = colab_files.upload()
        if len(uploaded2) == 0:
            raise RuntimeError("Only one image uploaded. Need two.")
        names += list(uploaded2.keys())
    ref_path, test_path = names[0], names[1]
    return ref_path, test_path

# ----------------------------
# 1) IO & conversions
# ----------------------------
def validate_image_file(path):
    """Validate that the file exists and is a valid image format"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image file not found: {path}")

    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    ext = os.path.splitext(path)[1].lower()
    if ext not in valid_extensions:
        raise ValueError(f"Unsupported image format: {ext}. Supported formats: {', '.join(valid_extensions)}")

    return True

def validate_image_dimensions(img, min_size=100, max_size=10000):
    """Validate image dimensions are within acceptable range"""
    h, w = img.shape[:2]
    if h < min_size or w < min_size:
        raise ValueError(f"Image too small: {w}x{h}. Minimum size: {min_size}x{min_size}")
    if h > max_size or w > max_size:
        raise ValueError(f"Image too large: {w}x{h}. Maximum size: {max_size}x{max_size}")
    return True

def read_rgb(path):
    """Read RGB image with validation"""
    try:
        validate_image_file(path)
        img = Image.open(path).convert("RGB")
        arr = np.array(img)
        validate_image_dimensions(arr)
        return arr
    except Exception as e:
        raise RuntimeError(f"Failed to read image {path}: {str(e)}")

def to_same_size(a, b):
    h = min(a.shape[0], b.shape[0])
    w = min(a.shape[1], b.shape[1])
    a2 = cv2.resize(a, (w, h), interpolation=cv2.INTER_AREA)
    b2 = cv2.resize(b, (w, h), interpolation=cv2.INTER_AREA)
    return a2, b2

# sRGB -> XYZ (D65)
def srgb_to_xyz(rgb):
    x = rgb.astype(float) / 255.0
    mask = x > 0.04045
    x[mask] = ((x[mask] + 0.055) / 1.055) ** 2.4
    x[~mask] = x[~mask] / 12.92
    x *= 100.0
    M = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]])
    return x @ M.T

# Bradford CAT for illuminant adaptation
WHITE_POINTS = {
    "D65": np.array([95.047, 100.000, 108.883]),
    "D50": np.array([96.422, 100.000, 82.521]),
    "TL84": np.array([101.385, 100.000, 65.231]),  # F11/TL84
    "A":   np.array([109.850, 100.000, 35.585]),
    "F2":  np.array([99.187, 100.000, 67.395]),  # Cool White Fluorescent
    "CWF": np.array([103.280, 100.000, 69.026]),  # Cool White Fluorescent
    "F7":  np.array([95.044, 100.000, 108.755]),  # Daylight Fluorescent
    "F11": np.array([100.966, 100.000, 64.370]),  # TL84 equivalent
}
M_BRADFORD = np.array([[ 0.8951,  0.2664, -0.1614],
                       [-0.7502,  1.7135,  0.0367],
                       [ 0.0389, -0.0685,  1.0296]])
M_BRADFORD_INV = np.linalg.inv(M_BRADFORD)

# ----------------------------
# CIE Standard Observer & Illuminant SPDs (for spectral data)
# ----------------------------
# Simplified CIE 1931 2° observer (380-780nm, 5nm step)
CIE_2DEG_WAVELENGTHS = np.arange(380, 781, 5)

# CIE 1931 2° color matching functions (simplified, key wavelengths)
CIE_2DEG_CMF = {
    'x_bar': np.array([0.001368, 0.004243, 0.014310, 0.043510, 0.134380, 0.283900, 0.348280, 0.336200, 0.290800, 0.195360,
                       0.095640, 0.032010, 0.004900, 0.009300, 0.063270, 0.165500, 0.290400, 0.433450, 0.594500, 0.762100,
                       0.916300, 1.026300, 1.062200, 1.002600, 0.854450, 0.642400, 0.447900, 0.283500, 0.164900, 0.087400,
                       0.046770, 0.022700, 0.011359, 0.005790, 0.002899, 0.001440, 0.000690, 0.000332, 0.000166, 0.000083,
                       0.000042, 0.000021, 0.000010, 0.000005, 0.000003, 0.000001, 0.000001, 0.000000, 0.000000, 0.000000,
                       0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                       0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                       0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000]),
    'y_bar': np.array([0.000039, 0.000120, 0.000396, 0.001210, 0.004000, 0.011600, 0.023000, 0.038000, 0.060000, 0.090980,
                       0.139020, 0.208020, 0.323000, 0.503000, 0.710000, 0.862000, 0.954000, 0.994950, 0.995000, 0.952000,
                       0.870000, 0.757000, 0.631000, 0.503000, 0.381000, 0.265000, 0.175000, 0.107000, 0.061000, 0.032000,
                       0.017000, 0.008210, 0.004102, 0.002091, 0.001047, 0.000520, 0.000249, 0.000120, 0.000060, 0.000030,
                       0.000015, 0.000007, 0.000004, 0.000002, 0.000001, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                       0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                       0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                       0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000]),
    'z_bar': np.array([0.006450, 0.020050, 0.067850, 0.207400, 0.645600, 1.385600, 1.747060, 1.772110, 1.669200, 1.287640,
                       0.812950, 0.465180, 0.272000, 0.158200, 0.078250, 0.042160, 0.020300, 0.008750, 0.003900, 0.002100,
                       0.001650, 0.001100, 0.000800, 0.000340, 0.000190, 0.000050, 0.000020, 0.000000, 0.000000, 0.000000,
                       0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                       0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                       0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                       0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                       0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000])
}

# Simplified D65 illuminant SPD (relative, 380-780nm, 5nm step)
D65_SPD = np.array([49.98, 52.31, 54.65, 68.70, 82.75, 87.12, 91.49, 92.46, 93.43, 90.06,
                    86.68, 95.77, 104.86, 110.94, 117.01, 117.41, 117.81, 116.34, 114.86, 115.39,
                    115.92, 112.37, 108.81, 109.08, 109.35, 108.58, 107.80, 106.30, 104.79, 106.24,
                    107.69, 106.05, 104.41, 104.23, 104.05, 102.02, 100.00, 98.17, 96.33, 96.06,
                    95.79, 92.24, 88.69, 89.35, 90.01, 89.80, 89.60, 88.65, 87.70, 85.49,
                    83.29, 83.49, 83.70, 81.86, 80.03, 80.12, 80.21, 81.25, 82.28, 80.28,
                    78.28, 74.00, 69.72, 70.67, 71.61, 72.98, 74.35, 67.98, 61.60, 65.74,
                    69.89, 72.49, 75.09, 69.34, 63.59, 55.01, 46.42, 56.61, 66.81, 65.09, 63.38])

def adapt_white_xyz(xyz, src_wp, dst_wp):
    src_lms = (M_BRADFORD @ xyz.reshape(-1,3).T).T
    src_wp_lms = M_BRADFORD @ src_wp
    dst_wp_lms = M_BRADFORD @ dst_wp
    D = (dst_wp_lms / src_wp_lms)
    dst_lms = (src_lms * D)
    out = (M_BRADFORD_INV @ dst_lms.T).T
    return out.reshape(xyz.shape)

def xyz_to_lab(xyz, wp):
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

def rgb_to_cmyk(rgb):
    r, g, b = rgb[...,0]/255.0, rgb[...,1]/255.0, rgb[...,2]/255.0
    k = 1 - np.max(np.stack([r,g,b], axis=-1), axis=-1)
    denom = 1 - k + 1e-8
    c = (1 - r - k) / denom
    m = (1 - g - k) / denom
    y = (1 - b - k) / denom
    return np.stack([c,m,y,k], axis=-1)

# ----------------------------
# 2) ΔE formulas
# ----------------------------
def deltaE76(lab1, lab2):
    d = lab1 - lab2
    return np.sqrt(np.sum(d**2, axis=-1))

def deltaE94(lab1, lab2, kL=1, kC=1, kH=1, K1=0.045, K2=0.015):
    L1,a1,b1 = lab1[...,0], lab1[...,1], lab1[...,2]
    L2,a2,b2 = lab2[...,0], lab2[...,1], lab2[...,2]
    dL = L1 - L2
    C1 = np.sqrt(a1**2 + b1**2)
    C2 = np.sqrt(a2**2 + b2**2)
    dC = C1 - C2
    da = a1 - a2
    db = b1 - b2
    dH_sq = da**2 + db**2 - dC**2
    dH_sq = np.maximum(dH_sq, 0)  # Prevent negative values due to numerical errors
    SL = 1
    SC = 1 + K1*C1
    SH = 1 + K2*C1
    dH = np.sqrt(dH_sq)
    return np.sqrt((dL/(kL*SL))**2 + (dC/(kC*SC))**2 + (dH/(kH*SH))**2)

def deltaE2000(lab1, lab2, kL=1, kC=1, kH=1):
    L1,a1,b1 = lab1[...,0], lab1[...,1], lab1[...,2]
    L2,a2,b2 = lab2[...,0], lab2[...,1], lab2[...,2]
    C1 = np.sqrt(a1**2 + b1**2)
    C2 = np.sqrt(a2**2 + b2**2)
    Cm = (C1 + C2) / 2
    G = 0.5 * (1 - np.sqrt((Cm**7) / (Cm**7 + 25**7)))
    a1p = (1 + G) * a1
    a2p = (1 + G) * a2
    C1p = np.sqrt(a1p**2 + b1**2)
    C2p = np.sqrt(a2p**2 + b2**2)
    h1p = (np.degrees(np.arctan2(b1, a1p)) + 360) % 360
    h2p = (np.degrees(np.arctan2(b2, a2p)) + 360) % 360
    dLp = L2 - L1
    dCp = C2p - C1p
    dhp = h2p - h1p
    dhp = np.where(dhp > 180, dhp - 360, dhp)
    dhp = np.where(dhp < -180, dhp + 360, dhp)
    dHp = 2*np.sqrt(C1p*C2p)*np.sin(np.radians(dhp)/2)
    Lpm = (L1 + L2)/2
    Cpm = (C1p + C2p)/2
    hp_sum = h1p + h2p
    hpm = np.where((np.abs(h1p - h2p) > 180), (hp_sum + 360)/2, hp_sum/2)
    T = 1 - 0.17*np.cos(np.radians(hpm - 30)) + 0.24*np.cos(np.radians(2*hpm)) + \
        0.32*np.cos(np.radians(3*hpm + 6)) - 0.20*np.cos(np.radians(4*hpm - 63))
    dRo = 30*np.exp(-((hpm - 275)/25)**2)
    Rc = 2*np.sqrt((Cpm**7) / (Cpm**7 + 25**7))
    Sl = 1 + (0.015*((Lpm - 50)**2)) / np.sqrt(20 + (Lpm - 50)**2)
    Sc = 1 + 0.045*Cpm
    Sh = 1 + 0.015*Cpm*T
    Rt = -np.sin(np.radians(2*dRo)) * Rc
    return np.sqrt((dLp/(kL*Sl))**2 + (dCp/(kC*Sc))**2 + (dHp/(kH*Sh))**2 + Rt*(dCp/(kC*Sc))*(dHp/(kH*Sh)))

def deltaE_CMC(lab1, lab2, l=2, c=1):
    """CMC l:c color difference (typically l:c = 2:1 or 1:1)"""
    L1, a1, b1 = lab1[...,0], lab1[...,1], lab1[...,2]
    L2, a2, b2 = lab2[...,0], lab2[...,1], lab2[...,2]

    dL = L1 - L2
    C1 = np.sqrt(a1**2 + b1**2)
    C2 = np.sqrt(a2**2 + b2**2)
    dC = C1 - C2
    da = a1 - a2
    db = b1 - b2
    dH_sq = da**2 + db**2 - dC**2
    dH_sq = np.maximum(dH_sq, 0)  # Avoid negative due to numerical errors

    H1 = np.degrees(np.arctan2(b1, a1))
    H1 = np.where(H1 < 0, H1 + 360, H1)

    # Weighting functions
    F = np.sqrt(C1**4 / (C1**4 + 1900))
    T = np.where((H1 >= 164) & (H1 <= 345),
                 0.56 + np.abs(0.2 * np.cos(np.radians(H1 + 168))),
                 0.36 + np.abs(0.4 * np.cos(np.radians(H1 + 35))))

    SL = np.where(L1 < 16, 0.511, (0.040975 * L1) / (1 + 0.01765 * L1))
    SC = ((0.0638 * C1) / (1 + 0.0131 * C1)) + 0.638
    SH = SC * (F * T + 1 - F)

    return np.sqrt((dL/(l*SL))**2 + (dC/(c*SC))**2 + (dH_sq/(SH**2)))

def cie_whiteness_tint(xyz, illuminant='D65'):
    """CIE Whiteness and Tint (ISO 11475) for illuminant D65 with 10° observer"""
    # For D65/10°, the formula uses normalized chromaticity coordinates
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

def astm_e313_yellowness(xyz):
    """ASTM E313 Yellowness Index"""
    X, Y, Z = xyz[...,0], xyz[...,1], xyz[...,2]

    # Coefficients for D65/10° (newer standard)
    C_x = 1.3013
    C_z = 1.1498

    YI = 100 * (C_x * X - C_z * Z) / np.maximum(Y, 1e-8)

    return YI

# ----------------------------
# 2b) SPECTRAL DATA PROCESSING
# ----------------------------
def parse_spectral_csv(csv_path):
    """
    Parse spectral CSV file (wavelength, reflectance).

    Args:
        csv_path: Path to CSV file with spectral data

    Returns:
        tuple: (wavelengths, reflectance) arrays, or (None, None) on error
    """
    try:
        if not os.path.exists(csv_path):
            logger.error(f"Spectral CSV file not found: {csv_path}")
            return None, None

        df = pd.read_csv(csv_path)

        if df.empty:
            logger.error(f"Spectral CSV file is empty: {csv_path}")
            return None, None

        # Try common column name variations
        wl_cols = [c for c in df.columns if 'wave' in c.lower() or 'nm' in c.lower() or 'λ' in c.lower()]
        ref_cols = [c for c in df.columns if 'ref' in c.lower() or 'r(' in c.lower() or '%' in c.lower()]

        if not wl_cols or not ref_cols:
            # Assume first two columns
            if len(df.columns) < 2:
                logger.error(f"Spectral CSV must have at least 2 columns: {csv_path}")
                return None, None
            wavelengths = df.iloc[:, 0].values
            reflectance = df.iloc[:, 1].values
        else:
            wavelengths = df[wl_cols[0]].values
            reflectance = df[ref_cols[0]].values

        # Validate data ranges
        if np.any(wavelengths < 300) or np.any(wavelengths > 800):
            logger.warning(f"Wavelengths outside typical range (300-800nm) in {csv_path}")

        if np.any(reflectance < 0) or np.any(reflectance > 100):
            logger.warning(f"Reflectance values outside 0-100% range in {csv_path}")
            reflectance = np.clip(reflectance, 0, 100)

        # Filter to 380-700nm range
        mask = (wavelengths >= 380) & (wavelengths <= 700)
        filtered_wl = wavelengths[mask]
        filtered_ref = reflectance[mask]

        if len(filtered_wl) == 0:
            logger.error(f"No data in valid wavelength range (380-700nm) in {csv_path}")
            return None, None

        logger.info(f"Parsed spectral CSV: {len(filtered_wl)} data points")
        return filtered_wl, filtered_ref

    except Exception as e:
        logger.error(f"Error parsing spectral CSV {csv_path}: {str(e)}")
        return None, None

def spectral_to_xyz(wavelengths, reflectance, illuminant='D65', observer='2'):
    """Compute XYZ tristimulus values from spectral reflectance"""
    # Interpolate spectral data to match CIE wavelengths (380-780nm, 5nm step)
    cie_wl = CIE_2DEG_WAVELENGTHS

    # Interpolate reflectance to CIE wavelengths
    reflectance_interp = np.interp(cie_wl, wavelengths, reflectance)

    # Get CMF
    x_bar = CIE_2DEG_CMF['x_bar']
    y_bar = CIE_2DEG_CMF['y_bar']
    z_bar = CIE_2DEG_CMF['z_bar']

    # Get illuminant SPD (using D65 as default, others can be added)
    spd = D65_SPD

    # Compute tristimulus values: X = k * Σ R(λ) * x̄(λ) * S(λ) * Δλ
    delta_lambda = 5  # 5nm step

    X = np.sum(reflectance_interp * x_bar * spd) * delta_lambda
    Y = np.sum(reflectance_interp * y_bar * spd) * delta_lambda
    Z = np.sum(reflectance_interp * z_bar * spd) * delta_lambda

    # Normalize to Y=100 for perfect white
    k = 100.0 / np.sum(y_bar * spd * delta_lambda)

    return np.array([X * k, Y * k, Z * k])

def find_spectral_peaks_valleys(wavelengths, reflectance, n_peaks=3):
    """Find peaks and valleys in spectral reflectance curve"""
    from scipy.signal import find_peaks

    # Find peaks
    peaks_idx, _ = find_peaks(reflectance, prominence=2)
    if len(peaks_idx) > 0:
        # Sort by reflectance value
        peak_heights = reflectance[peaks_idx]
        sorted_peaks = peaks_idx[np.argsort(peak_heights)[::-1]][:n_peaks]
    else:
        sorted_peaks = []

    # Find valleys (invert signal)
    valleys_idx, _ = find_peaks(-reflectance, prominence=2)
    if len(valleys_idx) > 0:
        valley_depths = reflectance[valleys_idx]
        sorted_valleys = valleys_idx[np.argsort(valley_depths)][:n_peaks]
    else:
        sorted_valleys = []

    results = []
    for idx in sorted_peaks:
        results.append({
            'type': 'Peak',
            'wavelength': wavelengths[idx],
            'reflectance': reflectance[idx]
        })

    for idx in sorted_valleys:
        results.append({
            'type': 'Valley',
            'wavelength': wavelengths[idx],
            'reflectance': reflectance[idx]
        })

    return results

# ----------------------------
# 3) Pattern helpers
# ----------------------------
def ssim_percent(ref_rgb, test_rgb):
    gr1 = rgb2gray(ref_rgb)
    gr2 = rgb2gray(test_rgb)
    return float(ssim(gr1, gr2, data_range=1.0) * 100.0)

def symmetry_score(gray):
    h, w = gray.shape
    left = gray[:, :w//2]
    right = np.fliplr(gray[:, w - w//2:])
    top = gray[:h//2, :]
    bottom = np.flipud(gray[h - h//2:, :])
    sh = ssim(left, right, data_range=1.0)
    sv = ssim(top, bottom, data_range=1.0)
    return float((sh + sv)/2 * 100)

def repeat_period_estimate(gray):
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

def edge_definition(gray):
    lap = cv2.Laplacian((gray*255).astype(np.uint8), cv2.CV_64F)
    var = np.var(lap)
    return float(min(100.0, var / 50.0))

# ----------------------------
# 3b) ADVANCED TEXTURE ANALYSIS
# ----------------------------

# ========== FOURIER DOMAIN ==========
def analyze_fft(gray, num_peaks=5, enable_notch=False):
    """2D FFT analysis with peak detection"""
    h, w = gray.shape
    f = np.fft.fft2(gray)
    f_shift = np.fft.fftshift(f)
    magnitude = np.abs(f_shift)
    power_spectrum = np.log(magnitude + 1)

    # Find peaks (excluding DC component)
    cy, cx = h // 2, w // 2
    magnitude_copy = magnitude.copy()
    magnitude_copy[cy-5:cy+5, cx-5:cx+5] = 0  # Mask DC

    peaks = []
    for _ in range(num_peaks):
        y, x = np.unravel_index(np.argmax(magnitude_copy), magnitude_copy.shape)
        if magnitude_copy[y, x] < 1e-5:
            break
        r = np.sqrt((y - cy)**2 + (x - cx)**2)
        angle = np.degrees(np.arctan2(y - cy, x - cx))
        peaks.append({'radius': r, 'angle': angle, 'magnitude': magnitude_copy[y, x]})
        # Mask region around peak
        magnitude_copy[max(0,y-3):min(h,y+3), max(0,x-3):min(w,x+3)] = 0

    # Fundamental period and orientation
    if peaks:
        fund_r = peaks[0]['radius']
        fund_period = min(h, w) / max(fund_r, 1e-5)
        fund_orientation = peaks[0]['angle']
    else:
        fund_period = 0
        fund_orientation = 0

    # Anisotropy ratio
    radial_profile = []
    for r in range(1, min(cx, cy)):
        mask = ((np.arange(h)[:, None] - cy)**2 + (np.arange(w) - cx)**2 < (r+1)**2) & \
               ((np.arange(h)[:, None] - cy)**2 + (np.arange(w) - cx)**2 >= r**2)
        radial_profile.append(np.mean(magnitude[mask]) if mask.any() else 0)

    anisotropy = np.std(radial_profile) / (np.mean(radial_profile) + 1e-8) if radial_profile else 0

    # Optional notch filter
    if enable_notch and peaks:
        for peak in peaks[:3]:
            y = int(cy + peak['radius'] * np.sin(np.radians(peak['angle'])))
            x = int(cx + peak['radius'] * np.cos(np.radians(peak['angle'])))
            cv2.circle(f_shift, (x, y), 10, 0, -1)
        f_ishift = np.fft.ifftshift(f_shift)
        filtered = np.fft.ifft2(f_ishift)
        residual = np.abs(filtered).real
    else:
        residual = None

    return {
        'power_spectrum': power_spectrum,
        'peaks': peaks,
        'fundamental_period': fund_period,
        'fundamental_orientation': fund_orientation,
        'anisotropy': anisotropy,
        'residual': residual
    }

# ========== GABOR FILTER BANK ==========
def analyze_gabor(gray, frequencies=[0.1, 0.2, 0.3], num_orientations=8):
    """Multi-scale, multi-orientation Gabor analysis"""
    results = []
    energy_maps = []

    for freq in frequencies:
        for i in range(num_orientations):
            theta = i * np.pi / num_orientations
            kernel = gabor_kernel(freq, theta=theta, sigma_x=3, sigma_y=3)
            filtered_real = ndimage.convolve(gray, kernel.real, mode='wrap')
            filtered_imag = ndimage.convolve(gray, kernel.imag, mode='wrap')
            energy = np.sqrt(filtered_real**2 + filtered_imag**2)
            energy_maps.append(energy)

            results.append({
                'frequency': freq,
                'orientation_deg': np.degrees(theta),
                'mean': float(np.mean(energy)),
                'variance': float(np.var(energy)),
                'max': float(np.max(energy))
            })

    # Dominant orientation
    mean_energies = [r['mean'] for r in results]
    dom_idx = np.argmax(mean_energies)
    dominant_orientation = results[dom_idx]['orientation_deg']

    # Coherency (ratio of max to mean energy)
    coherency = np.max(mean_energies) / (np.mean(mean_energies) + 1e-8)

    return {
        'results': results,
        'energy_maps': energy_maps,
        'dominant_orientation': dominant_orientation,
        'coherency': coherency
    }

# ========== GLCM / HARALICK ==========
def analyze_glcm(gray, distances=[1, 3, 5], angles=[0, 45, 90, 135]):
    """GLCM texture features"""
    # Convert to 8-bit
    gray_8bit = img_as_ubyte(gray)

    # Convert angles to radians
    angles_rad = [np.radians(a) for a in angles]

    # Compute GLCM
    glcm = graycomatrix(gray_8bit, distances=distances, angles=angles_rad,
                         levels=256, symmetric=True, normed=True)

    # Extract properties
    props = {}
    for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']:
        prop_name = 'energy' if prop == 'energy' else prop
        if prop_name != 'ASM':
            props[prop] = graycoprops(glcm, prop_name).mean()
        else:
            props['ASM'] = graycoprops(glcm, 'energy').mean()  # ASM = energy

    # Calculate entropy manually
    glcm_mean = glcm.mean(axis=(2, 3))
    entropy = -np.sum(glcm_mean * np.log(glcm_mean + 1e-10))
    props['entropy'] = entropy

    return props

def compute_glcm_zscores(glcm_ref, glcm_test):
    """Compute z-scores for GLCM features"""
    # Typical standard deviations for GLCM features (empirical values)
    typical_stds = {
        'contrast': 50.0,
        'dissimilarity': 5.0,
        'homogeneity': 0.1,
        'energy': 0.05,
        'correlation': 0.1,
        'ASM': 0.05,
        'entropy': 0.5
    }

    zscores = {}
    for feat in glcm_ref.keys():
        diff = glcm_test[feat] - glcm_ref[feat]
        std = typical_stds.get(feat, 1.0)
        zscores[feat] = diff / std

    return zscores

# ========== LBP ==========
def analyze_lbp(gray, P=24, R=3):
    """Local Binary Patterns"""
    lbp = local_binary_pattern(gray, P, R, method='uniform')

    # Histogram
    n_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)

    return {'lbp_map': lbp, 'histogram': hist, 'n_bins': n_bins}

def lbp_chi2_distance(hist1, hist2):
    """Chi-squared distance between LBP histograms"""
    return 0.5 * np.sum((hist1 - hist2)**2 / (hist1 + hist2 + 1e-10))

def lbp_bhattacharyya_distance(hist1, hist2):
    """Bhattacharyya distance"""
    bc = np.sum(np.sqrt(hist1 * hist2))
    return -np.log(bc + 1e-10)

# ========== WAVELET ==========
def analyze_wavelet(gray, wavelet='db4', levels=3):
    """Wavelet multiresolution analysis"""
    coeffs = pywt.wavedec2(gray, wavelet, level=levels)

    # Calculate energies for each level
    energies = []
    for i, (cH, cV, cD) in enumerate(coeffs[1:], start=1):
        energy_LL = np.sum(coeffs[0]**2) if i == 1 else 0
        energy_LH = np.sum(cH**2)
        energy_HL = np.sum(cV**2)
        energy_HH = np.sum(cD**2)

        energies.append({
            'level': i,
            'LL': energy_LL,
            'LH': energy_LH,
            'HL': energy_HL,
            'HH': energy_HH,
            'total': energy_LL + energy_LH + energy_HL + energy_HH
        })

    return {'coeffs': coeffs, 'energies': energies}

# ========== EDGE / STRUCTURE ==========
def analyze_structure_tensor(gray):
    """Structure tensor for coherency and line orientation"""
    # Gradients
    Iy, Ix = np.gradient(gray)

    # Structure tensor components
    Ixx = ndimage.gaussian_filter(Ix * Ix, sigma=1.5)
    Iyy = ndimage.gaussian_filter(Iy * Iy, sigma=1.5)
    Ixy = ndimage.gaussian_filter(Ix * Iy, sigma=1.5)

    # Eigenvalues
    trace = Ixx + Iyy
    det = Ixx * Iyy - Ixy**2
    lambda1 = trace / 2 + np.sqrt(np.maximum((trace/2)**2 - det, 0))
    lambda2 = trace / 2 - np.sqrt(np.maximum((trace/2)**2 - det, 0))

    # Coherency
    coherency = (lambda1 - lambda2) / (lambda1 + lambda2 + 1e-10)

    # Orientation (in radians)
    orientation = 0.5 * np.arctan2(2 * Ixy, Ixx - Iyy)

    # Create orientation histogram (convert to degrees)
    orientation_deg = np.degrees(orientation).flatten()
    # Filter by coherency (only include strong edges)
    strong_edges_mask = coherency.flatten() > 0.3
    filtered_orientations = orientation_deg[strong_edges_mask]

    return {
        'coherency': coherency,
        'orientation': orientation,
        'mean_coherency': float(np.mean(coherency)),
        'orientation_degrees': filtered_orientations
    }

def compute_hog_density(gray):
    """Compute HOG (Histogram of Oriented Gradients) edge density"""
    try:
        # Compute HOG features
        fd, hog_image = hog(gray, orientations=9, pixels_per_cell=(8, 8),
                           cells_per_block=(2, 2), visualize=True)

        # Edge density metric: mean magnitude of HOG features
        edge_density = float(np.mean(np.abs(fd)))

        return {
            'edge_density': edge_density,
            'hog_features': fd,
            'hog_image': hog_image
        }
    except Exception as e:
        print(f"⚠️ HOG computation failed: {e}")
        return {
            'edge_density': 0.0,
            'hog_features': None,
            'hog_image': None
        }

# ========== DEFECT SALIENCY & MORPHOLOGY ==========
def analyze_defects(gray, min_area=50, morph_kernel_size=5, saliency_strength=1.0):
    """
    Defect detection using morphology and saliency.

    Args:
        gray: Grayscale image (0-1 range)
        min_area: Minimum defect area in pixels
        morph_kernel_size: Size of morphological kernel
        saliency_strength: Saliency multiplier

    Returns:
        dict: Dictionary with defect analysis results
    """
    try:
        gray_8bit = img_as_ubyte(gray)
    except Exception as e:
        logger.error(f"Failed to convert image for defect detection: {str(e)}")
        return {
            'tophat': np.zeros_like(gray),
            'bottomhat': np.zeros_like(gray),
            'saliency_map': np.zeros_like(gray),
            'binary_map': np.zeros_like(gray, dtype=bool),
            'defects': [],
            'defect_count': 0
        }

    # Top-hat and bottom-hat
    selem = disk(morph_kernel_size)
    tophat = white_tophat(gray_8bit, selem)
    bottomhat = black_tophat(gray_8bit, selem)

    # Spectral residual saliency
    f = np.fft.fft2(gray)
    magnitude = np.abs(f)
    phase = np.angle(f)
    log_magnitude = np.log(magnitude + 1)
    spectral_residual = log_magnitude - ndimage.gaussian_filter(log_magnitude, sigma=3)
    saliency_map = np.abs(np.fft.ifft2(np.exp(spectral_residual + 1j * phase)))**2
    saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min() + 1e-10)
    saliency_map = saliency_map * saliency_strength

    # Threshold saliency
    thresh = threshold_otsu(saliency_map)
    binary = saliency_map > thresh

    # Label defects
    labeled = label(binary)
    regions = regionprops(labeled)

    # Filter by area
    defects = []
    for region in regions:
        if region.area >= min_area:
            y0, x0, y1, x1 = region.bbox
            defects.append({
                'type': 'Anomaly',  # Simple classification
                'area': region.area,
                'bbox': (x0, y0, x1, y1),
                'centroid': region.centroid
            })

    return {
        'tophat': tophat,
        'bottomhat': bottomhat,
        'saliency_map': saliency_map,
        'binary_map': binary,
        'defects': defects,
        'defect_count': len(defects)
    }

# ===========================
# PATTERN REPETITION ANALYSIS
# ===========================

# ========== BLOB DETECTION & CONNECTED COMPONENTS ==========
def analyze_blob_patterns(gray, min_area=100, max_area=5000, min_circularity=0.5, min_convexity=0.8):
    """Detect repeating patterns using blob detection"""
    try:
        # Convert to 8-bit
        gray_8bit = img_as_ubyte(gray)

        # Setup SimpleBlobDetector parameters
        params = cv2.SimpleBlobDetector_Params()

        # Filter by Area
        params.filterByArea = True
        params.minArea = min_area
        params.maxArea = max_area

        # Filter by Circularity
        params.filterByCircularity = True
        params.minCircularity = min_circularity

        # Filter by Convexity
        params.filterByConvexity = True
        params.minConvexity = min_convexity

        # Filter by Inertia
        params.filterByInertia = True
        params.minInertiaRatio = 0.01

        # Create detector
        detector = cv2.SimpleBlobDetector_create(params)

        # Detect blobs
        keypoints = detector.detect(gray_8bit)

        # Extract blob properties
        blobs = []
        for kp in keypoints:
            blobs.append({
                'center': (int(kp.pt[0]), int(kp.pt[1])),
                'size': float(kp.size),
                'area': float(np.pi * (kp.size / 2) ** 2)
            })

        # Calculate statistics
        if blobs:
            areas = [b['area'] for b in blobs]
            sizes = [b['size'] for b in blobs]
            mean_area = np.mean(areas)
            std_area = np.std(areas)
            cv_area = (std_area / mean_area * 100) if mean_area > 0 else 0
            mean_size = np.mean(sizes)
            std_size = np.std(sizes)
        else:
            mean_area = std_area = cv_area = mean_size = std_size = 0

        return {
            'blobs': blobs,
            'count': len(blobs),
            'keypoints': keypoints,
            'mean_area': mean_area,
            'std_area': std_area,
            'cv_area': cv_area,
            'mean_size': mean_size,
            'std_size': std_size
        }
    except Exception as e:
        print(f"⚠️ Blob detection failed: {e}")
        return {
            'blobs': [],
            'count': 0,
            'keypoints': [],
            'mean_area': 0,
            'std_area': 0,
            'cv_area': 0,
            'mean_size': 0,
            'std_size': 0
        }

# ========== CONNECTED COMPONENTS ANALYSIS ==========
def analyze_connected_components(gray, min_area=100, max_area=5000):
    """Analyze connected components for pattern counting"""
    try:
        # Convert to 8-bit and threshold
        gray_8bit = img_as_ubyte(gray)

        # Use Otsu thresholding
        thresh_val = threshold_otsu(gray_8bit)
        binary = gray_8bit > thresh_val

        # Label connected components
        labeled = label(binary)
        regions = regionprops(labeled)

        # Filter by area
        patterns = []
        for region in regions:
            if min_area <= region.area <= max_area:
                y0, x0, y1, x1 = region.bbox
                patterns.append({
                    'label': region.label,
                    'area': region.area,
                    'bbox': (x0, y0, x1, y1),
                    'centroid': (int(region.centroid[1]), int(region.centroid[0])),
                    'perimeter': region.perimeter,
                    'eccentricity': region.eccentricity,
                    'solidity': region.solidity
                })

        # Calculate statistics
        if patterns:
            areas = [p['area'] for p in patterns]
            mean_area = np.mean(areas)
            std_area = np.std(areas)
            cv_area = (std_area / mean_area * 100) if mean_area > 0 else 0
        else:
            mean_area = std_area = cv_area = 0

        return {
            'patterns': patterns,
            'count': len(patterns),
            'labeled_image': labeled,
            'binary_image': binary,
            'mean_area': mean_area,
            'std_area': std_area,
            'cv_area': cv_area
        }
    except Exception as e:
        print(f"⚠️ Connected components analysis failed: {e}")
        return {
            'patterns': [],
            'count': 0,
            'labeled_image': None,
            'binary_image': None,
            'mean_area': 0,
            'std_area': 0,
            'cv_area': 0
        }

# ========== KEYPOINT-BASED PATTERN MATCHING ==========
def analyze_keypoint_matching(gray_ref, gray_test, detector_type='ORB', match_threshold=0.7):
    """Match patterns using keypoint detection (SIFT, ORB, AKAZE)"""
    try:
        # Convert to 8-bit
        gray_ref_8bit = img_as_ubyte(gray_ref)
        gray_test_8bit = img_as_ubyte(gray_test)

        # Create detector based on type
        if detector_type == 'SIFT':
            try:
                detector = cv2.SIFT_create()
            except:
                detector = cv2.xfeatures2d.SIFT_create()
        elif detector_type == 'AKAZE':
            detector = cv2.AKAZE_create()
        else:  # ORB (default, patent-free)
            detector = cv2.ORB_create(nfeatures=1000)

        # Detect keypoints and compute descriptors
        kp_ref, desc_ref = detector.detectAndCompute(gray_ref_8bit, None)
        kp_test, desc_test = detector.detectAndCompute(gray_test_8bit, None)

        if desc_ref is None or desc_test is None or len(kp_ref) == 0 or len(kp_test) == 0:
            return {
                'keypoints_ref': [],
                'keypoints_test': [],
                'matches': [],
                'good_matches': [],
                'match_count': 0,
                'match_ratio': 0.0,
                'homography': None,
                'inliers': 0,
                'matching_score': 0.0
            }

        # Match descriptors
        if detector_type == 'ORB':
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        else:
            bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

        matches = bf.knnMatch(desc_ref, desc_test, k=2)

        # Apply ratio test (Lowe's ratio test)
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < match_threshold * n.distance:
                    good_matches.append(m)

        match_ratio = len(good_matches) / len(kp_ref) if len(kp_ref) > 0 else 0

        # Compute homography if enough matches
        homography = None
        inliers = 0
        if len(good_matches) >= 4:
            src_pts = np.float32([kp_ref[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_test[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            inliers = np.sum(mask) if mask is not None else 0

        # Calculate matching score
        matching_score = (len(good_matches) / max(len(kp_ref), len(kp_test))) * 100 if max(len(kp_ref), len(kp_test)) > 0 else 0

        return {
            'keypoints_ref': kp_ref,
            'keypoints_test': kp_test,
            'matches': matches,
            'good_matches': good_matches,
            'match_count': len(good_matches),
            'match_ratio': float(match_ratio),
            'homography': homography,
            'inliers': int(inliers),
            'matching_score': float(matching_score)
        }
    except Exception as e:
        print(f"⚠️ Keypoint matching failed: {e}")
        return {
            'keypoints_ref': [],
            'keypoints_test': [],
            'matches': [],
            'good_matches': [],
            'match_count': 0,
            'match_ratio': 0.0,
            'homography': None,
            'inliers': 0,
            'matching_score': 0.0
        }

# ========== AUTO-CORRELATION ANALYSIS ==========
def analyze_autocorrelation(gray):
    """Compute 2D auto-correlation to detect pattern periodicity"""
    try:
        # Normalize to zero mean
        gray_normalized = gray - np.mean(gray)

        # Compute 2D FFT
        f = np.fft.fft2(gray_normalized)
        power_spectrum = np.abs(f) ** 2

        # Inverse FFT to get auto-correlation
        autocorr = np.fft.ifft2(power_spectrum).real
        autocorr = np.fft.fftshift(autocorr)

        # Normalize
        autocorr = autocorr / autocorr.max()

        # Find peaks (excluding center)
        h, w = autocorr.shape
        cy, cx = h // 2, w // 2

        # Mask center region
        autocorr_masked = autocorr.copy()
        mask_radius = 20
        y_grid, x_grid = np.ogrid[:h, :w]
        mask = (x_grid - cx) ** 2 + (y_grid - cy) ** 2 <= mask_radius ** 2
        autocorr_masked[mask] = 0

        # Find local maxima
        from scipy.ndimage import maximum_filter
        local_max = maximum_filter(autocorr_masked, size=20)
        peaks_binary = (autocorr_masked == local_max) & (autocorr_masked > 0.1)

        # Get peak locations
        peak_coords = np.argwhere(peaks_binary)
        peaks = []
        for coord in peak_coords[:10]:  # Top 10 peaks
            y, x = coord
            distance = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
            angle = np.degrees(np.arctan2(y - cy, x - cx))
            peaks.append({
                'location': (int(x), int(y)),
                'distance': float(distance),
                'angle': float(angle),
                'value': float(autocorr[y, x])
            })

        # Calculate periodicity score (based on peak strength)
        if peaks:
            periodicity_score = np.mean([p['value'] for p in peaks]) * 100
        else:
            periodicity_score = 0

        # Estimate pattern spacing
        if peaks:
            distances = [p['distance'] for p in peaks]
            pattern_spacing = np.mean(distances)
            spacing_std = np.std(distances)
        else:
            pattern_spacing = 0
            spacing_std = 0

        return {
            'autocorr': autocorr,
            'peaks': peaks,
            'periodicity_score': float(periodicity_score),
            'pattern_spacing': float(pattern_spacing),
            'spacing_std': float(spacing_std),
            'regularity_score': float(100 - min(100, spacing_std / max(pattern_spacing, 1) * 100))
        }
    except Exception as e:
        print(f"⚠️ Auto-correlation analysis failed: {e}")
        return {
            'autocorr': np.zeros_like(gray),
            'peaks': [],
            'periodicity_score': 0.0,
            'pattern_spacing': 0.0,
            'spacing_std': 0.0,
            'regularity_score': 0.0
        }

# ========== GRID-BASED SPATIAL ANALYSIS ==========
def analyze_spatial_distribution(gray, patterns, cell_size=50):
    """Analyze pattern distribution using grid-based approach"""
    try:
        h, w = gray.shape
        n_rows = h // cell_size
        n_cols = w // cell_size

        # Create density grid
        density_grid = np.zeros((n_rows, n_cols))

        # Count patterns in each cell
        for pattern in patterns:
            cx, cy = pattern['centroid']
            grid_x = min(int(cx / cell_size), n_cols - 1)
            grid_y = min(int(cy / cell_size), n_rows - 1)
            if 0 <= grid_x < n_cols and 0 <= grid_y < n_rows:
                density_grid[grid_y, grid_x] += 1

        # Calculate uniformity metrics
        flat_density = density_grid.flatten()
        mean_density = np.mean(flat_density)
        std_density = np.std(flat_density)
        cv_density = (std_density / mean_density * 100) if mean_density > 0 else 0
        uniformity_score = max(0, 100 - cv_density)

        return {
            'density_grid': density_grid,
            'n_rows': n_rows,
            'n_cols': n_cols,
            'mean_density': float(mean_density),
            'std_density': float(std_density),
            'cv_density': float(cv_density),
            'uniformity_score': float(uniformity_score)
        }
    except Exception as e:
        print(f"⚠️ Spatial distribution analysis failed: {e}")
        return {
            'density_grid': np.zeros((1, 1)),
            'n_rows': 0,
            'n_cols': 0,
            'mean_density': 0.0,
            'std_density': 0.0,
            'cv_density': 0.0,
            'uniformity_score': 0.0
        }

# ========== PATTERN INTEGRITY ASSESSMENT ==========
def assess_pattern_integrity(patterns_ref, patterns_test, tolerance=0.15):
    """Assess integrity of patterns between reference and sample"""
    try:
        if not patterns_ref or not patterns_test:
            return {
                'integrity_score': 0.0,
                'size_similarity': 0.0,
                'shape_similarity': 0.0,
                'spatial_similarity': 0.0
            }

        # Size similarity (compare area distributions)
        areas_ref = [p['area'] for p in patterns_ref]
        areas_test = [p['area'] for p in patterns_test]
        mean_area_ref = np.mean(areas_ref)
        mean_area_test = np.mean(areas_test)
        size_diff = abs(mean_area_ref - mean_area_test) / max(mean_area_ref, 1)
        size_similarity = max(0, 100 * (1 - size_diff / tolerance))

        # Shape similarity (using solidity)
        if 'solidity' in patterns_ref[0] and 'solidity' in patterns_test[0]:
            solidity_ref = np.mean([p['solidity'] for p in patterns_ref])
            solidity_test = np.mean([p['solidity'] for p in patterns_test])
            shape_diff = abs(solidity_ref - solidity_test)
            shape_similarity = max(0, 100 * (1 - shape_diff))
        else:
            shape_similarity = 50.0

        # Spatial similarity (compare pattern spacing)
        centroids_ref = np.array([p['centroid'] for p in patterns_ref])
        centroids_test = np.array([p['centroid'] for p in patterns_test])

        if len(centroids_ref) > 1 and len(centroids_test) > 1:
            from scipy.spatial.distance import pdist
            spacing_ref = np.mean(pdist(centroids_ref))
            spacing_test = np.mean(pdist(centroids_test))
            spacing_diff = abs(spacing_ref - spacing_test) / max(spacing_ref, 1)
            spatial_similarity = max(0, 100 * (1 - spacing_diff / tolerance))
        else:
            spatial_similarity = 50.0

        # Overall integrity score
        integrity_score = (size_similarity + shape_similarity + spatial_similarity) / 3

        return {
            'integrity_score': float(integrity_score),
            'size_similarity': float(size_similarity),
            'shape_similarity': float(shape_similarity),
            'spatial_similarity': float(spatial_similarity)
        }
    except Exception as e:
        print(f"⚠️ Pattern integrity assessment failed: {e}")
        return {
            'integrity_score': 0.0,
            'size_similarity': 0.0,
            'shape_similarity': 0.0,
            'spatial_similarity': 0.0
        }

# ========== MISSING/EXTRA PATTERNS DETECTION ==========
def detect_missing_extra_patterns(patterns_ref, patterns_test, spatial_dist, tolerance=50):
    """Detect missing and extra patterns by spatial matching"""
    try:
        missing_patterns = []
        extra_patterns = []

        if not patterns_ref or not patterns_test:
            return {
                'missing_patterns': missing_patterns,
                'extra_patterns': extra_patterns,
                'missing_count': len(patterns_ref) if patterns_ref else 0,
                'extra_count': len(patterns_test) if patterns_test else 0
            }

        # Build KD-tree for efficient nearest neighbor search
        from scipy.spatial import cKDTree

        centroids_ref = np.array([p['centroid'] for p in patterns_ref])
        centroids_test = np.array([p['centroid'] for p in patterns_test])

        tree_test = cKDTree(centroids_test)
        tree_ref = cKDTree(centroids_ref)

        # Find missing patterns (in ref but not in test)
        matched_test = set()
        for i, pattern_ref in enumerate(patterns_ref):
            dist, idx = tree_test.query(centroids_ref[i])
            if dist > tolerance:
                # No match found in test - pattern is missing
                missing_patterns.append({
                    'location': pattern_ref['centroid'],
                    'expected_area': pattern_ref['area'],
                    'severity': 'High' if pattern_ref['area'] > np.median([p['area'] for p in patterns_ref]) else 'Medium'
                })
            else:
                matched_test.add(idx)

        # Find extra patterns (in test but not in ref)
        for i, pattern_test in enumerate(patterns_test):
            if i not in matched_test:
                dist, idx = tree_ref.query(centroids_test[i])
                if dist > tolerance:
                    # No match found in ref - pattern is extra
                    extra_patterns.append({
                        'location': pattern_test['centroid'],
                        'area': pattern_test['area'],
                        'severity': 'High' if pattern_test['area'] > np.median([p['area'] for p in patterns_test]) else 'Medium'
                    })

        return {
            'missing_patterns': missing_patterns,
            'extra_patterns': extra_patterns,
            'missing_count': len(missing_patterns),
            'extra_count': len(extra_patterns)
        }
    except Exception as e:
        print(f"⚠️ Missing/extra pattern detection failed: {e}")
        return {
            'missing_patterns': [],
            'extra_patterns': [],
            'missing_count': 0,
            'extra_count': 0
        }

# ----------------------------
# 4) Scoring & helpers
# ----------------------------
def color_uniformity_index(de_map):
    std = float(np.std(de_map))
    return max(0.0, 100.0 - std*10.0)

def pass_status(mean_de):
    if mean_de < 2.0: return "PASS"
    if mean_de <= 3.5: return "CONDITIONAL"
    return "FAIL"

def determine_status(value, pass_threshold, conditional_threshold, lower_is_better=True):
    """
    Unified status determination function.

    Args:
        value: The metric value to evaluate
        pass_threshold: Threshold for PASS status
        conditional_threshold: Threshold for CONDITIONAL status
        lower_is_better: If True, lower values are better (e.g., ΔE). If False, higher is better (e.g., SSIM)

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

def grid_points(h, w, n=5):
    ys = np.linspace(0.2, 0.8, n)
    xs = np.linspace(0.2, 0.8, n)
    pts = [(int(y*h), int(x*w)) for y,x in zip(ys, xs)]
    return pts[:n]

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

def pick_logo():
    """Pick the best available logo file"""
    if os.path.exists(PRIMARY_LOGO):
        return PRIMARY_LOGO
    for p in FALLBACK_LOGOS:
        if os.path.exists(p):
            return p
    logger.warning("No logo file found")
    return None

def apply_mask_to_image(img, mask):
    """Apply a binary mask to an image (supports grayscale and color)"""
    if len(img.shape) == 3:
        mask_3ch = np.stack([mask, mask, mask], axis=-1)
        return np.where(mask_3ch > 0, img, 0)
    else:
        return np.where(mask > 0, img, 0)

def apply_circular_crop(img, center_x, center_y, diameter):
    """Apply circular crop to image, masking outside as black"""
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    radius = diameter // 2
    cv2.circle(mask, (center_x, center_y), radius, 255, -1)
    return apply_mask_to_image(img, mask)

def apply_rectangular_crop(img, center_x, center_y, width, height):
    """Apply rectangular crop to image, masking outside as black"""
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

def apply_crop(img, settings):
    """Apply crop based on settings (circle or rectangle)"""
    if not settings.use_crop:
        return img

    if settings.crop_shape == "circle":
        return apply_circular_crop(img, settings.crop_center_x, settings.crop_center_y, settings.crop_diameter)
    else:  # rectangle
        return apply_rectangular_crop(img, settings.crop_center_x, settings.crop_center_y, settings.crop_width, settings.crop_height)

def draw_circle_on_image(img, center_x, center_y, diameter, color=(255, 0, 0), thickness=3):
    """Draw a circle on image for visualization"""
    img_copy = img.copy()
    radius = diameter // 2
    cv2.circle(img_copy, (center_x, center_y), radius, color, thickness)
    # Draw crosshair at center
    cross_size = 15
    cv2.line(img_copy, (center_x - cross_size, center_y), (center_x + cross_size, center_y), color, thickness)
    cv2.line(img_copy, (center_x, center_y - cross_size), (center_x, center_y + cross_size), color, thickness)
    return img_copy

def draw_rectangle_on_image(img, center_x, center_y, width, height, color=(255, 0, 0), thickness=3):
    """Draw a rectangle on image for visualization"""
    img_copy = img.copy()
    x1 = center_x - width // 2
    y1 = center_y - height // 2
    x2 = center_x + width // 2
    y2 = center_y + height // 2
    cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, thickness)
    # Draw crosshair at center
    cross_size = 15
    cv2.line(img_copy, (center_x - cross_size, center_y), (center_x + cross_size, center_y), color, thickness)
    cv2.line(img_copy, (center_x, center_y - cross_size), (center_x, center_y + cross_size), color, thickness)
    return img_copy

def draw_crop_region_on_image(img, settings, color=(255, 0, 0), thickness=3):
    """Draw crop region on image based on settings"""
    if settings.crop_shape == "circle":
        return draw_circle_on_image(img, settings.crop_center_x, settings.crop_center_y,
                                   settings.crop_diameter, color, thickness)
    else:  # rectangle
        return draw_rectangle_on_image(img, settings.crop_center_x, settings.crop_center_y,
                                      settings.crop_width, settings.crop_height, color, thickness)

def interactive_region_selector(ref_img, test_img, settings):
    """
    Interactive widget for selecting crop region on both images simultaneously.
    Updates settings in-place.
    """
    h, w = ref_img.shape[:2]

    # Initialize values from settings or use image center
    if settings.crop_center_x == 0 and settings.crop_center_y == 0:
        settings.crop_center_x = w // 2
        settings.crop_center_y = h // 2

    # Create output widget for displaying images
    image_output = Output()

    # Create control widgets
    shape_dropdown = widgets.Dropdown(
        options=['circle', 'rectangle'],
        value=settings.crop_shape,
        description='Shape:',
        style={'description_width': '120px'},
        layout=Layout(width='300px')
    )

    center_x_slider = IntText(
        value=settings.crop_center_x,
        min=0,
        max=w,
        description='Center X:',
        style={'description_width': '120px'},
        layout=Layout(width='300px')
    )

    center_y_slider = IntText(
        value=settings.crop_center_y,
        min=0,
        max=h,
        description='Center Y:',
        style={'description_width': '120px'},
        layout=Layout(width='300px')
    )

    diameter_slider = IntText(
        value=settings.crop_diameter,
        min=50,
        max=min(h, w),
        description='Diameter (px):',
        style={'description_width': '120px'},
        layout=Layout(width='300px')
    )

    width_slider = IntText(
        value=settings.crop_width,
        min=50,
        max=w,
        description='Width (px):',
        style={'description_width': '120px'},
        layout=Layout(width='300px')
    )

    height_slider = IntText(
        value=settings.crop_height,
        min=50,
        max=h,
        description='Height (px):',
        style={'description_width': '120px'},
        layout=Layout(width='300px')
    )

    apply_btn = Button(
        description='✅ Apply Selection',
        button_style='success',
        layout=Layout(width='300px', height='40px'),
        style={'button_color': '#27AE60', 'font_weight': 'bold'}
    )

    # Function to update image display
    def update_display():
        with image_output:
            image_output.clear_output(wait=True)

            # Get current values
            cx = center_x_slider.value
            cy = center_y_slider.value

            # Create temporary settings for display
            temp_settings = QCSettings()
            temp_settings.crop_shape = shape_dropdown.value
            temp_settings.crop_center_x = cx
            temp_settings.crop_center_y = cy
            temp_settings.crop_diameter = diameter_slider.value
            temp_settings.crop_width = width_slider.value
            temp_settings.crop_height = height_slider.value

            # Draw regions on both images
            ref_display = draw_crop_region_on_image(ref_img, temp_settings, color=(0, 255, 0), thickness=2)
            test_display = draw_crop_region_on_image(test_img, temp_settings, color=(0, 255, 0), thickness=2)

            # Resize for display if images are too large
            max_display_width = 500
            if w > max_display_width:
                scale = max_display_width / w
                new_w = int(w * scale)
                new_h = int(h * scale)
                ref_display = cv2.resize(ref_display, (new_w, new_h))
                test_display = cv2.resize(test_display, (new_w, new_h))

            # Combine images side by side
            combined = np.hstack([ref_display, test_display])

            # Convert to PIL and display
            combined_pil = Image.fromarray(combined)

            # Add labels
            fig, ax = plt.subplots(1, 1, figsize=(14, 7))
            ax.imshow(combined_pil)
            ax.axis('off')
            ax.set_title(f'Reference (Left) | Sample (Right) | Shape: {temp_settings.crop_shape.title()} | Center: ({cx}, {cy})',
                        fontsize=14, pad=10)
            plt.tight_layout()
            plt.show()

    # Function to toggle visibility of shape-specific controls
    def update_controls_visibility(change=None):
        if shape_dropdown.value == 'circle':
            diameter_slider.layout.display = 'block'
            width_slider.layout.display = 'none'
            height_slider.layout.display = 'none'
        else:
            diameter_slider.layout.display = 'none'
            width_slider.layout.display = 'block'
            height_slider.layout.display = 'block'
        update_display()

    # Function to apply selection
    def on_apply_clicked(b):
        settings.crop_shape = shape_dropdown.value
        settings.crop_center_x = center_x_slider.value
        settings.crop_center_y = center_y_slider.value
        settings.crop_diameter = diameter_slider.value
        settings.crop_width = width_slider.value
        settings.crop_height = height_slider.value
        settings.use_crop = True

        with image_output:
            image_output.clear_output(wait=True)
            success_html = f"""
            <div style='background: linear-gradient(135deg, #27AE60 0%, #2ECC71 100%);
                        padding: 20px; border-radius: 10px; text-align: center; margin: 10px 0;'>
                <h3 style='color: white; margin: 0;'>✅ Region Selection Applied!</h3>
                <p style='color: #ecf0f1; margin: 5px 0 0 0;'>
                    Shape: {settings.crop_shape.title()} | Center: ({settings.crop_center_x}, {settings.crop_center_y})
                </p>
            </div>
            """
            display(HTML(success_html))

    # Attach event handlers
    shape_dropdown.observe(update_controls_visibility, 'value')
    center_x_slider.observe(lambda change: update_display(), 'value')
    center_y_slider.observe(lambda change: update_display(), 'value')
    diameter_slider.observe(lambda change: update_display(), 'value')
    width_slider.observe(lambda change: update_display(), 'value')
    height_slider.observe(lambda change: update_display(), 'value')
    apply_btn.on_click(on_apply_clicked)

    # Initial display setup
    update_controls_visibility()

    # Create UI layout
    title_html = HTMLWidget(value="""
        <div style='background: linear-gradient(135deg, #3498DB 0%, #2980B9 100%);
                    padding: 15px; border-radius: 10px; text-align: center; margin: 10px 0;'>
            <h3 style='color: white; margin: 0;'>🎯 Interactive Region Selector</h3>
            <p style='color: #ecf0f1; margin: 5px 0 0 0; font-size: 12px;'>
                Adjust the parameters below to select the region of interest. The same region will be applied to both images.
            </p>
        </div>
    """)

    controls = VBox([
        title_html,
        shape_dropdown,
        center_x_slider,
        center_y_slider,
        diameter_slider,
        width_slider,
        height_slider,
        apply_btn
    ])

    # Display the UI
    display(controls)
    display(image_output)

# ----------------------------
# 5) Chart helpers (saved to PNG @ 300DPI)
# ----------------------------
TMP_IMG_DIR = ensure_dir("/content/_qc_report_imgs")

def save_fig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()

def plot_rgb_hist(img_rgb, title, path):
    data = img_rgb.reshape(-1,3)
    plt.figure(figsize=(6,2.6))
    plt.hist(data[:,0], bins=32, alpha=0.6, label='R')
    plt.hist(data[:,1], bins=32, alpha=0.6, label='G')
    plt.hist(data[:,2], bins=32, alpha=0.6, label='B')
    plt.title(title)
    plt.xlabel("Value")
    plt.ylabel("Count")
    plt.legend()
    save_fig(path)

def plot_heatmap(de_map, title, path):
    vmax = np.percentile(de_map, 99)
    plt.figure(figsize=(7,3))
    im = plt.imshow(de_map, cmap="inferno", vmin=0, vmax=vmax)
    plt.title(title)
    plt.axis("off")
    plt.colorbar(im, fraction=0.025)
    save_fig(path)

def plot_spectral_proxy(mean_rgb_ref, mean_rgb_test, path):
    # Build a simple proxy spectral curve using Gaussians for RGB primaries
    wl = np.linspace(380, 700, 161)
    def gaussian(w, mu, sigma):
        return np.exp(-0.5*((w-mu)/sigma)**2)
    # Centers approx: B~450, G~545, R~610 nm
    base_R = gaussian(wl, 610, 28)
    base_G = gaussian(wl, 545, 25)
    base_B = gaussian(wl, 450, 22)
    ref_curve = (mean_rgb_ref[0]*base_R + mean_rgb_ref[1]*base_G + mean_rgb_ref[2]*base_B)
    test_curve= (mean_rgb_test[0]*base_R + mean_rgb_test[1]*base_G + mean_rgb_test[2]*base_B)
    ref_curve /= ref_curve.max()+1e-8
    test_curve/= test_curve.max()+1e-8
    plt.figure(figsize=(7,3))
    plt.plot(wl, ref_curve, label="Reference")
    plt.plot(wl, test_curve, label="Sample")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Relative intensity")
    plt.title("Spectral Distribution (Proxy from RGB)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    save_fig(path)

def plot_ab_scatter(lab_ref, lab_test, path):
    a_ref = lab_ref[...,1].flatten()
    b_ref = lab_ref[...,2].flatten()
    a_test = lab_test[...,1].flatten()
    b_test = lab_test[...,2].flatten()
    plt.figure(figsize=(5,5))
    plt.axhline(0, color='k', lw=0.5)
    plt.axvline(0, color='k', lw=0.5)
    plt.scatter(a_ref[::100], b_ref[::100], s=6, alpha=0.4, label="Ref")
    plt.scatter(a_test[::100], b_test[::100], s=6, alpha=0.4, label="Sample")
    plt.xlabel("a* (green − red)")
    plt.ylabel("b* (blue − yellow)")
    plt.title("a* vs b* Scatter")
    plt.legend()
    save_fig(path)

def plot_lab_bars(lab_ref_mean, lab_test_mean, path):
    labels = ["L*", "a*", "b*"]
    ref_vals = [lab_ref_mean[0], lab_ref_mean[1], lab_ref_mean[2]]
    tst_vals = [lab_test_mean[0], lab_test_mean[1], lab_test_mean[2]]
    x = np.arange(len(labels))
    w = 0.35
    plt.figure(figsize=(6,3))
    plt.bar(x-w/2, ref_vals, width=w, label="Ref")
    plt.bar(x+w/2, tst_vals, width=w, label="Sample")
    plt.xticks(x, labels)
    plt.title("Lab Components — Mean")
    plt.legend()
    save_fig(path)

def overlay_regions(img, pts, radius=12):
    pil = Image.fromarray(img.copy())
    drw = ImageDraw.Draw(pil)
    for (y,x) in pts:
        drw.ellipse([(x-radius, y-radius), (x+radius, y+radius)], outline=(255,0,0), width=3)
    return np.array(pil)

# ----------------------------
# 5b) ADVANCED VISUALIZATION FUNCTIONS
# ----------------------------

def plot_fft_power_spectrum(power_spectrum, peaks, path):
    """Plot FFT power spectrum with peaks"""
    plt.figure(figsize=(7, 5))
    plt.imshow(power_spectrum, cmap='hot', origin='lower')
    plt.colorbar(label='Log Magnitude')
    plt.title('2D FFT Power Spectrum')

    # Mark peaks
    h, w = power_spectrum.shape
    cy, cx = h // 2, w // 2
    for i, peak in enumerate(peaks[:5]):
        y = cy + peak['radius'] * np.sin(np.radians(peak['angle']))
        x = cx + peak['radius'] * np.cos(np.radians(peak['angle']))
        plt.plot(x, y, 'go', markersize=8)
        plt.text(x+5, y+5, f"P{i+1}", color='white', fontsize=8)

    plt.xlabel('Frequency X')
    plt.ylabel('Frequency Y')
    save_fig(path)

def plot_gabor_montage(energy_maps, frequencies, num_orientations, path):
    """Plot Gabor filter response montage"""
    n_freq = len(frequencies)
    n_orient = num_orientations

    fig, axes = plt.subplots(n_freq, min(n_orient, 8), figsize=(12, n_freq * 1.5))
    if n_freq == 1:
        axes = axes[np.newaxis, :]

    idx = 0
    for i, freq in enumerate(frequencies):
        for j in range(min(n_orient, 8)):
            if idx < len(energy_maps):
                axes[i, j].imshow(energy_maps[idx], cmap='viridis')
                axes[i, j].axis('off')
                axes[i, j].set_title(f"{freq:.2f}, {j*180//n_orient}°", fontsize=8)
            idx += 1

    plt.suptitle('Gabor Filter Bank Responses', fontsize=12, fontweight='bold')
    plt.tight_layout()
    save_fig(path)

def plot_gabor_orientation_histogram(gabor_results, path):
    """Plot orientation histogram from Gabor"""
    orientations = [r['orientation_deg'] for r in gabor_results]
    energies = [r['mean'] for r in gabor_results]

    plt.figure(figsize=(6, 4))
    plt.bar(orientations, energies, width=15, alpha=0.7, color='steelblue')
    plt.xlabel('Orientation (degrees)')
    plt.ylabel('Mean Energy')
    plt.title('Gabor Orientation Energy Distribution')
    plt.grid(True, alpha=0.3)
    save_fig(path)

def plot_glcm_radar(glcm_props_ref, glcm_props_sample, path):
    """Radar chart for GLCM features"""
    categories = list(glcm_props_ref.keys())
    ref_values = [glcm_props_ref[k] for k in categories]
    sample_values = [glcm_props_sample[k] for k in categories]

    # Normalize to 0-1 for radar
    max_val = max(max(ref_values), max(sample_values)) + 1e-8
    ref_norm = [v / max_val for v in ref_values]
    sample_norm = [v / max_val for v in sample_values]

    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    ref_norm += ref_norm[:1]
    sample_norm += sample_norm[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(projection='polar'))
    ax.plot(angles, ref_norm, 'o-', linewidth=2, label='Reference', color='green')
    ax.fill(angles, ref_norm, alpha=0.15, color='green')
    ax.plot(angles, sample_norm, 'o-', linewidth=2, label='Sample', color='red')
    ax.fill(angles, sample_norm, alpha=0.15, color='red')
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=9)
    ax.set_ylim(0, 1)
    ax.set_title('GLCM Texture Features (Normalized)', size=12, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax.grid(True)
    save_fig(path)

def plot_lbp_map_and_hist(lbp_map, hist_ref, hist_sample, path):
    """Plot LBP map and histogram comparison"""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # LBP map (sample)
    axes[0].imshow(lbp_map, cmap='gray')
    axes[0].set_title('LBP Map (Sample)')
    axes[0].axis('off')

    # Histogram comparison
    x = np.arange(len(hist_ref))
    axes[1].bar(x - 0.2, hist_ref, width=0.4, alpha=0.7, label='Reference', color='green')
    axes[1].bar(x + 0.2, hist_sample, width=0.4, alpha=0.7, label='Sample', color='red')
    axes[1].set_xlabel('LBP Bin')
    axes[1].set_ylabel('Normalized Frequency')
    axes[1].set_title('LBP Histogram Comparison')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_fig(path)

def plot_wavelet_energy_bars(energies_ref, energies_sample, path):
    """Plot wavelet energy comparison"""
    levels = [e['level'] for e in energies_ref]

    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    bands = ['LH', 'HL', 'HH', 'total']
    colors_ref = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c']
    colors_sample = ['#27ae60', '#2980b9', '#8e44ad', '#c0392b']

    for idx, band in enumerate(bands):
        ax = axes[idx // 2, idx % 2]
        ref_vals = [e[band] for e in energies_ref]
        sample_vals = [e[band] for e in energies_sample]

        x = np.arange(len(levels))
        ax.bar(x - 0.2, ref_vals, width=0.4, alpha=0.7, label='Reference', color=colors_ref[idx])
        ax.bar(x + 0.2, sample_vals, width=0.4, alpha=0.7, label='Sample', color=colors_sample[idx])
        ax.set_xlabel('Level')
        ax.set_ylabel('Energy')
        ax.set_title(f'{band} Band Energy')
        ax.set_xticks(x)
        ax.set_xticklabels(levels)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle('Wavelet Decomposition Energy', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_fig(path)

def plot_defect_saliency(saliency_map, binary_map, defects, original_shape, path):
    """Plot defect saliency and detection results"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Saliency map
    axes[0].imshow(saliency_map, cmap='hot')
    axes[0].set_title('Saliency Map')
    axes[0].axis('off')

    # Binary map
    axes[1].imshow(binary_map, cmap='gray')
    axes[1].set_title(f'Binary Defect Map ({len(defects)} defects)')
    axes[1].axis('off')

    # Defects overlay
    overlay = np.zeros((*original_shape, 3), dtype=np.uint8)
    for defect in defects:
        x0, y0, x1, y1 = defect['bbox']
        cv2.rectangle(overlay, (x0, y0), (x1, y1), (255, 0, 0), 2)
    axes[2].imshow(overlay)
    axes[2].set_title('Detected Defects')
    axes[2].axis('off')

    plt.tight_layout()
    save_fig(path)

def plot_metamerism_illuminants(illuminants, delta_e_values, path):
    """Plot ΔE across different illuminants"""
    plt.figure(figsize=(8, 5))
    x = np.arange(len(illuminants))
    bars = plt.bar(x, delta_e_values, alpha=0.7, color='steelblue', edgecolor='navy')

    # Color bars by severity
    for i, de in enumerate(delta_e_values):
        if de < 2.0:
            bars[i].set_color('#27ae60')
        elif de < 3.5:
            bars[i].set_color('#f39c12')
        else:
            bars[i].set_color('#e74c3c')

    plt.axhline(y=2.0, color='green', linestyle='--', linewidth=1, alpha=0.5, label='PASS threshold')
    plt.axhline(y=3.5, color='orange', linestyle='--', linewidth=1, alpha=0.5, label='Conditional threshold')

    plt.xticks(x, illuminants, rotation=45)
    plt.xlabel('Illuminant')
    plt.ylabel('ΔE2000')
    plt.title('Metamerism Analysis: ΔE Across Illuminants')
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    save_fig(path)

def plot_spectral_curve(wavelengths, reflectance_ref, reflectance_sample, path):
    """Plot true spectral reflectance curves"""
    plt.figure(figsize=(8, 5))
    plt.plot(wavelengths, reflectance_ref, label='Reference', linewidth=2, color='green', marker='o', markersize=3)
    plt.plot(wavelengths, reflectance_sample, label='Sample', linewidth=2, color='red', marker='s', markersize=3)
    plt.xlabel('Wavelength (nm)', fontsize=12)
    plt.ylabel('Reflectance (%)', fontsize=12)
    plt.title('True Spectral Reflectance Curve', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.xlim(380, 700)
    plt.ylim(0, 100)
    # Add colored background for visible spectrum
    plt.axvspan(380, 450, alpha=0.1, color='blue', label='_')
    plt.axvspan(450, 495, alpha=0.1, color='cyan', label='_')
    plt.axvspan(495, 570, alpha=0.1, color='green', label='_')
    plt.axvspan(570, 590, alpha=0.1, color='yellow', label='_')
    plt.axvspan(590, 620, alpha=0.1, color='orange', label='_')
    plt.axvspan(620, 700, alpha=0.1, color='red', label='_')
    plt.tight_layout()
    save_fig(path)

def plot_line_angle_histogram(orientation_degrees, path):
    """Plot line angle histogram from structure tensor"""
    plt.figure(figsize=(7, 4))

    # Create histogram
    bins = np.arange(-90, 91, 10)
    hist, bin_edges = np.histogram(orientation_degrees, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    plt.bar(bin_centers, hist, width=8, alpha=0.7, color='steelblue', edgecolor='navy')
    plt.xlabel('Orientation Angle (degrees)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Line Angle Distribution (Structure Tensor)', fontsize=13, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    plt.xlim(-90, 90)

    # Add reference lines
    plt.axvline(0, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Horizontal')
    plt.axvline(90, color='green', linestyle='--', linewidth=1, alpha=0.5, label='Vertical')
    plt.axvline(-90, color='green', linestyle='--', linewidth=1, alpha=0.5)

    plt.legend(fontsize=9)
    plt.tight_layout()
    save_fig(path)

# ----------------------------
# 5c) PATTERN REPETITION VISUALIZATIONS
# ----------------------------

def plot_pattern_detection_map(img_rgb, patterns, title, path):
    """Plot original image with detected patterns marked"""
    plt.figure(figsize=(8, 6))
    img_display = img_rgb.copy()

    # Draw bounding boxes or circles for each pattern
    for pattern in patterns:
        if 'bbox' in pattern:
            x0, y0, x1, y1 = pattern['bbox']
            cv2.rectangle(img_display, (x0, y0), (x1, y1), (0, 255, 0), 2)
        if 'centroid' in pattern:
            cx, cy = pattern['centroid']
            cv2.circle(img_display, (cx, cy), 5, (255, 0, 0), -1)

    plt.imshow(img_display)
    plt.title(f'{title} ({len(patterns)} patterns detected)', fontsize=13, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    save_fig(path)

def plot_pattern_count_comparison(count_ref, count_test, path):
    """Bar chart comparing pattern counts"""
    plt.figure(figsize=(7, 5))

    categories = ['Reference', 'Sample']
    counts = [count_ref, count_test]
    colors_bar = ['#27ae60', '#e74c3c']

    bars = plt.bar(categories, counts, color=colors_bar, alpha=0.7, edgecolor='navy', linewidth=1.5)

    # Add value labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(count)}',
                ha='center', va='bottom', fontsize=14, fontweight='bold')

    plt.ylabel('Pattern Count', fontsize=12)
    plt.title('Pattern Count Comparison', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    plt.ylim(0, max(counts) * 1.2)

    # Add difference annotation
    diff = abs(count_ref - count_test)
    plt.text(0.5, max(counts) * 1.05, f'Δ = {int(diff)}', ha='center', fontsize=12,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    save_fig(path)

def plot_pattern_density_heatmap(density_grid, path):
    """Heatmap showing pattern density across grid cells"""
    plt.figure(figsize=(8, 6))

    im = plt.imshow(density_grid, cmap='YlOrRd', interpolation='nearest')
    plt.colorbar(im, label='Pattern Count per Cell')
    plt.title('Pattern Density Heatmap', fontsize=14, fontweight='bold')
    plt.xlabel('Grid Column', fontsize=11)
    plt.ylabel('Grid Row', fontsize=11)

    # Add grid lines
    ax = plt.gca()
    ax.set_xticks(np.arange(-.5, density_grid.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-.5, density_grid.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linestyle='-', linewidth=1.5)

    plt.tight_layout()
    save_fig(path)

def plot_missing_extra_patterns(img_rgb, missing_patterns, extra_patterns, path):
    """Visual overlay showing missing (red) and extra (blue) patterns"""
    plt.figure(figsize=(8, 6))
    img_display = img_rgb.copy()

    # Draw missing patterns (red circles)
    for pattern in missing_patterns:
        cx, cy = pattern['location']
        cv2.circle(img_display, (cx, cy), 15, (255, 0, 0), 3)
        cv2.circle(img_display, (cx, cy), 3, (255, 0, 0), -1)

    # Draw extra patterns (blue circles)
    for pattern in extra_patterns:
        cx, cy = pattern['location']
        cv2.circle(img_display, (cx, cy), 15, (0, 0, 255), 3)
        cv2.circle(img_display, (cx, cy), 3, (0, 0, 255), -1)

    plt.imshow(img_display)
    plt.title(f'Missing (Red: {len(missing_patterns)}) / Extra (Blue: {len(extra_patterns)}) Patterns',
             fontsize=12, fontweight='bold')
    plt.axis('off')

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='red', label=f'Missing ({len(missing_patterns)})'),
                       Patch(facecolor='blue', label=f'Extra ({len(extra_patterns)})')]
    plt.legend(handles=legend_elements, loc='upper right', fontsize=10)

    plt.tight_layout()
    save_fig(path)

def plot_pattern_size_distribution(areas_ref, areas_test, path):
    """Histogram comparing pattern size distributions"""
    plt.figure(figsize=(8, 5))

    # Determine bin range
    all_areas = list(areas_ref) + list(areas_test)
    bins = np.linspace(min(all_areas), max(all_areas), 20)

    plt.hist(areas_ref, bins=bins, alpha=0.6, label='Reference', color='green', edgecolor='black')
    plt.hist(areas_test, bins=bins, alpha=0.6, label='Sample', color='red', edgecolor='black')

    plt.xlabel('Pattern Area (px²)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Pattern Size Distribution Comparison', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3, axis='y')

    # Add mean lines
    if areas_ref:
        plt.axvline(np.mean(areas_ref), color='green', linestyle='--', linewidth=2,
                   label=f'Ref Mean: {np.mean(areas_ref):.1f}')
    if areas_test:
        plt.axvline(np.mean(areas_test), color='red', linestyle='--', linewidth=2,
                   label=f'Sample Mean: {np.mean(areas_test):.1f}')

    plt.legend(fontsize=9)
    plt.tight_layout()
    save_fig(path)

def plot_autocorrelation_surface(autocorr, peaks, path):
    """3D surface plot of auto-correlation"""
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Subsample for performance
    h, w = autocorr.shape
    step = max(1, h // 100)
    autocorr_sub = autocorr[::step, ::step]

    # Create meshgrid
    X, Y = np.meshgrid(np.arange(autocorr_sub.shape[1]), np.arange(autocorr_sub.shape[0]))

    # Plot surface
    surf = ax.plot_surface(X, Y, autocorr_sub, cmap='viridis', alpha=0.8,
                           linewidth=0, antialiased=True)

    ax.set_xlabel('X Position', fontsize=10)
    ax.set_ylabel('Y Position', fontsize=10)
    ax.set_zlabel('Correlation', fontsize=10)
    ax.set_title('Auto-correlation Surface (Pattern Periodicity)', fontsize=13, fontweight='bold')

    # Add colorbar
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.tight_layout()
    save_fig(path)

def plot_keypoint_matching(img_ref, img_test, kp_ref, kp_test, good_matches, path):
    """Visualization of matched keypoints between reference and sample"""
    try:
        # Draw matches
        img_ref_8bit = img_as_ubyte(rgb2gray(img_ref)) if len(img_ref.shape) == 3 else img_as_ubyte(img_ref)
        img_test_8bit = img_as_ubyte(rgb2gray(img_test)) if len(img_test.shape) == 3 else img_as_ubyte(img_test)

        # Convert to BGR for cv2.drawMatches
        img_ref_bgr = cv2.cvtColor(img_ref_8bit, cv2.COLOR_GRAY2BGR) if len(img_ref_8bit.shape) == 2 else img_ref
        img_test_bgr = cv2.cvtColor(img_test_8bit, cv2.COLOR_GRAY2BGR) if len(img_test_8bit.shape) == 2 else img_test

        # Draw only top 50 matches for clarity
        matches_to_draw = good_matches[:50]

        img_matches = cv2.drawMatches(img_ref_bgr, kp_ref, img_test_bgr, kp_test,
                                      matches_to_draw, None,
                                      matchColor=(0, 255, 0),
                                      singlePointColor=(255, 0, 0),
                                      flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        plt.figure(figsize=(12, 6))
        plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
        plt.title(f'Keypoint Matching ({len(good_matches)} matches, showing top 50)',
                 fontsize=13, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        save_fig(path)
    except Exception as e:
        print(f"⚠️ Keypoint matching visualization failed: {e}")
        # Create placeholder
        plt.figure(figsize=(12, 6))
        plt.text(0.5, 0.5, f'Keypoint Matching\n{len(good_matches)} matches found',
                ha='center', va='center', fontsize=14)
        plt.axis('off')
        plt.tight_layout()
        save_fig(path)

def plot_blob_detection(img_rgb, keypoints, path):
    """Visualization of blob detection results"""
    plt.figure(figsize=(8, 6))

    # Draw blobs
    img_with_blobs = cv2.drawKeypoints(img_rgb, keypoints, None,
                                       color=(0, 255, 0),
                                       flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    plt.imshow(img_with_blobs)
    plt.title(f'Blob Detection ({len(keypoints)} blobs)', fontsize=13, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    save_fig(path)

def plot_pattern_integrity_radar(integrity_data_ref, integrity_data_test, path):
    """Radar chart for pattern integrity comparison"""
    categories = ['Size\nSimilarity', 'Shape\nSimilarity', 'Spatial\nSimilarity', 'Overall\nIntegrity']

    # Get values (scale to 0-1)
    ref_values = [
        integrity_data_ref.get('size_similarity', 0) / 100,
        integrity_data_ref.get('shape_similarity', 0) / 100,
        integrity_data_ref.get('spatial_similarity', 0) / 100,
        integrity_data_ref.get('integrity_score', 0) / 100
    ]

    test_values = [
        integrity_data_test.get('size_similarity', 0) / 100,
        integrity_data_test.get('shape_similarity', 0) / 100,
        integrity_data_test.get('spatial_similarity', 0) / 100,
        integrity_data_test.get('integrity_score', 0) / 100
    ]

    # Number of variables
    N = len(categories)

    # Compute angle for each axis
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()

    # Complete the circle
    ref_values += ref_values[:1]
    test_values += test_values[:1]
    angles += angles[:1]

    # Plot
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(projection='polar'))

    ax.plot(angles, ref_values, 'o-', linewidth=2, label='Reference', color='green')
    ax.fill(angles, ref_values, alpha=0.15, color='green')

    ax.plot(angles, test_values, 'o-', linewidth=2, label='Sample', color='red')
    ax.fill(angles, test_values, alpha=0.15, color='red')

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=10)
    ax.set_ylim(0, 1)
    ax.set_title('Pattern Integrity Assessment', size=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
    ax.grid(True)

    plt.tight_layout()
    save_fig(path)

# ----------------------------
# 6) PDF helpers (ReportLab)
# ----------------------------
styles = getSampleStyleSheet()
StyleTitle = ParagraphStyle("Title", parent=styles["Heading1"], fontName="Helvetica-Bold",
                            fontSize=20, textColor=NEUTRAL_DARK, leading=24, alignment=TA_LEFT)
StyleH1 = ParagraphStyle("H1", parent=styles["Heading2"], fontName="Helvetica-Bold",
                         fontSize=15, textColor=BLUE1, leading=18, spaceAfter=8)
StyleH2 = ParagraphStyle("H2", parent=styles["Heading3"], fontName="Helvetica-Bold",
                         fontSize=12.5, textColor=BLUE2, leading=16, spaceAfter=6)
StyleBody = ParagraphStyle("Body", parent=styles["BodyText"], fontName="Helvetica",
                           fontSize=10, leading=14)
StyleSmall = ParagraphStyle("Small", parent=styles["BodyText"], fontName="Helvetica",
                            fontSize=9, leading=12, textColor=NEUTRAL)
StyleBadge = ParagraphStyle("Badge", parent=styles["BodyText"], fontName="Helvetica-Bold",
                            fontSize=10.5, leading=14, textColor=colors.white,
                            alignment=TA_CENTER)

def badge(text, back_color=NEUTRAL):
    # little colored label as a Flowable
    class _Badge(Flowable):
        def __init__(self, t, bg):
            super().__init__()
            self.t = t
            self.bg = bg
            self.w = max(60, 8*len(t))
            self.h = 16
        def draw(self):
            self.canv.setFillColor(self.bg)
            self.canv.roundRect(0,0,self.w,self.h,3,fill=1,stroke=0)
            self.canv.setFillColor(colors.white)
            self.canv.setFont("Helvetica-Bold", 9)
            self.canv.drawCentredString(self.w/2, 3, self.t)
        def wrap(self, availW, availH):
            return (self.w, self.h)
    return _Badge(text, back_color)

def fmt_pct(x):
    return f"{x:.1f}%"

def fmt2(x):
    return f"{x:.2f}"

def fmt1(x):
    return f"{x:.1f}"

def colored_status_cell(text, status):
    col = STATUS_COLORS.get(status, NEUTRAL)
    return [Paragraph(f"<b>{text}</b>", ParagraphStyle("s", textColor=colors.white, alignment=TA_CENTER)),
            col]

def make_table(data, colWidths=None, alt=True, header_bg=NEUTRAL_L):
    t = Table(data, colWidths=colWidths, hAlign="LEFT", repeatRows=1)
    style_cmds = [
        ("BACKGROUND", (0,0), (-1,0), header_bg),
        ("TEXTCOLOR", (0,0), (-1,0), colors.black),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE", (0,0), (-1,0), 10),
        ("BOTTOMPADDING", (0,0), (-1,0), 6),
        ("TOPPADDING", (0,0), (-1,0), 6),
        ("GRID", (0,0), (-1,-1), 0.25, colors.Color(0.8,0.8,0.8)),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("LEFTPADDING", (0,0), (-1,-1), 5),
        ("RIGHTPADDING", (0,0), (-1,-1), 5),
        ("TOPPADDING", (0,1), (-1,-1), 4),
        ("BOTTOMPADDING", (0,1), (-1,-1), 4),
        ("WORDWRAP", (0,0), (-1,-1), True),
    ]
    if alt:
        style_cmds += [("BACKGROUND", (0,i), (-1,i), colors.whitesmoke) for i in range(1,len(data),2)]
    t.setStyle(TableStyle(style_cmds))
    return t

# Header/Footer drawing
def header_footer(canvas_, doc):
    canvas_.saveState()
    width, height = PAGE_SIZE

    # Draw white rectangular frame with 3mm margins on all sides
    canvas_.setStrokeColor(colors.HexColor("#E0E0E0"))  # Light gray frame
    canvas_.setLineWidth(0.8)
    frame_x1 = FRAME_MARGIN
    frame_y1 = FRAME_MARGIN
    frame_x2 = width - FRAME_MARGIN
    frame_y2 = height - FRAME_MARGIN
    canvas_.rect(frame_x1, frame_y1, frame_x2 - frame_x1, frame_y2 - frame_y1, stroke=1, fill=0)

    # Header
    y = height - 40
    # line
    canvas_.setStrokeColor(NEUTRAL_L)
    canvas_.setLineWidth(0.6)
    canvas_.line(MARGIN_L, y, width - MARGIN_R, y)

    # company text - Company Name (blue, bold) | Subtitle (gray, smaller)
    canvas_.setFillColor(BLUE1)
    canvas_.setFont("Helvetica-Bold", 10.5)
    canvas_.drawString(MARGIN_L, y+10, COMPANY_NAME)

    # Calculate position for pipe symbol
    company_width = canvas_.stringWidth(COMPANY_NAME, "Helvetica-Bold", 10.5)

    # Draw black pipe symbol at current size
    canvas_.setFillColor(colors.black)
    canvas_.setFont("Helvetica-Bold", 10.5)
    canvas_.drawString(MARGIN_L + company_width + 5, y+10, " | ")

    # Calculate position for subtitle
    pipe_width = canvas_.stringWidth(" | ", "Helvetica-Bold", 10.5)

    # Draw gray subtitle in smaller font
    canvas_.setFillColor(NEUTRAL)  # Gray color
    canvas_.setFont("Helvetica", 8.5)  # Smaller font
    canvas_.drawString(MARGIN_L + company_width + pipe_width + 5, y+10, COMPANY_SUBTITLE)

    # Footer
    fy = 35  # Increased from 28 to provide safe distance from bottom
    canvas_.setStrokeColor(NEUTRAL_L)
    canvas_.setLineWidth(0.6)
    canvas_.line(MARGIN_L, fy+10, width - MARGIN_R, fy+10)

    # page number (start numbering so that "Color Unit" page is 2)
    pno = canvas_.getPageNumber()
    # first page (cover) is unnumbered; subsequent pages offset by +0 so second phys page shows "2"
    if pno >= 2:
        canvas_.setFillColor(NEUTRAL)
        canvas_.setFont("Helvetica", 9)
        canvas_.drawRightString(width - MARGIN_R, fy-2, f"Page {pno}")

        # Company logo on opposite side (left) - MUST use logo_vertical_512x256.png
        footer_logo = "logo_vertical_512x256.png"
        if os.path.exists(footer_logo):
            # Draw small logo thumbnail (aspect ratio 2:1 for vertical logo)
            logo_height = 20
            logo_width = 40  # 2:1 aspect ratio for logo_vertical_512x256.png
            try:
                canvas_.drawImage(footer_logo, MARGIN_L, fy-5, width=logo_width, height=logo_height,
                                preserveAspectRatio=True, mask='auto')
            except:
                pass  # Silently fail if logo cannot be drawn

    canvas_.restoreState()

def create_report_sections_ui(settings):
    """Create interactive report sections selector with hierarchical checkboxes"""

    section_style = "background: #f8f9fa; padding: 15px; margin: 10px 0; border-radius: 8px; border-left: 4px solid #F39C12;"
    subsection_style = "margin-left: 30px; padding: 5px 0;"

    title_html = """
        <div style='background: linear-gradient(135deg, #F39C12 0%, #E67E22 100%);
                    padding: 20px; border-radius: 10px 10px 0 0; margin-bottom: 0;'>
            <h2 style='color: white; margin: 0; font-family: Arial, sans-serif;'>
                📋 Report Sections Control
            </h2>
            <p style='color: #ecf0f1; margin: 10px 0 0 0; font-size: 13px;'>
                Enable or disable sections to customize your report and reduce processing time.
            </p>
        </div>
    """

    info_html = """
        <div style='background: #fff3cd; padding: 12px; margin: 10px 0; border-radius: 5px; border-left: 4px solid #F39C12;'>
            <p style='margin: 0; color: #856404; font-size: 12px;'>
                <b>💡 Tip:</b> Disabling a section will skip its processing entirely, saving time.
                Disabling a main section will also disable all its sub-sections.
            </p>
        </div>
    """

    # Create checkboxes for all sections
    # Main section: Analysis Settings
    chk_analysis_settings = widgets.Checkbox(
        value=settings.enable_analysis_settings,
        description='Analysis Settings',
        style={'description_width': 'initial', 'font_weight': 'bold'},
        layout=Layout(width='400px')
    )

    # Main section: Color Unit
    chk_color_unit = widgets.Checkbox(
        value=settings.enable_color_unit,
        description='Color Unit',
        style={'description_width': 'initial', 'font_weight': 'bold'},
        layout=Layout(width='400px')
    )

    # Color Unit sub-sections
    chk_color_input_images = widgets.Checkbox(
        value=settings.enable_color_input_images,
        description='Input Images',
        indent=False,
        layout=Layout(width='380px', margin='0 0 0 30px')
    )

    chk_color_measurements = widgets.Checkbox(
        value=settings.enable_color_measurements,
        description='Color Measurements',
        indent=False,
        layout=Layout(width='380px', margin='0 0 0 30px')
    )

    chk_color_difference = widgets.Checkbox(
        value=settings.enable_color_difference,
        description='Color Difference (ΔE)',
        indent=False,
        layout=Layout(width='380px', margin='0 0 0 30px')
    )

    chk_color_statistical = widgets.Checkbox(
        value=settings.enable_color_statistical,
        description='Statistical Analysis (RGB)',
        indent=False,
        layout=Layout(width='380px', margin='0 0 0 30px')
    )

    chk_color_spectral_proxy = widgets.Checkbox(
        value=settings.enable_color_spectral_proxy,
        description='Spectral Analysis (Proxy)',
        indent=False,
        layout=Layout(width='380px', margin='0 0 0 30px')
    )

    chk_color_visual_diff = widgets.Checkbox(
        value=settings.enable_color_visual_diff,
        description='Visual Difference Analysis',
        indent=False,
        layout=Layout(width='380px', margin='0 0 0 30px')
    )

    chk_color_lab_detailed = widgets.Checkbox(
        value=settings.enable_color_lab_detailed,
        description='Detailed Lab* Color Space Analysis',
        indent=False,
        layout=Layout(width='380px', margin='0 0 0 30px')
    )

    chk_color_lab_viz = widgets.Checkbox(
        value=settings.enable_color_lab_viz,
        description='Lab* Visualizations',
        indent=False,
        layout=Layout(width='380px', margin='0 0 0 30px')
    )

    chk_color_quality_assessment = widgets.Checkbox(
        value=settings.enable_color_quality_assessment,
        description='Quality Assessment (Lab* thresholds)',
        indent=False,
        layout=Layout(width='380px', margin='0 0 0 30px')
    )

    chk_color_scoring = widgets.Checkbox(
        value=settings.enable_color_scoring,
        description='Scoring & Status',
        indent=False,
        layout=Layout(width='380px', margin='0 0 0 30px')
    )

    chk_color_recommendations = widgets.Checkbox(
        value=settings.enable_color_recommendations,
        description='Recommendations',
        indent=False,
        layout=Layout(width='380px', margin='0 0 0 30px')
    )

    # Main section: Pattern Unit
    chk_pattern_unit = widgets.Checkbox(
        value=settings.enable_pattern_unit,
        description='Pattern Unit',
        style={'description_width': 'initial', 'font_weight': 'bold'},
        layout=Layout(width='400px')
    )

    # Pattern Unit sub-sections
    chk_pattern_ssim = widgets.Checkbox(
        value=settings.enable_pattern_ssim,
        description='SSIM Analysis',
        indent=False,
        layout=Layout(width='380px', margin='0 0 0 30px')
    )

    chk_pattern_symmetry = widgets.Checkbox(
        value=settings.enable_pattern_symmetry,
        description='Symmetry Analysis',
        indent=False,
        layout=Layout(width='380px', margin='0 0 0 30px')
    )

    chk_pattern_edge = widgets.Checkbox(
        value=settings.enable_pattern_edge,
        description='Edge Definition',
        indent=False,
        layout=Layout(width='380px', margin='0 0 0 30px')
    )

    chk_pattern_repeat = widgets.Checkbox(
        value=settings.enable_pattern_repeat,
        description='Repeat Period Estimation',
        indent=False,
        layout=Layout(width='380px', margin='0 0 0 30px')
    )

    chk_pattern_advanced = widgets.Checkbox(
        value=settings.enable_pattern_advanced,
        description='Advanced Texture Analysis',
        indent=False,
        layout=Layout(width='380px', margin='0 0 0 30px')
    )

    # Main section: Pattern Repetition Unit
    chk_pattern_repetition = widgets.Checkbox(
        value=settings.enable_pattern_repetition,
        description='Pattern Repetition Unit',
        style={'description_width': 'initial', 'font_weight': 'bold'},
        layout=Layout(width='400px')
    )

    # Pattern Repetition sub-sections
    chk_pattern_rep_summary = widgets.Checkbox(
        value=settings.enable_pattern_rep_summary,
        description='Pattern Detection Summary',
        indent=False,
        layout=Layout(width='380px', margin='0 0 0 30px')
    )

    chk_pattern_rep_count = widgets.Checkbox(
        value=settings.enable_pattern_rep_count,
        description='Pattern Count Analysis',
        indent=False,
        layout=Layout(width='380px', margin='0 0 0 30px')
    )

    chk_pattern_rep_blob = widgets.Checkbox(
        value=settings.enable_pattern_rep_blob,
        description='Blob Detection Results',
        indent=False,
        layout=Layout(width='380px', margin='0 0 0 30px')
    )

    chk_pattern_rep_keypoint = widgets.Checkbox(
        value=settings.enable_pattern_rep_keypoint,
        description='Keypoint Matching',
        indent=False,
        layout=Layout(width='380px', margin='0 0 0 30px')
    )

    chk_pattern_rep_autocorr = widgets.Checkbox(
        value=settings.enable_pattern_rep_autocorr,
        description='Auto-correlation Analysis',
        indent=False,
        layout=Layout(width='380px', margin='0 0 0 30px')
    )

    chk_pattern_rep_spatial = widgets.Checkbox(
        value=settings.enable_pattern_rep_spatial,
        description='Spatial Distribution',
        indent=False,
        layout=Layout(width='380px', margin='0 0 0 30px')
    )

    chk_pattern_rep_integrity = widgets.Checkbox(
        value=settings.enable_pattern_rep_integrity,
        description='Pattern Integrity',
        indent=False,
        layout=Layout(width='380px', margin='0 0 0 30px')
    )

    chk_pattern_rep_catalog = widgets.Checkbox(
        value=settings.enable_pattern_rep_catalog,
        description='Missing/Extra Patterns Catalog',
        indent=False,
        layout=Layout(width='380px', margin='0 0 0 30px')
    )

    # Main section: Spectrophotometer Simulation
    chk_spectro = widgets.Checkbox(
        value=settings.enable_spectrophotometer,
        description='Spectrophotometer Simulation',
        style={'description_width': 'initial', 'font_weight': 'bold'},
        layout=Layout(width='400px')
    )

    # Spectrophotometer sub-sections
    chk_spectro_config = widgets.Checkbox(
        value=settings.enable_spectro_config,
        description='Instrument Configuration',
        indent=False,
        layout=Layout(width='380px', margin='0 0 0 30px')
    )

    chk_spectro_color_diff_methods = widgets.Checkbox(
        value=settings.enable_spectro_color_diff_methods,
        description='Color Difference Methods',
        indent=False,
        layout=Layout(width='380px', margin='0 0 0 30px')
    )

    chk_spectro_whiteness = widgets.Checkbox(
        value=settings.enable_spectro_whiteness,
        description='Whiteness & Yellowness Indices',
        indent=False,
        layout=Layout(width='380px', margin='0 0 0 30px')
    )

    chk_spectro_metamerism = widgets.Checkbox(
        value=settings.enable_spectro_metamerism,
        description='Metamerism Analysis',
        indent=False,
        layout=Layout(width='380px', margin='0 0 0 30px')
    )

    chk_spectro_spectral_data = widgets.Checkbox(
        value=settings.enable_spectro_spectral_data,
        description='True Spectral Reflectance Analysis',
        indent=False,
        layout=Layout(width='380px', margin='0 0 0 30px')
    )

    chk_spectro_calibration = widgets.Checkbox(
        value=settings.enable_spectro_calibration,
        description='Calibration & Limitations',
        indent=False,
        layout=Layout(width='380px', margin='0 0 0 30px')
    )

    # Store all checkboxes for later access
    color_subsections = [
        chk_color_input_images, chk_color_measurements, chk_color_difference,
        chk_color_statistical, chk_color_spectral_proxy, chk_color_visual_diff,
        chk_color_lab_detailed, chk_color_lab_viz, chk_color_quality_assessment,
        chk_color_scoring, chk_color_recommendations
    ]

    pattern_subsections = [
        chk_pattern_ssim, chk_pattern_symmetry, chk_pattern_edge,
        chk_pattern_repeat, chk_pattern_advanced
    ]

    pattern_rep_subsections = [
        chk_pattern_rep_summary, chk_pattern_rep_count, chk_pattern_rep_blob,
        chk_pattern_rep_keypoint, chk_pattern_rep_autocorr, chk_pattern_rep_spatial,
        chk_pattern_rep_integrity, chk_pattern_rep_catalog
    ]

    spectro_subsections = [
        chk_spectro_config, chk_spectro_color_diff_methods, chk_spectro_whiteness,
        chk_spectro_metamerism, chk_spectro_spectral_data, chk_spectro_calibration
    ]

    # Define handlers for main section checkboxes to control sub-sections
    def on_color_unit_change(change):
        enabled = change['new']
        for chk in color_subsections:
            chk.value = enabled
            chk.disabled = not enabled

    def on_pattern_unit_change(change):
        enabled = change['new']
        for chk in pattern_subsections:
            chk.value = enabled
            chk.disabled = not enabled

    def on_pattern_rep_change(change):
        enabled = change['new']
        for chk in pattern_rep_subsections:
            chk.value = enabled
            chk.disabled = not enabled

    def on_spectro_change(change):
        enabled = change['new']
        for chk in spectro_subsections:
            chk.value = enabled
            chk.disabled = not enabled

    # Attach observers
    chk_color_unit.observe(on_color_unit_change, names='value')
    chk_pattern_unit.observe(on_pattern_unit_change, names='value')
    chk_pattern_repetition.observe(on_pattern_rep_change, names='value')
    chk_spectro.observe(on_spectro_change, names='value')

    # Initial state for sub-sections
    for chk in color_subsections:
        chk.disabled = not chk_color_unit.value
    for chk in pattern_subsections:
        chk.disabled = not chk_pattern_unit.value
    for chk in pattern_rep_subsections:
        chk.disabled = not chk_pattern_repetition.value
    for chk in spectro_subsections:
        chk.disabled = not chk_spectro.value

    # Apply button
    btn_apply = Button(
        description='✅ Apply Sections',
        button_style='success',
        layout=Layout(width='300px', height='40px', margin='20px 0'),
        style={'button_color': '#27AE60', 'font_weight': 'bold'}
    )

    result_output = Output()

    def on_apply_clicked(b):
        # Update settings
        settings.enable_analysis_settings = chk_analysis_settings.value
        settings.enable_color_unit = chk_color_unit.value
        settings.enable_color_input_images = chk_color_input_images.value
        settings.enable_color_measurements = chk_color_measurements.value
        settings.enable_color_difference = chk_color_difference.value
        settings.enable_color_statistical = chk_color_statistical.value
        settings.enable_color_spectral_proxy = chk_color_spectral_proxy.value
        settings.enable_color_visual_diff = chk_color_visual_diff.value
        settings.enable_color_lab_detailed = chk_color_lab_detailed.value
        settings.enable_color_lab_viz = chk_color_lab_viz.value
        settings.enable_color_quality_assessment = chk_color_quality_assessment.value
        settings.enable_color_scoring = chk_color_scoring.value
        settings.enable_color_recommendations = chk_color_recommendations.value

        settings.enable_pattern_unit = chk_pattern_unit.value
        settings.enable_pattern_ssim = chk_pattern_ssim.value
        settings.enable_pattern_symmetry = chk_pattern_symmetry.value
        settings.enable_pattern_edge = chk_pattern_edge.value
        settings.enable_pattern_repeat = chk_pattern_repeat.value
        settings.enable_pattern_advanced = chk_pattern_advanced.value

        settings.enable_pattern_repetition = chk_pattern_repetition.value
        settings.enable_pattern_rep_summary = chk_pattern_rep_summary.value
        settings.enable_pattern_rep_count = chk_pattern_rep_count.value
        settings.enable_pattern_rep_blob = chk_pattern_rep_blob.value
        settings.enable_pattern_rep_keypoint = chk_pattern_rep_keypoint.value
        settings.enable_pattern_rep_autocorr = chk_pattern_rep_autocorr.value
        settings.enable_pattern_rep_spatial = chk_pattern_rep_spatial.value
        settings.enable_pattern_rep_integrity = chk_pattern_rep_integrity.value
        settings.enable_pattern_rep_catalog = chk_pattern_rep_catalog.value

        settings.enable_spectrophotometer = chk_spectro.value
        settings.enable_spectro_config = chk_spectro_config.value
        settings.enable_spectro_color_diff_methods = chk_spectro_color_diff_methods.value
        settings.enable_spectro_whiteness = chk_spectro_whiteness.value
        settings.enable_spectro_metamerism = chk_spectro_metamerism.value
        settings.enable_spectro_spectral_data = chk_spectro_spectral_data.value
        settings.enable_spectro_calibration = chk_spectro_calibration.value

        with result_output:
            result_output.clear_output()
            enabled_count = sum([
                settings.enable_analysis_settings,
                settings.enable_color_unit,
                settings.enable_pattern_unit,
                settings.enable_pattern_repetition,
                settings.enable_spectrophotometer
            ])
            success_html = f"""
            <div style='background: linear-gradient(135deg, #27AE60 0%, #2ECC71 100%);
                        padding: 15px; border-radius: 10px; text-align: center; margin: 10px 0;'>
                <h3 style='color: white; margin: 0;'>✅ Sections Applied!</h3>
                <p style='color: #ecf0f1; margin: 5px 0 0 0;'>
                    {enabled_count} main section(s) enabled. Processing will be optimized accordingly.
                </p>
            </div>
            """
            display(HTML(success_html))

    btn_apply.on_click(on_apply_clicked)

    # Build the UI
    panel = VBox([
        HTMLWidget(value=title_html),
        HTMLWidget(value=info_html),
        HTMLWidget(value=f"<div style='{section_style}'><h3 style='margin-top:0; color:#F39C12; font-size: 16px;'>📊 Main Sections</h3></div>"),
        chk_analysis_settings,
        HTMLWidget(value="<div style='height: 10px;'></div>"),
        chk_color_unit,
        VBox(color_subsections, layout=Layout(margin='5px 0 15px 0')),
        chk_pattern_unit,
        VBox(pattern_subsections, layout=Layout(margin='5px 0 15px 0')),
        chk_pattern_repetition,
        VBox(pattern_rep_subsections, layout=Layout(margin='5px 0 15px 0')),
        chk_spectro,
        VBox(spectro_subsections, layout=Layout(margin='5px 0 15px 0')),
        HBox([btn_apply], layout=Layout(justify_content='center')),
        result_output
    ], layout=Layout(border='2px solid #F39C12', border_radius='10px', padding='10px', margin='20px 0'))

    return panel

def create_settings_summary_table(settings):
    """Create a comprehensive settings summary table"""
    data = [["Parameter", "Value"]]

    # Operator info
    data.append(["Operator", settings.operator_name])

    # Color thresholds
    data.append(["", ""])  # Separator
    data.append([Paragraph("<b>Color Analysis Thresholds</b>", StyleSmall), ""])
    data.append(["ΔE Threshold (PASS)", f"{settings.delta_e_threshold:.2f}"])
    data.append(["ΔE Conditional", f"{settings.delta_e_conditional:.2f}"])
    data.append(["Lab L* Threshold", f"{settings.lab_l_threshold:.2f}"])
    data.append(["Lab a*/b* Threshold", f"{settings.lab_ab_threshold:.2f}"])
    data.append(["Lab Overall Threshold", f"{settings.lab_overall_threshold:.2f}"])

    # Pattern thresholds
    data.append(["", ""])  # Separator
    data.append([Paragraph("<b>Pattern Analysis Thresholds</b>", StyleSmall), ""])
    data.append(["SSIM PASS Threshold", f"{settings.ssim_pass_threshold:.2f}"])
    data.append(["SSIM Conditional Threshold", f"{settings.ssim_conditional_threshold:.2f}"])

    # Scoring parameters
    data.append(["", ""])  # Separator
    data.append([Paragraph("<b>Scoring Parameters</b>", StyleSmall), ""])
    data.append(["Color Score Multiplier", f"{settings.color_score_multiplier:.1f}"])
    data.append(["Uniformity Std Multiplier", f"{settings.uniformity_std_multiplier:.1f}"])
    data.append(["Color Score Minimum", f"{settings.color_score_threshold:.1f}"])
    data.append(["Pattern Score Minimum", f"{settings.pattern_score_threshold:.1f}"])
    data.append(["Overall Score Minimum", f"{settings.overall_score_threshold:.1f}"])

    # Sampling
    data.append(["", ""])  # Separator
    data.append([Paragraph("<b>Sampling Configuration</b>", StyleSmall), ""])
    data.append(["Number of Sample Points", str(settings.num_sample_points)])

    # Region selection
    data.append(["", ""])  # Separator
    data.append([Paragraph("<b>Region of Interest</b>", StyleSmall), ""])
    data.append(["ROI Selection Enabled", "Yes" if settings.use_crop else "No"])
    if settings.use_crop:
        data.append(["ROI Shape", settings.crop_shape.title()])
        data.append(["Center X (px)", str(settings.crop_center_x)])
        data.append(["Center Y (px)", str(settings.crop_center_y)])
        if settings.crop_shape == "circle":
            data.append(["Diameter (px)", str(settings.crop_diameter)])
        else:
            data.append(["Width (px)", str(settings.crop_width)])
            data.append(["Height (px)", str(settings.crop_height)])

    # Spectrophotometer settings
    data.append(["", ""])  # Separator
    data.append([Paragraph("<b>Spectrophotometer Settings</b>", StyleSmall), ""])
    data.append(["Observer Angle", f"{settings.observer_angle}°"])
    data.append(["Geometry Mode", settings.geometry_mode])
    data.append(["ΔE CMC Enabled", "Yes" if settings.use_delta_e_cmc else "No"])
    if settings.use_delta_e_cmc:
        data.append(["CMC l:c Ratio", settings.cmc_l_c_ratio])
    data.append(["Whiteness Min", f"{settings.whiteness_min:.1f}"])
    data.append(["Yellowness Max", f"{settings.yellowness_max:.1f}"])

    # Advanced texture parameters
    data.append(["", ""])  # Separator
    data.append([Paragraph("<b>Advanced Texture Parameters</b>", StyleSmall), ""])
    data.append(["FFT Peaks to Detect", str(settings.fft_num_peaks)])
    data.append(["FFT Notch Filter", "Enabled" if settings.fft_enable_notch else "Disabled"])
    data.append(["Gabor Frequencies", settings.gabor_frequencies_str])
    data.append(["Gabor Orientations", str(settings.gabor_num_orientations)])
    data.append(["GLCM Distances", settings.glcm_distances_str])
    data.append(["GLCM Angles", settings.glcm_angles_str])
    data.append(["LBP Points (P)", str(settings.lbp_points)])
    data.append(["LBP Radius (R)", str(settings.lbp_radius)])
    data.append(["Wavelet Type", settings.wavelet_type])
    data.append(["Wavelet Levels", str(settings.wavelet_levels)])
    data.append(["Min Defect Area (px²)", str(settings.defect_min_area)])
    data.append(["Morph Kernel Size", str(settings.morph_kernel_size)])
    data.append(["Saliency Strength", f"{settings.saliency_strength:.1f}"])

    # Create table with proper wrapping
    table = Table(data, colWidths=[3.2*inch, 3.0*inch], repeatRows=1)

    style_cmds = [
        ("BACKGROUND", (0, 0), (-1, 0), BLUE2),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 11),
        ("ALIGN", (0, 0), (-1, 0), "CENTER"),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
        ("TOPPADDING", (0, 0), (-1, 0), 8),
        ("GRID", (0, 0), (-1, -1), 0.5, NEUTRAL_L),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 1), (-1, -1), 8.5),
        ("ALIGN", (0, 1), (0, -1), "LEFT"),
        ("ALIGN", (1, 1), (1, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 1), (-1, -1), 6),
        ("RIGHTPADDING", (0, 1), (-1, -1), 6),
        ("TOPPADDING", (0, 1), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 1), (-1, -1), 4),
    ]

    # Alternate row colors
    for i in range(2, len(data), 2):
        if data[i][0] != "":  # Skip separator rows
            style_cmds.append(("BACKGROUND", (0, i), (-1, i), colors.whitesmoke))

    # Make section headers stand out
    for i, row in enumerate(data):
        if i > 0 and isinstance(row[0], Paragraph):
            style_cmds.append(("BACKGROUND", (0, i), (-1, i), NEUTRAL_L))
            style_cmds.append(("SPAN", (0, i), (-1, i)))
            style_cmds.append(("FONTNAME", (0, i), (-1, i), "Helvetica-Bold"))
            style_cmds.append(("FONTSIZE", (0, i), (-1, i), 9))

    table.setStyle(TableStyle(style_cmds))
    return table

def first_page_header(canvas_, doc):
    # cover page: frame and subtle header line, no page number
    canvas_.saveState()
    width, height = PAGE_SIZE

    # Draw white rectangular frame with 3mm margins on all sides (same as other pages)
    canvas_.setStrokeColor(colors.HexColor("#E0E0E0"))  # Light gray frame
    canvas_.setLineWidth(0.8)
    frame_x1 = FRAME_MARGIN
    frame_y1 = FRAME_MARGIN
    frame_x2 = width - FRAME_MARGIN
    frame_y2 = height - FRAME_MARGIN
    canvas_.rect(frame_x1, frame_y1, frame_x2 - frame_x1, frame_y2 - frame_y1, stroke=1, fill=0)

    # Header line
    y = height - 40
    canvas_.setStrokeColor(NEUTRAL_L)
    canvas_.setLineWidth(0.6)
    canvas_.line(MARGIN_L, y, width - MARGIN_R, y)
    canvas_.restoreState()

# ----------------------------
# 6b) Interactive UI Components
# ----------------------------
def create_advanced_settings_ui(settings, ref_img, test_img):
    """Create interactive advanced settings panel with professional UI"""

    # Style definitions
    title_style = """
        <div style='background: linear-gradient(135deg, #2980B9 0%, #3498DB 100%);
                    padding: 20px; border-radius: 10px 10px 0 0; margin-bottom: 0;'>
            <h2 style='color: white; margin: 0; font-family: Arial, sans-serif;'>
                ⚙️ Advanced Quality Control Settings
            </h2>
        </div>
    """

    section_style = "background: #f8f9fa; padding: 15px; margin: 10px 0; border-radius: 8px; border-left: 4px solid #2980B9;"

    # Operator info
    operator_input = Text(
        value=settings.operator_name,
        description='Operator:',
        style={'description_width': '180px'},
        layout=Layout(width='400px')
    )

    # Color thresholds
    delta_e_thresh = FloatText(value=settings.delta_e_threshold, description='ΔE Threshold (PASS):',
                                style={'description_width': '180px'}, layout=Layout(width='300px'))
    delta_e_cond = FloatText(value=settings.delta_e_conditional, description='ΔE Conditional:',
                             style={'description_width': '180px'}, layout=Layout(width='300px'))
    lab_l_thresh = FloatText(value=settings.lab_l_threshold, description='Lab L* Threshold:',
                             style={'description_width': '180px'}, layout=Layout(width='300px'))
    lab_ab_thresh = FloatText(value=settings.lab_ab_threshold, description='Lab a*/b* Threshold:',
                              style={'description_width': '180px'}, layout=Layout(width='300px'))
    lab_overall_thresh = FloatText(value=settings.lab_overall_threshold, description='Lab Overall Threshold:',
                                   style={'description_width': '180px'}, layout=Layout(width='300px'))

    # Pattern thresholds
    ssim_pass = FloatText(value=settings.ssim_pass_threshold, description='SSIM PASS (>):',
                         style={'description_width': '180px'}, layout=Layout(width='300px'))
    ssim_cond = FloatText(value=settings.ssim_conditional_threshold, description='SSIM Conditional (>):',
                         style={'description_width': '180px'}, layout=Layout(width='300px'))

    # Scoring parameters
    color_multiplier = FloatText(value=settings.color_score_multiplier, description='Color Score Multiplier:',
                                 style={'description_width': '180px'}, layout=Layout(width='300px'))
    uniformity_multiplier = FloatText(value=settings.uniformity_std_multiplier, description='Uniformity Multiplier:',
                                      style={'description_width': '180px'}, layout=Layout(width='300px'))

    # Quality thresholds
    color_score_thresh = FloatText(value=settings.color_score_threshold, description='Color Score Min:',
                                   style={'description_width': '180px'}, layout=Layout(width='300px'))
    pattern_score_thresh = FloatText(value=settings.pattern_score_threshold, description='Pattern Score Min:',
                                     style={'description_width': '180px'}, layout=Layout(width='300px'))
    overall_score_thresh = FloatText(value=settings.overall_score_threshold, description='Overall Score Min:',
                                     style={'description_width': '180px'}, layout=Layout(width='300px'))

    # Number of sample points
    num_points = IntText(value=settings.num_sample_points, description='Sample Points:',
                        style={'description_width': '180px'}, layout=Layout(width='300px'))

    # Build the panel
    title_widget = HTMLWidget(value=title_style)

    operator_section = VBox([
        HTMLWidget(value=f"<div style='{section_style}'><h3 style='margin-top:0; color:#2980B9;'>👤 Operator Information</h3></div>"),
        operator_input
    ])

    color_section = VBox([
        HTMLWidget(value=f"<div style='{section_style}'><h3 style='margin-top:0; color:#2980B9;'>🎨 Color Analysis Thresholds</h3></div>"),
        delta_e_thresh, delta_e_cond, lab_l_thresh, lab_ab_thresh, lab_overall_thresh
    ])

    pattern_section = VBox([
        HTMLWidget(value=f"<div style='{section_style}'><h3 style='margin-top:0; color:#2980B9;'>📐 Pattern Analysis Thresholds</h3></div>"),
        ssim_pass, ssim_cond
    ])

    scoring_section = VBox([
        HTMLWidget(value=f"<div style='{section_style}'><h3 style='margin-top:0; color:#2980B9;'>📊 Scoring Parameters</h3></div>"),
        color_multiplier, uniformity_multiplier, color_score_thresh, pattern_score_thresh, overall_score_thresh
    ])

    sampling_section = VBox([
        HTMLWidget(value=f"<div style='{section_style}'><h3 style='margin-top:0; color:#2980B9;'>📍 Sampling Configuration</h3></div>"),
        num_points
    ])

    # ===== ADVANCED TEXTURE PARAMETERS =====
    texture_style = "background: #e8f5e9; padding: 15px; margin: 10px 0; border-radius: 8px; border-left: 4px solid #27AE60;"

    # FFT parameters
    fft_num_peaks_widget = IntText(value=settings.fft_num_peaks, description='FFT Peaks to Detect:',
                                    style={'description_width': '180px'}, layout=Layout(width='300px'))
    fft_notch_widget = widgets.Checkbox(value=settings.fft_enable_notch, description='Enable FFT Notch Filter',
                                         style={'description_width': 'initial'}, layout=Layout(width='300px'))

    # Gabor parameters
    gabor_freq_widget = Text(value=settings.gabor_frequencies_str, description='Gabor Frequencies:',
                             style={'description_width': '180px'}, layout=Layout(width='400px'),
                             placeholder='e.g., 0.1, 0.2, 0.3')
    gabor_num_orient_widget = IntText(value=settings.gabor_num_orientations, description='Gabor Orientations:',
                                       style={'description_width': '180px'}, layout=Layout(width='300px'))

    # GLCM parameters (newly editable)
    glcm_distances_widget = Text(value=settings.glcm_distances_str, description='GLCM Distances (px):',
                                  style={'description_width': '180px'}, layout=Layout(width='400px'),
                                  placeholder='e.g., 1, 3, 5')
    glcm_angles_widget = Text(value=settings.glcm_angles_str, description='GLCM Angles (degrees):',
                               style={'description_width': '180px'}, layout=Layout(width='400px'),
                               placeholder='e.g., 0, 45, 90, 135')

    # LBP parameters
    lbp_points_widget = IntText(value=settings.lbp_points, description='LBP Points (P):',
                                 style={'description_width': '180px'}, layout=Layout(width='300px'))
    lbp_radius_widget = IntText(value=settings.lbp_radius, description='LBP Radius (R):',
                                 style={'description_width': '180px'}, layout=Layout(width='300px'))

    # Wavelet parameters (newly editable type)
    wavelet_type_widget = widgets.Dropdown(options=['db4', 'db2', 'db8', 'haar', 'sym2', 'coif1'],
                                            value=settings.wavelet_type,
                                            description='Wavelet Family:',
                                            style={'description_width': '180px'}, layout=Layout(width='300px'))
    wavelet_levels_widget = IntText(value=settings.wavelet_levels, description='Wavelet Levels:',
                                     style={'description_width': '180px'}, layout=Layout(width='300px'))

    # Defect detection
    defect_min_area_widget = IntText(value=settings.defect_min_area, description='Min Defect Area (px²):',
                                      style={'description_width': '180px'}, layout=Layout(width='300px'))
    morph_kernel_widget = IntText(value=settings.morph_kernel_size, description='Morph Kernel Size:',
                                   style={'description_width': '180px'}, layout=Layout(width='300px'))
    saliency_strength_widget = FloatText(value=settings.saliency_strength, description='Saliency Strength:',
                                          style={'description_width': '180px'}, layout=Layout(width='300px'))

    texture_section = VBox([
        HTMLWidget(value=f"<div style='{texture_style}'><h3 style='margin-top:0; color:#27AE60;'>🔬 Advanced Texture Parameters</h3></div>"),
        HTMLWidget(value="<p style='margin: 10px 0; color: #555;'><b>FFT Analysis:</b></p>"),
        fft_num_peaks_widget, fft_notch_widget,
        HTMLWidget(value="<p style='margin: 10px 0; color: #555;'><b>Gabor Filter Bank:</b></p>"),
        gabor_freq_widget, gabor_num_orient_widget,
        HTMLWidget(value="<p style='margin: 10px 0; color: #555;'><b>GLCM Parameters:</b></p>"),
        glcm_distances_widget, glcm_angles_widget,
        HTMLWidget(value="<p style='margin: 10px 0; color: #555;'><b>LBP Parameters:</b></p>"),
        HBox([lbp_points_widget, lbp_radius_widget]),
        HTMLWidget(value="<p style='margin: 10px 0; color: #555;'><b>Wavelet Analysis:</b></p>"),
        wavelet_type_widget, wavelet_levels_widget,
        HTMLWidget(value="<p style='margin: 10px 0; color: #555;'><b>Defect Detection:</b></p>"),
        defect_min_area_widget, morph_kernel_widget, saliency_strength_widget
    ])

    # ===== ENHANCED COLOR PARAMETERS =====
    color_adv_style = "background: #fff3e0; padding: 15px; margin: 10px 0; border-radius: 8px; border-left: 4px solid #F39C12;"

    # Observer and geometry
    observer_widget = widgets.Dropdown(options=['2', '10'], value=settings.observer_angle,
                                        description='Observer Angle:',
                                        style={'description_width': '180px'}, layout=Layout(width='300px'))
    geometry_widget = widgets.Dropdown(options=['d/8 SCI', 'd/8 SCE', '45/0'], value=settings.geometry_mode,
                                        description='Geometry Mode:',
                                        style={'description_width': '180px'}, layout=Layout(width='300px'))

    # CMC parameters
    cmc_enable_widget = widgets.Checkbox(value=settings.use_delta_e_cmc, description='Enable ΔE CMC',
                                          style={'description_width': 'initial'}, layout=Layout(width='200px'))
    cmc_ratio_widget = widgets.Dropdown(options=['2:1', '1:1'], value=settings.cmc_l_c_ratio,
                                         description='CMC l:c Ratio:',
                                         style={'description_width': '180px'}, layout=Layout(width='300px'))

    # Whiteness/Yellowness thresholds
    whiteness_min_widget = FloatText(value=settings.whiteness_min, description='Min Whiteness:',
                                      style={'description_width': '180px'}, layout=Layout(width='300px'))
    yellowness_max_widget = FloatText(value=settings.yellowness_max, description='Max Yellowness:',
                                       style={'description_width': '180px'}, layout=Layout(width='300px'))

    # Metamerism illuminants (multi-select)
    illuminant_options = ['D65', 'D50', 'TL84', 'A', 'F2', 'CWF', 'F7', 'F11']
    illuminant_select = widgets.SelectMultiple(
        options=illuminant_options,
        value=settings.metamerism_illuminants,
        description='Illuminants:',
        style={'description_width': '180px'},
        layout=Layout(width='400px', height='120px')
    )

    # Spectral CSV upload
    spectral_enable_widget = widgets.Checkbox(value=settings.spectral_enable,
                                               description='Use Spectral Data (overrides RGB)',
                                               style={'description_width': 'initial'},
                                               layout=Layout(width='350px'))

    spectral_ref_upload = widgets.FileUpload(accept='.csv', multiple=False, description='Ref CSV:',
                                              layout=Layout(width='350px'))
    spectral_sample_upload = widgets.FileUpload(accept='.csv', multiple=False, description='Sample CSV:',
                                                 layout=Layout(width='350px'))

    color_enhanced_section = VBox([
        HTMLWidget(value=f"<div style='{color_adv_style}'><h3 style='margin-top:0; color:#F39C12;'>🌈 Spectrophotometer Settings</h3></div>"),
        HTMLWidget(value="<p style='margin: 10px 0; color: #555;'><b>Instrument Configuration:</b></p>"),
        observer_widget, geometry_widget,
        HTMLWidget(value="<p style='margin: 10px 0; color: #555;'><b>Color Difference:</b></p>"),
        cmc_enable_widget, cmc_ratio_widget,
        HTMLWidget(value="<p style='margin: 10px 0; color: #555;'><b>Whiteness & Yellowness:</b></p>"),
        whiteness_min_widget, yellowness_max_widget,
        HTMLWidget(value="<p style='margin: 10px 0; color: #555;'><b>Metamerism Illuminants:</b></p>"),
        illuminant_select,
        HTMLWidget(value="<p style='margin: 10px 0; color: #555;'><b>Spectral Data (Optional):</b></p>"),
        spectral_enable_widget,
        HTMLWidget(value="<p style='margin: 5px 0; color: #777; font-size: 11px;'><i>CSV format: wavelength (nm), reflectance (%)</i></p>"),
        spectral_ref_upload,
        spectral_sample_upload
    ])

    # Circle selector section
    circle_section = create_circle_selector_ui(settings, ref_img, test_img)

    settings_panel = VBox([
        title_widget,
        operator_section,
        color_section,
        pattern_section,
        scoring_section,
        sampling_section,
        texture_section,
        color_enhanced_section,
        circle_section
    ], layout=Layout(border='2px solid #2980B9', border_radius='10px', margin='20px 0'))

    # Return both the panel and the widget references for value extraction
    widgets_dict = {
        'operator_name': operator_input,
        'delta_e_threshold': delta_e_thresh,
        'delta_e_conditional': delta_e_cond,
        'lab_l_threshold': lab_l_thresh,
        'lab_ab_threshold': lab_ab_thresh,
        'lab_overall_threshold': lab_overall_thresh,
        'ssim_pass_threshold': ssim_pass,
        'ssim_conditional_threshold': ssim_cond,
        'color_score_multiplier': color_multiplier,
        'uniformity_std_multiplier': uniformity_multiplier,
        'color_score_threshold': color_score_thresh,
        'pattern_score_threshold': pattern_score_thresh,
        'overall_score_threshold': overall_score_thresh,
        'num_sample_points': num_points,
        # Texture parameters (newly editable)
        'fft_num_peaks': fft_num_peaks_widget,
        'fft_enable_notch': fft_notch_widget,
        'gabor_frequencies_str': gabor_freq_widget,
        'gabor_num_orientations': gabor_num_orient_widget,
        'glcm_distances_str': glcm_distances_widget,
        'glcm_angles_str': glcm_angles_widget,
        'lbp_points': lbp_points_widget,
        'lbp_radius': lbp_radius_widget,
        'wavelet_type': wavelet_type_widget,
        'wavelet_levels': wavelet_levels_widget,
        'defect_min_area': defect_min_area_widget,
        'morph_kernel_size': morph_kernel_widget,
        'saliency_strength': saliency_strength_widget,
        # Color enhanced parameters
        'observer_angle': observer_widget,
        'geometry_mode': geometry_widget,
        'use_delta_e_cmc': cmc_enable_widget,
        'cmc_l_c_ratio': cmc_ratio_widget,
        'whiteness_min': whiteness_min_widget,
        'yellowness_max': yellowness_max_widget,
        'metamerism_illuminants': illuminant_select,
        'spectral_enable': spectral_enable_widget,
        'spectral_ref_upload': spectral_ref_upload,
        'spectral_sample_upload': spectral_sample_upload
    }

    return settings_panel, widgets_dict

def create_circle_selector_ui(settings, ref_img, test_img):
    """Create interactive region selector for region of interest (circle or rectangle)"""

    h, w = ref_img.shape[:2]

    # Initialize crop parameters
    crop_enable = widgets.Checkbox(
        value=settings.use_crop,
        description='Enable ROI Selection',
        style={'description_width': 'initial'},
        layout=Layout(width='300px')
    )

    crop_shape = widgets.Dropdown(
        options=[('Circle', 'circle'), ('Rectangle', 'rectangle')],
        value=settings.crop_shape,
        description='Crop Shape:',
        style={'description_width': '180px'},
        layout=Layout(width='300px')
    )

    crop_x = IntText(
        value=w // 2 if settings.crop_center_x == 0 else settings.crop_center_x,
        description='Center X (px):',
        style={'description_width': '180px'},
        layout=Layout(width='300px')
    )

    crop_y = IntText(
        value=h // 2 if settings.crop_center_y == 0 else settings.crop_center_y,
        description='Center Y (px):',
        style={'description_width': '180px'},
        layout=Layout(width='300px')
    )

    crop_diameter = IntText(
        value=settings.crop_diameter,
        description='Diameter (px):',
        style={'description_width': '180px'},
        layout=Layout(width='300px')
    )

    crop_width = IntText(
        value=settings.crop_width,
        description='Width (px):',
        style={'description_width': '180px'},
        layout=Layout(width='300px')
    )

    crop_height = IntText(
        value=settings.crop_height,
        description='Height (px):',
        style={'description_width': '180px'},
        layout=Layout(width='300px')
    )

    # Preview output
    preview_output = Output()

    def update_shape_visibility(change=None):
        """Update visibility of shape-specific controls"""
        if crop_shape.value == 'circle':
            crop_diameter.layout.display = 'block'
            crop_width.layout.display = 'none'
            crop_height.layout.display = 'none'
        else:  # rectangle
            crop_diameter.layout.display = 'none'
            crop_width.layout.display = 'block'
            crop_height.layout.display = 'block'
        update_preview()

    def update_preview(change=None):
        """Update image preview with crop region overlay"""
        with preview_output:
            preview_output.clear_output(wait=True)

            if not crop_enable.value:
                print("ℹ️ ROI selection is disabled. Full images will be processed.")
                return

            cx, cy = crop_x.value, crop_y.value
            shape = crop_shape.value

            # Resize for display
            display_width = 400
            scale = display_width / w
            display_h = int(h * scale)

            ref_display = cv2.resize(ref_img, (display_width, display_h))
            test_display = cv2.resize(test_img, (display_width, display_h))

            # Scale crop parameters
            cx_scaled = int(cx * scale)
            cy_scaled = int(cy * scale)

            # Draw crop region based on shape
            if shape == 'circle':
                diam_scaled = int(crop_diameter.value * scale)
                ref_with_region = draw_circle_on_image(ref_display, cx_scaled, cy_scaled, diam_scaled)
                test_with_region = draw_circle_on_image(test_display, cx_scaled, cy_scaled, diam_scaled)
                shape_info = f"Circle (Ø{crop_diameter.value}px)"
            else:  # rectangle
                width_scaled = int(crop_width.value * scale)
                height_scaled = int(crop_height.value * scale)
                ref_with_region = draw_rectangle_on_image(ref_display, cx_scaled, cy_scaled, width_scaled, height_scaled)
                test_with_region = draw_rectangle_on_image(test_display, cx_scaled, cy_scaled, width_scaled, height_scaled)
                shape_info = f"Rectangle ({crop_width.value}×{crop_height.value}px)"

            # Combine side by side
            combined = np.hstack([ref_with_region, test_with_region])

            # Display
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.imshow(combined)
            ax.set_title(f'Preview: Reference (Left) | Sample (Right) | {shape_info} @ ({cx}, {cy})',
                        fontsize=12, fontweight='bold', color='#2980B9')
            ax.axis('off')
            ax.text(display_width // 2, -20, 'REFERENCE', ha='center', fontsize=10,
                   fontweight='bold', color='#27AE60')
            ax.text(display_width + display_width // 2, -20, 'SAMPLE', ha='center', fontsize=10,
                   fontweight='bold', color='#E74C3C')
            plt.tight_layout()
            plt.show()

    # Attach observers
    crop_enable.observe(update_preview, names='value')
    crop_shape.observe(update_shape_visibility, names='value')
    crop_x.observe(update_preview, names='value')
    crop_y.observe(update_preview, names='value')
    crop_diameter.observe(update_preview, names='value')
    crop_width.observe(update_preview, names='value')
    crop_height.observe(update_preview, names='value')

    # Initial display setup
    update_shape_visibility()

    section_style = "background: #f8f9fa; padding: 15px; margin: 10px 0; border-radius: 8px; border-left: 4px solid #27AE60;"

    crop_section = VBox([
        HTMLWidget(value=f"<div style='{section_style}'><h3 style='margin-top:0; color:#27AE60;'>🎯 Region of Interest Selection</h3></div>"),
        crop_enable,
        HTMLWidget(value="<p style='margin: 10px 0; color: #555;'><i>Select a specific region to analyze. Choose between circular or rectangular shapes. The same region will be applied to both images.</i></p>"),
        crop_shape,
        HBox([crop_x, crop_y]),
        crop_diameter,
        HBox([crop_width, crop_height]),
        HTMLWidget(value="<p style='margin: 10px 0; color: #777; font-size: 11px;'><i>💡 Tip: Image dimensions - Width: {}px, Height: {}px</i></p>".format(w, h)),
        preview_output
    ])

    # Store widget references in settings for later extraction
    settings._crop_widgets = {
        'use_crop': crop_enable,
        'crop_shape': crop_shape,
        'crop_center_x': crop_x,
        'crop_center_y': crop_y,
        'crop_diameter': crop_diameter,
        'crop_width': crop_width,
        'crop_height': crop_height
    }

    return crop_section

def extract_settings_from_widgets(settings, widgets_dict):
    """Extract values from widgets and update settings object"""
    settings.operator_name = widgets_dict['operator_name'].value
    settings.delta_e_threshold = widgets_dict['delta_e_threshold'].value
    settings.delta_e_conditional = widgets_dict['delta_e_conditional'].value
    settings.lab_l_threshold = widgets_dict['lab_l_threshold'].value
    settings.lab_ab_threshold = widgets_dict['lab_ab_threshold'].value
    settings.lab_overall_threshold = widgets_dict['lab_overall_threshold'].value
    settings.ssim_pass_threshold = widgets_dict['ssim_pass_threshold'].value
    settings.ssim_conditional_threshold = widgets_dict['ssim_conditional_threshold'].value
    settings.color_score_multiplier = widgets_dict['color_score_multiplier'].value
    settings.uniformity_std_multiplier = widgets_dict['uniformity_std_multiplier'].value
    settings.color_score_threshold = widgets_dict['color_score_threshold'].value
    settings.pattern_score_threshold = widgets_dict['pattern_score_threshold'].value
    settings.overall_score_threshold = widgets_dict['overall_score_threshold'].value
    settings.num_sample_points = widgets_dict['num_sample_points'].value

    # Extract texture parameters
    settings.fft_num_peaks = widgets_dict['fft_num_peaks'].value
    settings.fft_enable_notch = widgets_dict['fft_enable_notch'].value

    # Parse Gabor frequencies from text
    settings.gabor_frequencies_str = widgets_dict['gabor_frequencies_str'].value
    try:
        settings.gabor_frequencies = [float(x.strip()) for x in settings.gabor_frequencies_str.split(',')]
    except:
        settings.gabor_frequencies = [0.1, 0.2, 0.3]  # Default on error

    settings.gabor_num_orientations = widgets_dict['gabor_num_orientations'].value

    # Parse GLCM distances and angles from text
    settings.glcm_distances_str = widgets_dict['glcm_distances_str'].value
    settings.glcm_angles_str = widgets_dict['glcm_angles_str'].value
    try:
        settings.glcm_distances = [int(x.strip()) for x in settings.glcm_distances_str.split(',')]
    except:
        settings.glcm_distances = [1, 3, 5]  # Default on error
    try:
        settings.glcm_angles = [int(x.strip()) for x in settings.glcm_angles_str.split(',')]
    except:
        settings.glcm_angles = [0, 45, 90, 135]  # Default on error

    settings.lbp_points = widgets_dict['lbp_points'].value
    settings.lbp_radius = widgets_dict['lbp_radius'].value
    settings.wavelet_type = widgets_dict['wavelet_type'].value
    settings.wavelet_levels = widgets_dict['wavelet_levels'].value
    settings.defect_min_area = widgets_dict['defect_min_area'].value
    settings.morph_kernel_size = widgets_dict['morph_kernel_size'].value
    settings.saliency_strength = widgets_dict['saliency_strength'].value

    # Extract color enhanced parameters
    settings.observer_angle = widgets_dict['observer_angle'].value
    settings.geometry_mode = widgets_dict['geometry_mode'].value
    settings.use_delta_e_cmc = widgets_dict['use_delta_e_cmc'].value
    settings.cmc_l_c_ratio = widgets_dict['cmc_l_c_ratio'].value
    settings.whiteness_min = widgets_dict['whiteness_min'].value
    settings.yellowness_max = widgets_dict['yellowness_max'].value
    settings.metamerism_illuminants = list(widgets_dict['metamerism_illuminants'].value)

    # Extract spectral data
    settings.spectral_enable = widgets_dict['spectral_enable'].value

    # Process uploaded CSV files if present
    ref_upload = widgets_dict['spectral_ref_upload']
    sample_upload = widgets_dict['spectral_sample_upload']

    if ref_upload.value:
        # Save uploaded file temporarily
        for filename, file_info in ref_upload.value.items():
            temp_path = f"/tmp/spectral_ref_{filename}"
            with open(temp_path, 'wb') as f:
                f.write(file_info['content'])
            wl, ref = parse_spectral_csv(temp_path)
            if wl is not None:
                settings.spectral_ref_wavelengths = wl
                settings.spectral_ref_reflectance = ref
                settings.spectral_ref_path = filename
                print(f"✅ Loaded reference spectral data: {filename}")

    if sample_upload.value:
        for filename, file_info in sample_upload.value.items():
            temp_path = f"/tmp/spectral_sample_{filename}"
            with open(temp_path, 'wb') as f:
                f.write(file_info['content'])
            wl, ref = parse_spectral_csv(temp_path)
            if wl is not None:
                settings.spectral_sample_wavelengths = wl
                settings.spectral_sample_reflectance = ref
                settings.spectral_sample_path = filename
                print(f"✅ Loaded sample spectral data: {filename}")

    # Extract crop settings if available
    if hasattr(settings, '_crop_widgets'):
        settings.use_crop = settings._crop_widgets['use_crop'].value
        settings.crop_shape = settings._crop_widgets['crop_shape'].value
        settings.crop_center_x = settings._crop_widgets['crop_center_x'].value
        settings.crop_center_y = settings._crop_widgets['crop_center_y'].value
        settings.crop_diameter = settings._crop_widgets['crop_diameter'].value
        settings.crop_width = settings._crop_widgets['crop_width'].value
        settings.crop_height = settings._crop_widgets['crop_height'].value

    return settings

# ----------------------------
# 6b) Export Functions
# ----------------------------
def export_analysis_to_csv(df_samples, color_metrics, pattern_metrics, output_path):
    """
    Export analysis results to CSV format.

    Args:
        df_samples: DataFrame with sample point measurements
        color_metrics: Dictionary of color analysis metrics
        pattern_metrics: Dictionary of pattern analysis metrics
        output_path: Path to save CSV file
    """
    try:
        # Save sample data
        sample_csv = output_path.replace('.csv', '_samples.csv')
        df_samples.to_csv(sample_csv, index=False)
        logger.info(f"Exported sample data to {sample_csv}")

        # Save summary metrics
        summary_data = {
            'Metric': [],
            'Value': [],
            'Unit': []
        }

        # Add color metrics
        for key, value in color_metrics.items():
            summary_data['Metric'].append(key)
            summary_data['Value'].append(value)
            summary_data['Unit'].append('')

        # Add pattern metrics
        for key, value in pattern_metrics.items():
            summary_data['Metric'].append(key)
            summary_data['Value'].append(value)
            summary_data['Unit'].append('')

        summary_df = pd.DataFrame(summary_data)
        summary_csv = output_path.replace('.csv', '_summary.csv')
        summary_df.to_csv(summary_csv, index=False)
        logger.info(f"Exported summary metrics to {summary_csv}")

        return sample_csv, summary_csv
    except Exception as e:
        logger.error(f"Failed to export CSV: {str(e)}")
        return None, None

# ----------------------------
# 7) Main pipeline
# ----------------------------
def run_pipeline_and_build_pdf(ref_path, test_path, ref, test, settings):
    """
    Main analysis pipeline with custom settings.

    Args:
        ref_path: Path to reference image
        test_path: Path to test image
        ref: Reference image array
        test: Test image array
        settings: QCSettings object with analysis parameters

    Returns:
        str: Path to generated PDF report

    Raises:
        RuntimeError: If analysis or PDF generation fails
    """
    try:
        logger.info(f"Starting analysis pipeline for {os.path.basename(ref_path)} vs {os.path.basename(test_path)}")

        # Apply crop if enabled (circle or rectangle)
        if settings.use_crop:
            logger.info(f"Applying {settings.crop_shape} crop to images")
            ref = apply_crop(ref, settings)
            test = apply_crop(test, settings)
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {str(e)}")
        raise RuntimeError(f"Pipeline initialization failed: {str(e)}")

    H, W = ref.shape[:2]
    small_w = 640
    scale = small_w / W
    small_h = max(1, int(H * scale))
    ref_small = cv2.resize(ref, (small_w, small_h), interpolation=cv2.INTER_AREA)
    test_small = cv2.resize(test, (small_w, small_h), interpolation=cv2.INTER_AREA)

    # ----- Color analysis under D65 (source) then adapted to chosen illuminants for metamerism
    src_wp = WHITE_POINTS["D65"]
    xyz_ref = srgb_to_xyz(ref_small)
    xyz_test = srgb_to_xyz(test_small)

    def mean_de_under(ill_name):
        dst_wp = WHITE_POINTS[ill_name]
        r = adapt_white_xyz(xyz_ref, src_wp, dst_wp)
        t = adapt_white_xyz(xyz_test, src_wp, dst_wp)
        lab_r = xyz_to_lab(r, dst_wp)
        lab_t = xyz_to_lab(t, dst_wp)
        return (lab_r, lab_t,
                float(np.mean(deltaE2000(lab_r, lab_t))),
                float(np.mean(deltaE76(lab_r, lab_t))),
                r, t)

    lab_ref_D65, lab_test_D65, mean_de00_D65, mean_de76_D65, xyz_ref_D65, xyz_test_D65 = mean_de_under("D65")
    de76_map = deltaE76(lab_ref_D65, lab_test_D65)
    de94_map = deltaE94(lab_ref_D65, lab_test_D65)
    de00_map = deltaE2000(lab_ref_D65, lab_test_D65)

    mean76 = float(np.mean(de76_map)); std76 = float(np.std(de76_map))
    min76 = float(np.min(de76_map)); max76 = float(np.max(de76_map))
    # Uniformity index: higher std deviation = lower uniformity
    uni_idx = max(0.0, 100.0 - std76 * settings.uniformity_std_multiplier)
    # Determine status using settings thresholds
    status_color = determine_status(mean76, settings.delta_e_threshold, settings.delta_e_conditional, lower_is_better=True)

    # Metamerism across illuminants
    _, _, mean_de00_TL84, _, _, _ = mean_de_under("TL84")
    _, _, mean_de00_A,    _, _, _ = mean_de_under("A")
    metamerism_index = float(np.std([mean_de00_D65, mean_de00_TL84, mean_de00_A]) * 10)

    # Region samples (use settings)
    pts = grid_points(small_h, small_w, settings.num_sample_points)
    rows = []
    for i,(y,x) in enumerate(pts, start=1):
        rR, rG, rB = ref_small[y,x].tolist()
        tR, tG, tB = test_small[y,x].tolist()
        xyz_r = srgb_to_xyz(np.array([[rR, rG, rB]], float))[0]
        xyz_t = srgb_to_xyz(np.array([[tR, tG, tB]], float))[0]
        lab_r = xyz_to_lab(xyz_r[None,:], src_wp)[0]
        lab_t = xyz_to_lab(xyz_t[None,:], src_wp)[0]
        cmyk_r = (rgb_to_cmyk(np.array([[rR, rG, rB]]))[0]*100)
        cmyk_t = (rgb_to_cmyk(np.array([[tR, tG, tB]]))[0]*100)
        d76 = float(deltaE76(lab_r[None,:], lab_t[None,:])[0])
        d94 = float(deltaE94(lab_r[None,:], lab_t[None,:])[0])
        d00 = float(deltaE2000(lab_r[None,:], lab_t[None,:])[0])
        rows.append({
            "Region": i, "x": x, "y": y,
            "Ref R": rR, "Ref G": rG, "Ref B": rB,
            "Test R": tR, "Test G": tG, "Test B": tB,
            "Ref L*": lab_r[0], "Ref a*": lab_r[1], "Ref b*": lab_r[2],
            "Test L*": lab_t[0], "Test a*": lab_t[1], "Test b*": lab_t[2],
            "Ref C%": cmyk_r[0], "Ref M%": cmyk_r[1], "Ref Y%": cmyk_r[2], "Ref K%": cmyk_r[3],
            "Test C%": cmyk_t[0], "Test M%": cmyk_t[1], "Test Y%": cmyk_t[2], "Test K%": cmyk_t[3],
            "Ref X": xyz_r[0], "Ref Y": xyz_r[1], "Ref Z": xyz_r[2],
            "Test X": xyz_t[0], "Test Y": xyz_t[1], "Test Z": xyz_t[2],
            "ΔE76": d76, "ΔE94": d94, "ΔE2000": d00
        })
    df_samples = pd.DataFrame(rows)

    # Pattern analysis
    gray_ref = rgb2gray(ref_small)
    gray_test = rgb2gray(test_small)
    ssim_score = float(ssim(gray_ref, gray_test, data_range=1.0))
    sym_ref = symmetry_score(gray_ref)
    sym_test = symmetry_score(gray_test)
    symmetry = (sym_ref + sym_test)/2
    px, py = repeat_period_estimate(gray_test)
    edge_def = edge_definition(gray_test)
    reg_accuracy = min(100.0, ssim_score * 100.0)

    diff = cv2.absdiff((gray_ref*255).astype(np.uint8),(gray_test*255).astype(np.uint8))
    thr = cv2.threshold(diff, 0, 255, cv2.THRESH_OTSU)[1]
    defect_ratio = float(np.sum(thr>0)/thr.size)
    defect_density = defect_ratio * 10_000  # heuristic scale

    # ============ ADVANCED TEXTURE ANALYSIS ============
    logger.info("Running advanced texture analysis...")

    # FFT Analysis
    fft_ref = analyze_fft(gray_ref, num_peaks=settings.fft_num_peaks, enable_notch=settings.fft_enable_notch)
    fft_test = analyze_fft(gray_test, num_peaks=settings.fft_num_peaks, enable_notch=settings.fft_enable_notch)

    # Gabor Filter Bank
    gabor_ref = analyze_gabor(gray_ref, frequencies=settings.gabor_frequencies, num_orientations=settings.gabor_num_orientations)
    gabor_test = analyze_gabor(gray_test, frequencies=settings.gabor_frequencies, num_orientations=settings.gabor_num_orientations)

    # GLCM Features
    glcm_ref = analyze_glcm(gray_ref, distances=settings.glcm_distances, angles=settings.glcm_angles)
    glcm_test = analyze_glcm(gray_test, distances=settings.glcm_distances, angles=settings.glcm_angles)

    # LBP
    lbp_ref = analyze_lbp(gray_ref, P=settings.lbp_points, R=settings.lbp_radius)
    lbp_test = analyze_lbp(gray_test, P=settings.lbp_points, R=settings.lbp_radius)
    lbp_chi2 = lbp_chi2_distance(lbp_ref['histogram'], lbp_test['histogram'])
    lbp_bhatt = lbp_bhattacharyya_distance(lbp_ref['histogram'], lbp_test['histogram'])

    # Wavelet Analysis
    wavelet_ref = analyze_wavelet(gray_ref, wavelet=settings.wavelet_type, levels=settings.wavelet_levels)
    wavelet_test = analyze_wavelet(gray_test, wavelet=settings.wavelet_type, levels=settings.wavelet_levels)

    # Structure Tensor
    struct_ref = analyze_structure_tensor(gray_ref)
    struct_test = analyze_structure_tensor(gray_test)

    # HOG Density
    hog_ref = compute_hog_density(gray_ref)
    hog_test = compute_hog_density(gray_test)

    # GLCM Z-scores
    glcm_zscores = compute_glcm_zscores(glcm_ref, glcm_test)

    # Defect Detection
    defects_analysis = analyze_defects(gray_test, min_area=settings.defect_min_area,
                                       morph_kernel_size=settings.morph_kernel_size,
                                       saliency_strength=settings.saliency_strength)

    # ============ PATTERN REPETITION ANALYSIS ============
    if settings.enable_pattern_repetition:
        logger.info("Detecting repeating patterns...")

        # Connected Components Analysis
        cc_ref = analyze_connected_components(gray_ref, min_area=settings.pattern_min_area,
                                              max_area=settings.pattern_max_area)
        cc_test = analyze_connected_components(gray_test, min_area=settings.pattern_min_area,
                                               max_area=settings.pattern_max_area)

        # Blob Detection
        blob_ref = analyze_blob_patterns(gray_ref, min_area=settings.pattern_min_area,
                                        max_area=settings.pattern_max_area,
                                        min_circularity=settings.blob_min_circularity,
                                        min_convexity=settings.blob_min_convexity)
        blob_test = analyze_blob_patterns(gray_test, min_area=settings.pattern_min_area,
                                          max_area=settings.pattern_max_area,
                                          min_circularity=settings.blob_min_circularity,
                                          min_convexity=settings.blob_min_convexity)

        # Keypoint-based Matching
        keypoint_matching = analyze_keypoint_matching(gray_ref, gray_test,
                                                      detector_type=settings.keypoint_detector,
                                                      match_threshold=settings.pattern_match_threshold)

        # Auto-correlation Analysis
        autocorr_ref = analyze_autocorrelation(gray_ref)
        autocorr_test = analyze_autocorrelation(gray_test)

        # Spatial Distribution
        spatial_ref = analyze_spatial_distribution(gray_ref, cc_ref['patterns'],
                                                   cell_size=settings.grid_cell_size)
        spatial_test = analyze_spatial_distribution(gray_test, cc_test['patterns'],
                                                    cell_size=settings.grid_cell_size)

        # Pattern Integrity Assessment
        integrity_assessment = assess_pattern_integrity(cc_ref['patterns'], cc_test['patterns'])

        # Missing/Extra Patterns Detection
        missing_extra = detect_missing_extra_patterns(cc_ref['patterns'], cc_test['patterns'],
                                                      spatial_ref, tolerance=50)

        # Pattern Repetition Status Determination
        count_diff = abs(cc_ref['count'] - cc_test['count'])
        if count_diff <= settings.pattern_count_tolerance:
            pattern_rep_status = "PASS"
        elif count_diff <= settings.pattern_count_tolerance * 2:
            pattern_rep_status = "CONDITIONAL"
        else:
            pattern_rep_status = "FAIL"

        logger.info(f"Pattern repetition analysis complete! ({cc_ref['count']} ref, {cc_test['count']} test patterns)")
    else:
        # Placeholder values if pattern repetition is disabled
        cc_ref = cc_test = blob_ref = blob_test = None
        keypoint_matching = autocorr_ref = autocorr_test = None
        spatial_ref = spatial_test = integrity_assessment = missing_extra = None
        pattern_rep_status = "DISABLED"

    # ============ ENHANCED COLOR ANALYSIS ============
    logger.info("Running enhanced color analysis...")

    # Check if spectral data is provided
    spectral_data_available = (settings.spectral_enable and
                               len(settings.spectral_ref_wavelengths) > 0 and
                               len(settings.spectral_sample_wavelengths) > 0)

    if spectral_data_available:
        logger.info("Processing spectral data...")
        # Use spectral data to compute XYZ
        xyz_ref_spectral = spectral_to_xyz(settings.spectral_ref_wavelengths,
                                           settings.spectral_ref_reflectance)
        xyz_test_spectral = spectral_to_xyz(settings.spectral_sample_wavelengths,
                                            settings.spectral_sample_reflectance)

        # Override mean XYZ with spectral data
        xyz_ref_mean = xyz_ref_spectral
        xyz_test_mean = xyz_test_spectral

        # Find spectral peaks and valleys
        spectral_features_ref = find_spectral_peaks_valleys(settings.spectral_ref_wavelengths,
                                                            settings.spectral_ref_reflectance)
        spectral_features_sample = find_spectral_peaks_valleys(settings.spectral_sample_wavelengths,
                                                               settings.spectral_sample_reflectance)
    else:
        # Use RGB-derived XYZ
        xyz_ref_mean = xyz_ref_D65.reshape(-1, 3).mean(axis=0)
        xyz_test_mean = xyz_test_D65.reshape(-1, 3).mean(axis=0)
        spectral_features_ref = []
        spectral_features_sample = []

    # CMC Color Difference
    if settings.use_delta_e_cmc:
        l_val, c_val = (2, 1) if settings.cmc_l_c_ratio == "2:1" else (1, 1)
        de_cmc_map = deltaE_CMC(lab_ref_D65, lab_test_D65, l=l_val, c=c_val)
        mean_de_cmc = float(np.mean(de_cmc_map))
    else:
        de_cmc_map = None
        mean_de_cmc = 0.0
    whiteness_ref, tint_ref = cie_whiteness_tint(xyz_ref_mean)
    whiteness_test, tint_test = cie_whiteness_tint(xyz_test_mean)

    # Yellowness Index
    yi_ref = astm_e313_yellowness(xyz_ref_mean)
    yi_test = astm_e313_yellowness(xyz_test_mean)

    # Extended Metamerism Analysis
    metamerism_results = []
    for ill_name in settings.metamerism_illuminants:
        if ill_name in WHITE_POINTS:
            _, _, de00_ill, _, _, _ = mean_de_under(ill_name)
            metamerism_results.append({'illuminant': ill_name, 'delta_e': de00_ill})

    worst_metamerism = max(metamerism_results, key=lambda x: x['delta_e']) if metamerism_results else None

    # QC metrics (using settings)
    color_score = max(0.0, 100.0 - mean76 * settings.color_score_multiplier)  # ΔE76 -> score
    pattern_score = ssim_score * 100.0
    overall_score = (color_score + pattern_score) / 2.0
    pattern_status = determine_status(ssim_score, settings.ssim_pass_threshold, settings.ssim_conditional_threshold, lower_is_better=False)

    # Decision logic based on scores
    if color_score >= settings.color_score_threshold and pattern_score >= settings.pattern_score_threshold:
        decision = "ACCEPT"
    elif overall_score >= settings.overall_score_threshold:
        decision = "CONDITIONAL ACCEPT"
    else:
        decision = "REJECT"

    # ---------------- Charts / images to embed ----------------
    logger.info("Generating visualizations...")
    # RGB histograms
    hist_ref_path  = os.path.join(TMP_IMG_DIR, "hist_ref.png")
    hist_test_path = os.path.join(TMP_IMG_DIR, "hist_test.png")
    plot_rgb_hist(ref_small, "Reference RGB Histogram", hist_ref_path)
    plot_rgb_hist(test_small,"Sample RGB Histogram",   hist_test_path)

    # ΔE heatmap
    heatmap_path = os.path.join(TMP_IMG_DIR, "heatmap_de00.png")
    plot_heatmap(de00_map, "ΔE2000 Heatmap (D65)", heatmap_path)

    # Spectral distribution (proxy)
    mean_rgb_ref  = ref_small.reshape(-1,3).mean(axis=0)/255.0
    mean_rgb_test = test_small.reshape(-1,3).mean(axis=0)/255.0
    spectral_path = os.path.join(TMP_IMG_DIR, "spectral_proxy.png")
    plot_spectral_proxy(mean_rgb_ref, mean_rgb_test, spectral_path)

    # a*b scatter + Lab bars
    ab_scatter_path = os.path.join(TMP_IMG_DIR, "ab_scatter.png")
    plot_ab_scatter(lab_ref_D65, lab_test_D65, ab_scatter_path)
    lab_ref_mean = lab_ref_D65.reshape(-1,3).mean(axis=0)
    lab_test_mean= lab_test_D65.reshape(-1,3).mean(axis=0)
    lab_bars_path = os.path.join(TMP_IMG_DIR, "lab_bars.png")
    plot_lab_bars(lab_ref_mean, lab_test_mean, lab_bars_path)

    # Region overlay image
    overlay_ref = overlay_regions(ref_small, pts)
    overlay_test= overlay_regions(test_small, pts)
    overlay_ref_path  = os.path.join(TMP_IMG_DIR, "ref_overlay.png")
    overlay_test_path = os.path.join(TMP_IMG_DIR, "test_overlay.png")
    Image.fromarray(overlay_ref).save(overlay_ref_path, "PNG")
    Image.fromarray(overlay_test).save(overlay_test_path, "PNG")

    # Difference and mask images
    diff_img_path = os.path.join(TMP_IMG_DIR, "abs_diff.png")
    thr_img_path  = os.path.join(TMP_IMG_DIR, "defect_mask.png")
    Image.fromarray(diff).save(diff_img_path, "PNG")
    Image.fromarray(thr).save(thr_img_path, "PNG")

    # ============ ADVANCED TEXTURE VISUALIZATIONS ============
    # FFT Power Spectrum
    fft_spectrum_path = os.path.join(TMP_IMG_DIR, "fft_power_spectrum.png")
    plot_fft_power_spectrum(fft_test['power_spectrum'], fft_test['peaks'], fft_spectrum_path)

    # Gabor Montage
    gabor_montage_path = os.path.join(TMP_IMG_DIR, "gabor_montage.png")
    plot_gabor_montage(gabor_test['energy_maps'], settings.gabor_frequencies,
                       settings.gabor_num_orientations, gabor_montage_path)

    # Gabor Orientation Histogram
    gabor_orient_path = os.path.join(TMP_IMG_DIR, "gabor_orientation.png")
    plot_gabor_orientation_histogram(gabor_test['results'], gabor_orient_path)

    # GLCM Radar Chart
    glcm_radar_path = os.path.join(TMP_IMG_DIR, "glcm_radar.png")
    plot_glcm_radar(glcm_ref, glcm_test, glcm_radar_path)

    # LBP Map and Histogram
    lbp_map_hist_path = os.path.join(TMP_IMG_DIR, "lbp_map_hist.png")
    plot_lbp_map_and_hist(lbp_test['lbp_map'], lbp_ref['histogram'], lbp_test['histogram'], lbp_map_hist_path)

    # Wavelet Energy Bars
    wavelet_energy_path = os.path.join(TMP_IMG_DIR, "wavelet_energy.png")
    plot_wavelet_energy_bars(wavelet_ref['energies'], wavelet_test['energies'], wavelet_energy_path)

    # Defect Saliency Map
    defect_saliency_path = os.path.join(TMP_IMG_DIR, "defect_saliency.png")
    plot_defect_saliency(defects_analysis['saliency_map'], defects_analysis['binary_map'],
                         defects_analysis['defects'], gray_test.shape, defect_saliency_path)

    # Line-Angle Histogram (Structure Tensor)
    line_angle_hist_path = os.path.join(TMP_IMG_DIR, "line_angle_histogram.png")
    if len(struct_test['orientation_degrees']) > 0:
        plot_line_angle_histogram(struct_test['orientation_degrees'], line_angle_hist_path)

    # ============ ENHANCED COLOR VISUALIZATIONS ============
    # Metamerism across illuminants
    metamerism_plot_path = os.path.join(TMP_IMG_DIR, "metamerism_illuminants.png")
    if metamerism_results:
        illuminant_names = [m['illuminant'] for m in metamerism_results]
        illuminant_des = [m['delta_e'] for m in metamerism_results]
        plot_metamerism_illuminants(illuminant_names, illuminant_des, metamerism_plot_path)

    # True Spectral Curve (if available)
    spectral_curve_path = os.path.join(TMP_IMG_DIR, "spectral_curve.png")
    if spectral_data_available:
        plot_spectral_curve(settings.spectral_ref_wavelengths, settings.spectral_ref_reflectance,
                           settings.spectral_sample_reflectance, spectral_curve_path)

    # ============ PATTERN REPETITION VISUALIZATIONS ============
    if settings.enable_pattern_repetition and cc_ref is not None:
        logger.info("Generating pattern repetition visualizations...")

        # Pattern Detection Maps
        pattern_detection_ref_path = os.path.join(TMP_IMG_DIR, "pattern_detection_ref.png")
        pattern_detection_test_path = os.path.join(TMP_IMG_DIR, "pattern_detection_test.png")
        plot_pattern_detection_map(ref_small, cc_ref['patterns'], "Reference", pattern_detection_ref_path)
        plot_pattern_detection_map(test_small, cc_test['patterns'], "Sample", pattern_detection_test_path)

        # Pattern Count Comparison
        pattern_count_path = os.path.join(TMP_IMG_DIR, "pattern_count_comparison.png")
        plot_pattern_count_comparison(cc_ref['count'], cc_test['count'], pattern_count_path)

        # Pattern Density Heatmaps
        pattern_density_ref_path = os.path.join(TMP_IMG_DIR, "pattern_density_ref.png")
        pattern_density_test_path = os.path.join(TMP_IMG_DIR, "pattern_density_test.png")
        plot_pattern_density_heatmap(spatial_ref['density_grid'], pattern_density_ref_path)
        plot_pattern_density_heatmap(spatial_test['density_grid'], pattern_density_test_path)

        # Missing/Extra Patterns Overlay
        missing_extra_path = os.path.join(TMP_IMG_DIR, "missing_extra_patterns.png")
        plot_missing_extra_patterns(test_small, missing_extra['missing_patterns'],
                                   missing_extra['extra_patterns'], missing_extra_path)

        # Pattern Size Distribution
        if cc_ref['patterns'] and cc_test['patterns']:
            pattern_size_dist_path = os.path.join(TMP_IMG_DIR, "pattern_size_distribution.png")
            areas_ref = [p['area'] for p in cc_ref['patterns']]
            areas_test = [p['area'] for p in cc_test['patterns']]
            plot_pattern_size_distribution(areas_ref, areas_test, pattern_size_dist_path)
        else:
            pattern_size_dist_path = None

        # Auto-correlation Surface
        autocorr_surface_path = os.path.join(TMP_IMG_DIR, "autocorrelation_surface.png")
        plot_autocorrelation_surface(autocorr_test['autocorr'], autocorr_test['peaks'], autocorr_surface_path)

        # Keypoint Matching Visualization
        keypoint_matching_path = os.path.join(TMP_IMG_DIR, "keypoint_matching.png")
        if keypoint_matching and keypoint_matching['keypoints_ref'] and keypoint_matching['keypoints_test']:
            plot_keypoint_matching(ref_small, test_small,
                                 keypoint_matching['keypoints_ref'],
                                 keypoint_matching['keypoints_test'],
                                 keypoint_matching['good_matches'], keypoint_matching_path)

        # Blob Detection Visualization
        blob_detection_ref_path = os.path.join(TMP_IMG_DIR, "blob_detection_ref.png")
        blob_detection_test_path = os.path.join(TMP_IMG_DIR, "blob_detection_test.png")
        if blob_ref and blob_ref['keypoints']:
            plot_blob_detection(ref_small, blob_ref['keypoints'], blob_detection_ref_path)
        if blob_test and blob_test['keypoints']:
            plot_blob_detection(test_small, blob_test['keypoints'], blob_detection_test_path)

        # Pattern Integrity Radar
        pattern_integrity_path = os.path.join(TMP_IMG_DIR, "pattern_integrity_radar.png")
        # Create dummy data for reference (perfect integrity)
        integrity_ref = {'size_similarity': 100.0, 'shape_similarity': 100.0,
                        'spatial_similarity': 100.0, 'integrity_score': 100.0}
        plot_pattern_integrity_radar(integrity_ref, integrity_assessment, pattern_integrity_path)

    # ---------------- PDF Build ----------------
    logger.info("Building PDF report...")
    now = get_local_time()  # Use Turkey timezone
    fname_stamp = now.strftime("%Y%m%d-%H%M%S")
    pdf_name = f"SpectraMatch Report {fname_stamp}.pdf"
    pdf_path = os.path.join("/content", pdf_name)

    doc = SimpleDocTemplate(
        pdf_path, pagesize=PAGE_SIZE,
        leftMargin=MARGIN_L, rightMargin=MARGIN_R,
        topMargin=MARGIN_T, bottomMargin=MARGIN_B
    )

    elements = []

    # ==== Cover Page (not numbered) ====
    logo_path = pick_logo()
    if logo_path:
        elements.append(RLImage(logo_path, width=1.8*inch, height=1.8*inch))  # Increased from 1.2 to 1.8 inches
        elements.append(Spacer(1, 15))

    elements.append(Paragraph(f"<font color='{BLUE1}'><b>{COMPANY_NAME}</b></font>", StyleTitle))
    elements.append(Paragraph(COMPANY_SUBTITLE, StyleSmall))
    elements.append(Spacer(1, 18))
    elements.append(Paragraph(f"<font color='{NEUTRAL_DARK}'><b>{REPORT_TITLE}</b></font>", ParagraphStyle("rt", parent=StyleTitle, fontSize=24)))
    elements.append(Spacer(1, 6))

    # Report metadata
    nice_date = now.strftime("%B %d, %Y at %I:%M %p")
    analysis_id = f"SPEC_{now.strftime('%Y%m%d_%H%M%S')}"
    operator = settings.operator_name  # Use operator from settings

    meta_data = [
        ["Report Date", nice_date],
        ["Operator", operator],
        ["Analysis ID", analysis_id],
        ["Software Version", SOFTWARE_VERSION],
    ]
    t = make_table([["Report Metadata",""]] + meta_data, colWidths=[1.8*inch, 4.0*inch])
    t.setStyle(TableStyle([("SPAN", (0,0),(1,0)),
                           ("BACKGROUND",(0,0),(1,0), BLUE2),
                           ("TEXTCOLOR",(0,0),(1,0), colors.white),
                           ("ALIGN",(0,0),(1,0), "LEFT")]))
    elements.append(t)
    elements.append(Spacer(1, 24))
    elements.append(PageBreak())

    # ==== ANALYSIS SETTINGS ====
    if settings.enable_analysis_settings:
        elements.append(Paragraph("<b>Analysis Settings</b>", StyleH1))
        elements.append(Spacer(1, 8))
        elements.append(Paragraph("The following settings were used for this analysis:", StyleBody))
        elements.append(Spacer(1, 8))
        elements.append(create_settings_summary_table(settings))
        elements.append(Spacer(1, 12))
        elements.append(PageBreak())  # Color Unit starts on a new page

    # ==== COLOR UNIT ====
    if settings.enable_color_unit:
        elements.append(Paragraph("<b>Color Unit</b>", StyleH1))

        # A. Input Images Section
        if settings.enable_color_input_images:
            elements.append(Paragraph("Input Images", StyleH2))
            img_row = [
                [RLImage(overlay_ref_path, width=2.5*inch, height=2.0*inch),
                 RLImage(overlay_test_path, width=2.5*inch, height=2.0*inch)]
            ]
            t_imgs = Table(img_row, colWidths=[2.7*inch, 2.7*inch])
            elements.append(t_imgs)
            elements.append(Spacer(1, 6))
            elements.append(make_table(
                [["Filenames",""],["Reference", os.path.basename(ref_path)], ["Sample", os.path.basename(test_path)]],
                colWidths=[1.2*inch, 4.8*inch]
            ))

            # Regional analysis info
            elements.append(Spacer(1, 6))
            rad = 12
            reg_info = [["Regional Analysis", ""],
                        ["Mode", "5-point grid within central area"],
                        ["Circle radius (px)", str(rad)],
                        ["Centers (x,y)", ", ".join([f"({x},{y})" for (y,x) in pts])]]
            t_reg = make_table(reg_info, colWidths=[1.6*inch, 4.4*inch])
            t_reg.setStyle(TableStyle([("SPAN",(0,0),(1,0)),
                                       ("BACKGROUND",(0,0),(1,0), NEUTRAL_L),
                                       ("FONTNAME",(0,0),(1,0),"Helvetica-Bold")]))
            elements.append(t_reg)
            elements.append(Spacer(1, 12))

        # B. Color Measurements Section (REDESIGNED for better readability)
        if settings.enable_color_measurements:
            elements.append(Paragraph("Color Measurements", StyleH2))
            elements.append(Paragraph("5-point regional analysis with Reference vs Sample comparison", StyleSmall))
            elements.append(Spacer(1, 6))

            # REDESIGNED TABLE: Group by measurement type for better readability
            # Define RGB background colors
            light_red_bg = colors.HexColor("#FFE6E6")    # Light red for R columns
            light_green_bg = colors.HexColor("#E6FFE6")  # Light green for G columns
            light_blue_bg = colors.HexColor("#E6E6FF")   # Light blue for B columns

            # Table 1: Position and RGB Values
            elements.append(Paragraph("<b>RGB Color Values</b>", StyleBody))
            rgb_cols = ["Region", "Position", "Ref R", "Test R", "Ref G", "Test G", "Ref B", "Test B"]
            rgb_tbl = [rgb_cols]
            for _,r in df_samples.iterrows():
                row = [int(r["Region"]),
                       f"({int(r['x'])}, {int(r['y'])})",
                       int(r["Ref R"]), int(r["Test R"]),
                       int(r["Ref G"]), int(r["Test G"]),
                       int(r["Ref B"]), int(r["Test B"])]
                rgb_tbl.append(row)

            t_rgb = Table(rgb_tbl, colWidths=[0.6*inch, 0.9*inch, 0.6*inch, 0.6*inch, 0.6*inch, 0.6*inch, 0.6*inch, 0.6*inch])
            rgb_style = [
                ("BACKGROUND", (0, 0), (-1, 0), BLUE2),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 9),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("GRID", (0, 0), (-1, -1), 0.5, NEUTRAL_L),
                ("FONTSIZE", (0, 1), (-1, -1), 8),
                # Highlight comparison columns with RED, GREEN, BLUE backgrounds
                ("BACKGROUND", (2, 1), (3, -1), light_red_bg),    # R columns - Red
                ("BACKGROUND", (4, 1), (5, -1), light_green_bg),  # G columns - Green
                ("BACKGROUND", (6, 1), (7, -1), light_blue_bg),   # B columns - Blue
            ]
            t_rgb.setStyle(TableStyle(rgb_style))
            elements.append(t_rgb)
            elements.append(Spacer(1, 10))

            # Table 2: LAB Values (ALL COLUMNS REMAIN WHITE - No highlighting)
            elements.append(Paragraph("<b>LAB* Color Space Values</b>", StyleBody))
            lab_cols = ["Region", "Ref L*", "Test L*", "Ref a*", "Test a*", "Ref b*", "Test b*"]
            lab_tbl = [lab_cols]
            for _,r in df_samples.iterrows():
                row = [int(r["Region"]),
                       fmt2(r["Ref L*"]), fmt2(r["Test L*"]),
                       fmt2(r["Ref a*"]), fmt2(r["Test a*"]),
                       fmt2(r["Ref b*"]), fmt2(r["Test b*"])]
                lab_tbl.append(row)

            t_lab = Table(lab_tbl, colWidths=[0.6*inch, 0.8*inch, 0.8*inch, 0.8*inch, 0.8*inch, 0.8*inch, 0.8*inch])
            lab_style = [
                ("BACKGROUND", (0, 0), (-1, 0), BLUE2),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 9),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("GRID", (0, 0), (-1, -1), 0.5, NEUTRAL_L),
                ("FONTSIZE", (0, 1), (-1, -1), 8),
                # NO BACKGROUND HIGHLIGHTING - All LAB columns remain white
            ]
            t_lab.setStyle(TableStyle(lab_style))
            elements.append(t_lab)
            elements.append(Spacer(1, 10))

            # Table 3: Color Difference (ΔE) Values
            elements.append(Paragraph("<b>Color Difference Metrics</b>", StyleBody))
            de_cols = ["Region", "ΔE76", "ΔE94", "ΔE2000", "Status"]
            de_tbl = [de_cols]
            for _,r in df_samples.iterrows():
                de2000_val = r["ΔE2000"]
                status = "PASS" if de2000_val < 2.0 else ("CONDITIONAL" if de2000_val <= 3.5 else "FAIL")
                row = [int(r["Region"]),
                       fmt2(r["ΔE76"]),
                       fmt2(r["ΔE94"]),
                       fmt2(r["ΔE2000"]),
                       status]
                de_tbl.append(row)

            t_de = Table(de_tbl, colWidths=[0.6*inch, 0.9*inch, 0.9*inch, 0.9*inch, 1.0*inch])
            de_style = [
                ("BACKGROUND", (0, 0), (-1, 0), BLUE2),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 9),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("GRID", (0, 0), (-1, -1), 0.5, NEUTRAL_L),
                ("FONTSIZE", (0, 1), (-1, -1), 8),
            ]
            # Add color coding for status column
            for i in range(1, len(de_tbl)):
                status = de_tbl[i][4]
                if status == "PASS":
                    de_style.append(("BACKGROUND", (4, i), (4, i), GREEN))
                    de_style.append(("TEXTCOLOR", (4, i), (4, i), colors.white))
                elif status == "CONDITIONAL":
                    de_style.append(("BACKGROUND", (4, i), (4, i), ORANGE))
                    de_style.append(("TEXTCOLOR", (4, i), (4, i), colors.white))
                else:  # FAIL
                    de_style.append(("BACKGROUND", (4, i), (4, i), RED))
                    de_style.append(("TEXTCOLOR", (4, i), (4, i), colors.white))

            t_de.setStyle(TableStyle(de_style))
            elements.append(t_de)
            elements.append(Spacer(1, 12))

        # C. Color Difference (ΔE) Section - NEW PAGE
        if settings.enable_color_difference:
            elements.append(PageBreak())  # Start on new page (Page 3)
            elements.append(Paragraph("Color Difference (ΔE)", StyleH2))
            # Determine status using settings thresholds
            th = settings.delta_e_threshold
            status = "PASS" if mean_de00_D65 < th else ("CONDITIONAL" if mean_de00_D65 <= settings.delta_e_conditional else "FAIL")
            d_table = [["Metric","Value","Threshold","Status","Interpretation"],
                       ["ΔE2000 (mean)", fmt2(mean_de00_D65), fmt2(th), status,
                        ("Not perceptible" if mean_de00_D65 < 1.0 else
                         "Perceptible (close observation)" if mean_de00_D65 < 2.0 else
                         "Perceptible at a glance" if mean_de00_D65 < 3.5 else
                         "Clear difference" if mean_de00_D65 < 5.0 else
                         "More different than similar")]]
            elements.append(make_table(d_table, colWidths=[1.4*inch,1.1*inch,1.1*inch,1.1*inch,2.0*inch]))
            elements.append(Spacer(1,6))

        # D. Statistical Analysis for RGB
        if settings.enable_color_statistical:
            elements.append(Paragraph("Statistical Analysis (RGB)", StyleH2))
            diff_rgb = (ref_small.astype(float) - test_small.astype(float))
            stats = []
            for i,ch in enumerate(["R","G","B"]):
                d = diff_rgb[...,i].ravel()
                stats.append([ch, fmt2(d.mean()), fmt2(d.std()), fmt2(d.max()), fmt2(d.min()),
                              fmt2(np.sqrt((d**2).mean()))])
            stats_tbl = [["Channel","Mean Diff","Std Dev","Max Diff","Min Diff","RMSE"]] + stats
            elements.append(make_table(stats_tbl, colWidths=[1.0*inch,1.0*inch,1.0*inch,1.0*inch,1.0*inch,1.0*inch]))
            elements.append(Spacer(1, 6))

        # E. Spectral Analysis (proxy)
        if settings.enable_color_spectral_proxy:
            elements.append(Paragraph("Spectral Analysis (Proxy)", StyleH2))
            elements.append(Paragraph("The chart approximates spectral behavior from RGB averages to aid visual comparison.", StyleSmall))
            elements.append(RLImage(spectral_path, width=5.3*inch, height=2.2*inch))
            elements.append(Spacer(1, 6))

        # F. Visual Difference Analysis (Keep together)
        if settings.enable_color_visual_diff:
            visual_diff_elements = []
            visual_diff_elements.append(Paragraph("Visual Difference Analysis", StyleH2))
            v_row = [
                [Paragraph("ΔE2000 Heatmap (D65)", StyleSmall), Paragraph("Absolute Difference (gray)", StyleSmall), Paragraph("Defect Mask (Otsu)", StyleSmall)],
                [RLImage(heatmap_path, width=2.7*inch, height=1.4*inch),
                 RLImage(diff_img_path, width=1.8*inch, height=1.4*inch),
                 RLImage(thr_img_path,  width=1.8*inch, height=1.4*inch)]
            ]
            t_v = Table(v_row, colWidths=[2.7*inch, 2.0*inch, 2.0*inch])
            visual_diff_elements.append(t_v)
            visual_diff_elements.append(Spacer(1, 10))
            elements.append(KeepTogether(visual_diff_elements))

        # G. Detailed Lab Color Space Analysis - NEW PAGE
        if settings.enable_color_lab_detailed:
            elements.append(PageBreak())  # Start on new page (Page 4)
            elements.append(Paragraph("Detailed Lab* Color Space Analysis", StyleH2))
            dL = float(lab_test_mean[0] - lab_ref_mean[0])
            da = float(lab_test_mean[1] - lab_ref_mean[1])
            db = float(lab_test_mean[2] - lab_ref_mean[2])
            lab_detail = [["Component","Reference","Sample","Difference","Interpretation"],
                          ["L* (Lightness)", fmt2(lab_ref_mean[0]), fmt2(lab_test_mean[0]), fmt2(dL),
                           ("No significant change" if abs(dL)<1.0 else ("Lighter" if dL>0 else "Darker"))],
                          ["a* (Green-Red)", fmt2(lab_ref_mean[1]), fmt2(lab_test_mean[1]), fmt2(da),
                           ("No significant shift" if abs(da)<1.0 else ("More Red" if da>0 else "More Green"))],
                          ["b* (Blue-Yellow)", fmt2(lab_ref_mean[2]), fmt2(lab_test_mean[2]), fmt2(db),
                           ("No significant shift" if abs(db)<1.0 else ("More Yellow" if db>0 else "More Blue"))]]
            elements.append(make_table(lab_detail, colWidths=[1.6*inch,1.1*inch,1.1*inch,1.0*inch,2.3*inch]))
            elements.append(Spacer(1, 6))

        # H. Visual Representation of Lab* (Keep together)
        if settings.enable_color_lab_viz:
            lab_viz_elements = []
            lab_viz_elements.append(Paragraph("Lab* Visualizations", StyleH2))
            lab_viz_elements.append(RLImage(ab_scatter_path, width=2.6*inch, height=2.6*inch))
            lab_viz_elements.append(Spacer(1, 6))
            lab_viz_elements.append(RLImage(lab_bars_path, width=4.6*inch, height=2.2*inch))
            lab_viz_elements.append(Spacer(1, 6))
            elements.append(KeepTogether(lab_viz_elements))

        # I. Quality Assessment based on Lab*
        if settings.enable_color_quality_assessment:
            dL = float(lab_test_mean[0] - lab_ref_mean[0]) if not settings.enable_color_lab_detailed else dL
            da = float(lab_test_mean[1] - lab_ref_mean[1]) if not settings.enable_color_lab_detailed else da
            db = float(lab_test_mean[2] - lab_ref_mean[2]) if not settings.enable_color_lab_detailed else db
            elements.append(Paragraph("Quality Assessment (Lab* thresholds)", StyleH2))
            overall_mag = math.sqrt(dL**2 + da**2 + db**2)
            qa = [["Parameter","Threshold","Actual","Status"],
                  ["ΔL*", f"≤ {settings.lab_l_threshold}", fmt2(abs(dL)), ("PASS" if abs(dL)<=settings.lab_l_threshold else "FAIL")],
                  ["Δa*", f"≤ {settings.lab_ab_threshold}", fmt2(abs(da)), ("PASS" if abs(da)<=settings.lab_ab_threshold else "FAIL")],
                  ["Δb*", f"≤ {settings.lab_ab_threshold}", fmt2(abs(db)), ("PASS" if abs(db)<=settings.lab_ab_threshold else "FAIL")],
                  ["Overall Magnitude", f"≤ {settings.lab_overall_threshold}", fmt2(overall_mag), ("PASS" if overall_mag<=settings.lab_overall_threshold else "FAIL")]]
            elements.append(make_table(qa, colWidths=[1.8*inch,1.2*inch,1.2*inch,1.0*inch]))
            elements.append(Spacer(1, 6))

        # J. Recommendations based on Lab* - NEW PAGE
        if settings.enable_color_recommendations:
            elements.append(PageBreak())  # Start on new page (Page 5)
            dL = float(lab_test_mean[0] - lab_ref_mean[0]) if not (settings.enable_color_lab_detailed or settings.enable_color_quality_assessment) else dL
            da = float(lab_test_mean[1] - lab_ref_mean[1]) if not (settings.enable_color_lab_detailed or settings.enable_color_quality_assessment) else da
            db = float(lab_test_mean[2] - lab_ref_mean[2]) if not (settings.enable_color_lab_detailed or settings.enable_color_quality_assessment) else db
            overall_mag = math.sqrt(dL**2 + da**2 + db**2) if not settings.enable_color_quality_assessment else overall_mag
            elements.append(Paragraph("Recommendations (Based on Lab*)", StyleH2))
            recs = []
            if abs(dL) > settings.lab_l_threshold:
                recs.append(("Lightness", "Adjust dye concentration / dwell time to correct L*"))
            if abs(da) > settings.lab_ab_threshold:
                recs.append(("Red–Green Axis", "Tune dye formulation on a* (shift toward opposite hue)"))
            if abs(db) > settings.lab_ab_threshold:
                recs.append(("Blue–Yellow Axis", "Modify temperature/pH to counter b* deviation"))
            if overall_mag > settings.lab_overall_threshold:
                recs.append(("Overall", "Review process parameters; consider re-processing and tighter QC sampling"))
            if not recs:
                recs.append(("Status", "Within tight tolerances. Maintain current parameters and monitor periodically."))
            rec_tbl = [["Parameter","Action"]] + recs
            elements.append(make_table(rec_tbl, colWidths=[1.8*inch, 4.0*inch]))
            elements.append(Spacer(1, 12))

        # K. Scoring & Status
        if settings.enable_color_scoring:
            pass  # Scoring section will be added after pattern analysis

    # ==== PATTERN UNIT (starts on new page) ====
    if settings.enable_pattern_unit:
        elements.append(PageBreak())
        elements.append(Paragraph("<b>Pattern Unit</b>", StyleH1))

        if settings.enable_pattern_ssim:
            patt_tbl = [["Metric","Value","Status"],
                        ["SSIM", fmt1(ssim_score*100)+"%", ("PASS" if ssim_score>settings.ssim_pass_threshold else ("CONDITIONAL" if ssim_score>settings.ssim_conditional_threshold else "FAIL"))]]
            if settings.enable_pattern_symmetry:
                patt_tbl.append(["Symmetry", fmt1(symmetry)+"%", ""])
            if settings.enable_pattern_repeat:
                patt_tbl.append(["Repeat (px)", f"H:{px}  V:{py}", ""])
            if settings.enable_pattern_edge:
                patt_tbl.append(["Edge Definition", fmt1(edge_def)+"/100", ""])
            patt_tbl.append(["Defect Density (rel.)", fmt1(defect_density), ""])
            patt_tbl.append(["Metamerism Index (D65/TL84/A)", fmt2(metamerism_index), ""])
            patt_tbl.append(["Uniformity Index (color)", fmt1(uni_idx)+"/100", ""])
            elements.append(make_table(patt_tbl, colWidths=[2.2*inch,1.6*inch,1.2*inch]))
            elements.append(Spacer(1, 8))

        # Histograms row
        elements.append(Paragraph("Histograms (RGB)", StyleH2))
        hist_row = [[RLImage(hist_ref_path, width=3.2*inch, height=1.6*inch),
                     RLImage(hist_test_path, width=3.2*inch, height=1.6*inch)]]
        elements.append(Table(hist_row, colWidths=[3.3*inch,3.3*inch]))
        elements.append(Spacer(1, 8))

        # ==== ADVANCED TEXTURE ANALYSIS ====
        if settings.enable_pattern_advanced:
            elements.append(PageBreak())
            elements.append(Paragraph("<b>Advanced Texture Analysis</b>", StyleH1))

            # A. FFT / Fourier Domain
            elements.append(Paragraph("Fourier Domain Analysis", StyleH2))
            elements.append(Paragraph("2D Fast Fourier Transform reveals periodic structures and directional patterns in the fabric.", StyleSmall))
            elements.append(RLImage(fft_spectrum_path, width=5*inch, height=3.5*inch))
            elements.append(Spacer(1, 6))

            # FFT Peaks Table
            fft_peak_tbl = [["Peak", "Radius", "Angle (°)", "Magnitude"]]
            for i, peak in enumerate(fft_test['peaks'][:5], start=1):
                fft_peak_tbl.append([f"P{i}", fmt2(peak['radius']), fmt2(peak['angle']), fmt2(peak['magnitude'])])
            elements.append(make_table(fft_peak_tbl, colWidths=[1.0*inch, 1.5*inch, 1.5*inch, 1.5*inch]))
            elements.append(Spacer(1, 6))

            # FFT Metrics
            fft_metrics_tbl = [["Metric", "Reference", "Sample"],
                               ["Fundamental Period (px)", fmt2(fft_ref['fundamental_period']), fmt2(fft_test['fundamental_period'])],
                               ["Dominant Orientation (°)", fmt2(fft_ref['fundamental_orientation']), fmt2(fft_test['fundamental_orientation'])],
                               ["Anisotropy Ratio", fmt2(fft_ref['anisotropy']), fmt2(fft_test['anisotropy'])]]
            elements.append(make_table(fft_metrics_tbl, colWidths=[2.5*inch, 1.5*inch, 1.5*inch]))
            elements.append(Spacer(1, 12))

            # B. Gabor Filter Bank - NEW PAGE
            elements.append(PageBreak())  # Start on new page (Page 7)
            elements.append(Paragraph("Gabor Filter Bank Analysis", StyleH2))
            elements.append(Paragraph("Multi-scale and multi-orientation responses capture texture at different frequencies and angles.", StyleSmall))
            elements.append(RLImage(gabor_montage_path, width=6.5*inch, height=3*inch))
            elements.append(Spacer(1, 6))
            elements.append(RLImage(gabor_orient_path, width=4.5*inch, height=3*inch))
            elements.append(Spacer(1, 6))

            # Gabor Statistics
            gabor_stats_tbl = [["Metric", "Reference", "Sample", "Δ"]]
            gabor_stats_tbl.append(["Dominant Orientation (°)", fmt2(gabor_ref['dominant_orientation']),
                                    fmt2(gabor_test['dominant_orientation']),
                                    fmt2(abs(gabor_ref['dominant_orientation'] - gabor_test['dominant_orientation']))])
            gabor_stats_tbl.append(["Coherency", fmt2(gabor_ref['coherency']), fmt2(gabor_test['coherency']),
                                    fmt2(abs(gabor_ref['coherency'] - gabor_test['coherency']))])
            elements.append(make_table(gabor_stats_tbl, colWidths=[2.0*inch, 1.5*inch, 1.5*inch, 1.0*inch]))
            elements.append(Spacer(1, 12))

            # C. GLCM / Haralick Features - NEW PAGE
            elements.append(PageBreak())  # Start on new page (Page 8)
            elements.append(Paragraph("GLCM Texture Features", StyleH2))
            elements.append(Paragraph("Gray Level Co-occurrence Matrix (GLCM) quantifies spatial relationships in texture.", StyleSmall))
            elements.append(RLImage(glcm_radar_path, width=5*inch, height=5*inch))
            elements.append(Spacer(1, 6))

            # GLCM Summary Table with Z-scores
            glcm_summary = [["Feature", "Reference", "Sample", "Δ", "z-score", "Interp."]]
            for feat in glcm_ref.keys():
                delta = abs(glcm_ref[feat] - glcm_test[feat])
                z = glcm_zscores[feat]
                # Interpretation based on z-score
                if abs(z) < 2:
                    z_interp = "Similar"
                elif abs(z) < 3:
                    z_interp = "Moderate"
                else:
                    z_interp = "Significant"
                glcm_summary.append([feat.capitalize(), fmt2(glcm_ref[feat]), fmt2(glcm_test[feat]),
                                    fmt2(delta), fmt2(z), z_interp])
            elements.append(make_table(glcm_summary, colWidths=[1.0*inch, 1.0*inch, 1.0*inch, 0.7*inch, 0.8*inch, 1.0*inch]))
            elements.append(Spacer(1, 12))

            # D. LBP (Local Binary Patterns) - NEW PAGE
            elements.append(PageBreak())  # Start on new page (Page 9)
            elements.append(Paragraph("Local Binary Patterns (LBP)", StyleH2))
            elements.append(Paragraph("LBP captures local texture by encoding pixel neighborhoods into binary patterns.", StyleSmall))
            elements.append(RLImage(lbp_map_hist_path, width=6.5*inch, height=3*inch))
            elements.append(Spacer(1, 6))

            # LBP Similarity Metrics
            lbp_sim_tbl = [["Metric", "Value", "Interpretation"],
                           ["χ² Distance", fmt2(lbp_chi2), "Lower is more similar"],
                           ["Bhattacharyya Distance", fmt2(lbp_bhatt), "Lower is more similar"]]
            elements.append(make_table(lbp_sim_tbl, colWidths=[2.5*inch, 1.5*inch, 2.0*inch]))
            elements.append(Spacer(1, 12))

            # E. Wavelet Multiresolution
            elements.append(Paragraph("Wavelet Decomposition", StyleH2))
            elements.append(Paragraph(f"Multiresolution analysis using {settings.wavelet_type} wavelet at {settings.wavelet_levels} levels.", StyleSmall))
            elements.append(RLImage(wavelet_energy_path, width=6.5*inch, height=5*inch))
            elements.append(Spacer(1, 6))

            # Wavelet Energy Table
            wavelet_tbl = [["Level", "Band", "Ref Energy", "Sample Energy", "Ratio"]]
            for i, (e_ref, e_test) in enumerate(zip(wavelet_ref['energies'], wavelet_test['energies'])):
                for band in ['LH', 'HL', 'HH']:
                    ratio = e_test[band] / (e_ref[band] + 1e-8)
                    wavelet_tbl.append([str(e_ref['level']), band, f"{e_ref[band]:.1e}", f"{e_test[band]:.1e}", fmt2(ratio)])
            elements.append(make_table(wavelet_tbl, colWidths=[0.8*inch, 0.8*inch, 1.5*inch, 1.5*inch, 1.0*inch]))
            elements.append(Spacer(1, 12))

            # F. Structure Tensor & Coherency
            elements.append(Paragraph("Structure Tensor Analysis", StyleH2))
            struct_coherency_tbl = [["Metric", "Reference", "Sample"],
                                    ["Mean Coherency", fmt2(struct_ref['mean_coherency']), fmt2(struct_test['mean_coherency'])],
                                    ["HOG Edge Density", fmt2(hog_ref['edge_density']), fmt2(hog_test['edge_density'])]]
            elements.append(make_table(struct_coherency_tbl, colWidths=[2.5*inch, 1.5*inch, 1.5*inch]))
            elements.append(Spacer(1, 6))

            # Line-Angle Histogram
            if len(struct_test['orientation_degrees']) > 0:
                elements.append(Paragraph("Line Angle Distribution", StyleH2))
                elements.append(Paragraph("Histogram showing dominant line orientations in the fabric structure.", StyleSmall))
                elements.append(RLImage(line_angle_hist_path, width=5.5*inch, height=3*inch))
            elements.append(Spacer(1, 12))

            # G. Defect Detection & Saliency - NEW PAGE
            elements.append(PageBreak())  # Start on new page (Page 11)
            elements.append(Paragraph("Defect Detection & Saliency Map", StyleH2))
            elements.append(Paragraph("Spectral residual saliency combined with morphological operations identifies potential defects.", StyleSmall))
            elements.append(RLImage(defect_saliency_path, width=6.5*inch, height=3*inch))
            elements.append(Spacer(1, 6))

            # Defect Catalog
            if defects_analysis['defects']:
                defect_catalog = [["ID", "Type", "Area (px²)", "Bounding Box (x0,y0,x1,y1)"]]
                for i, defect in enumerate(defects_analysis['defects'][:20], start=1):  # Limit to 20
                    bbox_str = f"({defect['bbox'][0]},{defect['bbox'][1]},{defect['bbox'][2]},{defect['bbox'][3]})"
                    defect_catalog.append([str(i), defect['type'], str(defect['area']), bbox_str])
                elements.append(make_table(defect_catalog, colWidths=[0.5*inch, 1.5*inch, 1.5*inch, 2.5*inch]))
                elements.append(Paragraph(f"Total defects detected: {defects_analysis['defect_count']}", StyleSmall))
            else:
                elements.append(Paragraph("No significant defects detected.", StyleBody))
            elements.append(Spacer(1, 12))

    # ==== PATTERN REPETITION UNIT (New Page) ====
    if settings.enable_pattern_repetition and cc_ref is not None:
        elements.append(PageBreak())
        elements.append(Paragraph("<b>Pattern Repetition Unit</b>", StyleH1))
        elements.append(Spacer(1, 6))

        # A. Pattern Detection Summary
        if settings.enable_pattern_rep_summary:
            elements.append(Paragraph("Pattern Detection Summary", StyleH2))

            # Status determination
            count_diff = abs(cc_ref['count'] - cc_test['count'])
            count_status = pattern_rep_status

            # Calculate spacing uniformity status
            spacing_status = "PASS" if spatial_test['uniformity_score'] >= 85 else ("CONDITIONAL" if spatial_test['uniformity_score'] >= 70 else "FAIL")

            summary_table = [[" Metric", "Reference", "Sample", "Δ", "Status"]]
            summary_table.append(["Total Pattern Count", str(cc_ref['count']), str(cc_test['count']),
                                f"{count_diff:+d}", count_status])
            summary_table.append(["Mean Pattern Area (px²)", fmt2(cc_ref['mean_area']), fmt2(cc_test['mean_area']),
                                fmt2(cc_test['mean_area'] - cc_ref['mean_area']), ""])
            summary_table.append(["Pattern Size CV%", fmt2(cc_ref['cv_area']), fmt2(cc_test['cv_area']),
                                f"{cc_test['cv_area'] - cc_ref['cv_area']:+.1f}%", ""])
            summary_table.append(["Spacing Uniformity (%)", fmt2(spatial_ref['uniformity_score']),
                                fmt2(spatial_test['uniformity_score']),
                                fmt2(spatial_test['uniformity_score'] - spatial_ref['uniformity_score']), spacing_status])
            summary_table.append(["Pattern Integrity (%)", "100.0", fmt2(integrity_assessment['integrity_score']),
                                fmt2(integrity_assessment['integrity_score'] - 100), ""])

            elements.append(make_table(summary_table, colWidths=[2.0*inch, 1.2*inch, 1.2*inch, 0.8*inch, 1.0*inch]))
            elements.append(Spacer(1, 12))

        # B. Pattern Count Analysis
        if settings.enable_pattern_rep_count:
            elements.append(Paragraph("Pattern Count Analysis", StyleH2))
            elements.append(Paragraph("Detected patterns in reference and sample images using connected components analysis.",
                                    StyleSmall))
            elements.append(Spacer(1, 4))

            # Side-by-side detection maps
            detection_row = [[RLImage(pattern_detection_ref_path, width=3.2*inch, height=2.4*inch),
                            RLImage(pattern_detection_test_path, width=3.2*inch, height=2.4*inch)]]
            elements.append(Table(detection_row, colWidths=[3.3*inch, 3.3*inch]))
            elements.append(Spacer(1, 6))

            # Count comparison chart
            elements.append(RLImage(pattern_count_path, width=5*inch, height=3.5*inch))
            elements.append(Spacer(1, 12))

        # C. Blob Detection Results - NEW PAGE
        if settings.enable_pattern_rep_blob:
            elements.append(PageBreak())  # Start on new page (Page 13)
            elements.append(Paragraph("Blob Detection Results", StyleH2))
            elements.append(Paragraph("SimpleBlobDetector analysis with circularity and convexity filtering.", StyleSmall))
            elements.append(Spacer(1, 4))

            # Blob statistics table
            blob_stats = [["Metric", "Reference", "Sample"]]
            blob_stats.append(["Blob Count", str(blob_ref['count']) if blob_ref else "0",
                             str(blob_test['count']) if blob_test else "0"])
            if blob_ref and blob_test and blob_ref['count'] > 0 and blob_test['count'] > 0:
                blob_stats.append(["Mean Area (px²)", fmt2(blob_ref['mean_area']), fmt2(blob_test['mean_area'])])
                blob_stats.append(["Area CV%", fmt2(blob_ref['cv_area']), fmt2(blob_test['cv_area'])])
                blob_stats.append(["Mean Size", fmt2(blob_ref['mean_size']), fmt2(blob_test['mean_size'])])

            elements.append(make_table(blob_stats, colWidths=[2.0*inch, 2.0*inch, 2.0*inch]))
            elements.append(Spacer(1, 6))

            # Blob detection visualizations (if available)
            if blob_ref and blob_test and blob_ref['keypoints'] and blob_test['keypoints']:
                blob_row = [[RLImage(blob_detection_ref_path, width=3.2*inch, height=2.4*inch),
                           RLImage(blob_detection_test_path, width=3.2*inch, height=2.4*inch)]]
                elements.append(Table(blob_row, colWidths=[3.3*inch, 3.3*inch]))
            elements.append(Spacer(1, 12))

        # D. Keypoint Matching Analysis
        if settings.enable_pattern_rep_keypoint and keypoint_matching:
            elements.append(Paragraph("Keypoint Matching Analysis", StyleH2))
            elements.append(Paragraph(f"Feature-based matching using {settings.keypoint_detector} detector.", StyleSmall))
            elements.append(Spacer(1, 4))

            # Keypoint statistics
            kp_stats = [["Metric", "Value"]]
            kp_stats.append(["Detector Type", settings.keypoint_detector])
            kp_stats.append(["Keypoints (Reference)", str(len(keypoint_matching['keypoints_ref']))])
            kp_stats.append(["Keypoints (Sample)", str(len(keypoint_matching['keypoints_test']))])
            kp_stats.append(["Good Matches", str(keypoint_matching['match_count'])])
            kp_stats.append(["Match Ratio", fmt2(keypoint_matching['match_ratio'] * 100) + "%"])
            kp_stats.append(["Matching Score", fmt2(keypoint_matching['matching_score']) + "%"])
            kp_stats.append(["Inliers (RANSAC)", str(keypoint_matching['inliers'])])

            elements.append(make_table(kp_stats, colWidths=[2.5*inch, 3.0*inch]))
            elements.append(Spacer(1, 6))

            # Keypoint matching visualization
            if keypoint_matching['keypoints_ref'] and keypoint_matching['keypoints_test']:
                elements.append(RLImage(keypoint_matching_path, width=6.5*inch, height=3.2*inch))
            elements.append(Spacer(1, 12))

        # E. Auto-correlation Analysis - NEW PAGE
        if settings.enable_pattern_rep_autocorr and autocorr_test:
            elements.append(PageBreak())  # Start on new page (Page 14)
            elements.append(Paragraph("Auto-correlation Analysis", StyleH2))
            elements.append(Paragraph("2D auto-correlation reveals pattern periodicity and regularity.", StyleSmall))
            elements.append(Spacer(1, 4))

            # Auto-correlation metrics
            autocorr_table = [["Metric", "Reference", "Sample"]]
            autocorr_table.append(["Periodicity Score", fmt2(autocorr_ref['periodicity_score']),
                                 fmt2(autocorr_test['periodicity_score'])])
            autocorr_table.append(["Pattern Spacing (px)", fmt2(autocorr_ref['pattern_spacing']),
                                 fmt2(autocorr_test['pattern_spacing'])])
            autocorr_table.append(["Regularity Score", fmt2(autocorr_ref['regularity_score']),
                                 fmt2(autocorr_test['regularity_score'])])
            autocorr_table.append(["Detected Peaks", str(len(autocorr_ref['peaks'])),
                                 str(len(autocorr_test['peaks']))])

            elements.append(make_table(autocorr_table, colWidths=[2.5*inch, 1.5*inch, 1.5*inch]))
            elements.append(Spacer(1, 6))

            # Auto-correlation surface plot
            elements.append(RLImage(autocorr_surface_path, width=6*inch, height=4.2*inch))
            elements.append(Spacer(1, 12))

        # F. Spatial Distribution Analysis - NEW PAGE
        if settings.enable_pattern_rep_spatial and spatial_test:
            elements.append(PageBreak())  # Start on new page (Page 15)
            elements.append(Paragraph("Spatial Distribution Analysis", StyleH2))
            elements.append(Paragraph(f"Grid-based pattern density analysis (cell size: {settings.grid_cell_size}px).",
                                    StyleSmall))
            elements.append(Spacer(1, 4))

            # Spatial metrics
            spatial_metrics = [["Metric", "Reference", "Sample"]]
            spatial_metrics.append(["Grid Size", f"{spatial_ref['n_rows']} × {spatial_ref['n_cols']}",
                                  f"{spatial_test['n_rows']} × {spatial_test['n_cols']}"])
            spatial_metrics.append(["Mean Density", fmt2(spatial_ref['mean_density']),
                                  fmt2(spatial_test['mean_density'])])
            spatial_metrics.append(["Density Std Dev", fmt2(spatial_ref['std_density']),
                                  fmt2(spatial_test['std_density'])])
            spatial_metrics.append(["Density CV%", fmt2(spatial_ref['cv_density']),
                                  fmt2(spatial_test['cv_density'])])
            spatial_metrics.append(["Uniformity Score", fmt2(spatial_ref['uniformity_score']),
                                  fmt2(spatial_test['uniformity_score'])])

            elements.append(make_table(spatial_metrics, colWidths=[2.0*inch, 2.0*inch, 2.0*inch]))
            elements.append(Spacer(1, 6))

            # Density heatmaps
            density_row = [[RLImage(pattern_density_ref_path, width=3.0*inch, height=2.4*inch),
                          RLImage(pattern_density_test_path, width=3.0*inch, height=2.4*inch)]]
            elements.append(Table(density_row, colWidths=[3.2*inch, 3.2*inch]))
            elements.append(Spacer(1, 12))

        # G. Pattern Integrity Assessment - NEW PAGE
        if settings.enable_pattern_rep_integrity and integrity_assessment:
            elements.append(PageBreak())  # Start on new page (Page 16)
            elements.append(Paragraph("Pattern Integrity Assessment", StyleH2))
            elements.append(Paragraph("Multi-dimensional comparison of pattern properties.", StyleSmall))
            elements.append(Spacer(1, 4))

            # Integrity scores
            integrity_table = [["Metric", "Score (%)", "Status"]]

            size_status = "PASS" if integrity_assessment['size_similarity'] >= 85 else ("CONDITIONAL" if integrity_assessment['size_similarity'] >= 70 else "FAIL")
            shape_status = "PASS" if integrity_assessment['shape_similarity'] >= 85 else ("CONDITIONAL" if integrity_assessment['shape_similarity'] >= 70 else "FAIL")
            spatial_status = "PASS" if integrity_assessment['spatial_similarity'] >= 85 else ("CONDITIONAL" if integrity_assessment['spatial_similarity'] >= 70 else "FAIL")
            overall_status = "PASS" if integrity_assessment['integrity_score'] >= 85 else ("CONDITIONAL" if integrity_assessment['integrity_score'] >= 70 else "FAIL")

            integrity_table.append(["Size Similarity", fmt2(integrity_assessment['size_similarity']), size_status])
            integrity_table.append(["Shape Similarity", fmt2(integrity_assessment['shape_similarity']), shape_status])
            integrity_table.append(["Spatial Similarity", fmt2(integrity_assessment['spatial_similarity']), spatial_status])
            integrity_table.append(["Overall Integrity", fmt2(integrity_assessment['integrity_score']), overall_status])

            elements.append(make_table(integrity_table, colWidths=[2.5*inch, 1.5*inch, 1.5*inch]))
            elements.append(Spacer(1, 6))

            # Integrity radar chart
            elements.append(RLImage(pattern_integrity_path, width=5*inch, height=5*inch))
            elements.append(Spacer(1, 6))

            # Pattern size distribution
            if pattern_size_dist_path:
                elements.append(RLImage(pattern_size_dist_path, width=5.5*inch, height=3.5*inch))
            elements.append(Spacer(1, 12))

        # H. Missing/Extra Patterns Catalog
        if settings.enable_pattern_rep_catalog and missing_extra:
            elements.append(Paragraph("Missing/Extra Patterns Catalog", StyleH2))
            elements.append(Spacer(1, 4))

            # Missing/extra patterns overlay
            elements.append(RLImage(missing_extra_path, width=5.5*inch, height=4*inch))
            elements.append(Spacer(1, 6))

            # Missing patterns table
            if missing_extra['missing_patterns']:
                elements.append(Paragraph(f"<b>Missing Patterns ({missing_extra['missing_count']})</b>", StyleBody))
                elements.append(Spacer(1, 4))
                missing_table = [["ID", "Location (x, y)", "Expected Size (px²)", "Severity"]]
                for i, pattern in enumerate(missing_extra['missing_patterns'][:20], start=1):  # Limit to 20
                    loc_str = f"({pattern['location'][0]}, {pattern['location'][1]})"
                    missing_table.append([str(i), loc_str, f"~{int(pattern['expected_area'])}", pattern['severity']])
                elements.append(make_table(missing_table, colWidths=[0.5*inch, 1.5*inch, 1.8*inch, 1.0*inch]))
                if missing_extra['missing_count'] > 20:
                    elements.append(Paragraph(f"... and {missing_extra['missing_count'] - 20} more missing patterns.", StyleSmall))
                elements.append(Spacer(1, 8))
            else:
                elements.append(Paragraph("<b>No missing patterns detected.</b>", StyleBody))
                elements.append(Spacer(1, 8))

            # Extra patterns table
            if missing_extra['extra_patterns']:
                elements.append(Paragraph(f"<b>Extra Patterns ({missing_extra['extra_count']})</b>", StyleBody))
                elements.append(Spacer(1, 4))
                extra_table = [["ID", "Location (x, y)", "Area (px²)", "Severity"]]
                for i, pattern in enumerate(missing_extra['extra_patterns'][:20], start=1):  # Limit to 20
                    loc_str = f"({pattern['location'][0]}, {pattern['location'][1]})"
                    extra_table.append([str(i), loc_str, str(int(pattern['area'])), pattern['severity']])
                elements.append(make_table(extra_table, colWidths=[0.5*inch, 1.5*inch, 1.8*inch, 1.0*inch]))
                if missing_extra['extra_count'] > 20:
                    elements.append(Paragraph(f"... and {missing_extra['extra_count'] - 20} more extra patterns.", StyleSmall))
                elements.append(Spacer(1, 8))
            else:
                elements.append(Paragraph("<b>No extra patterns detected.</b>", StyleBody))
                elements.append(Spacer(1, 8))

            # Recommendations
            elements.append(Paragraph("Recommendations", StyleH2))
            recs = []
            if count_diff > settings.pattern_count_tolerance * 2:
                recs.append(("Pattern Count Mismatch", "Critical: Investigate dyeing/printing process for pattern dropout or duplication"))
            elif count_diff > settings.pattern_count_tolerance:
                recs.append(("Pattern Count Variation", "Monitor: Pattern count is acceptable but close to limit"))

            if spatial_test['uniformity_score'] < 70:
                recs.append(("Poor Spatial Uniformity", "Check fabric tension and printing alignment"))

            if integrity_assessment['integrity_score'] < 80:
                recs.append(("Pattern Integrity Issues", "Review pattern size and shape consistency in production"))

            if not recs:
                recs.append(("Status", "Pattern repetition is within acceptable limits. Maintain current parameters."))

            rec_tbl = [["Parameter", "Action"]] + recs
            elements.append(make_table(rec_tbl, colWidths=[2.0*inch, 4.0*inch]))
            elements.append(Spacer(1, 12))

    # ==== SPECTROPHOTOMETER-LIKE COLOR MODULE (New Page) ====
    if settings.enable_spectrophotometer:
        elements.append(PageBreak())
        elements.append(Paragraph("<b>Spectrophotometer Simulation</b>", StyleH1))

        # Instrument Metadata
        if settings.enable_spectro_config:
            elements.append(Paragraph("Instrument Configuration", StyleH2))
            inst_meta = [["Parameter", "Value"],
                         ["Observer Angle", f"{settings.observer_angle}°"],
                         ["Geometry Mode", settings.geometry_mode],
                         ["Illuminant (Primary)", "D65"],
                         ["UV Control", settings.uv_control_note]]
            elements.append(make_table(inst_meta, colWidths=[2.5*inch, 3.5*inch]))
            elements.append(Spacer(1, 12))

        # Enhanced Color Difference Suite
        if settings.enable_spectro_color_diff_methods:
            elements.append(Paragraph("Color Difference Methods", StyleH2))
            color_diff_suite = [["Method", "Mean ΔE", "Status"]]
            color_diff_suite.append(["ΔE76 (CIE 1976)", fmt2(mean76), status_color])
            color_diff_suite.append(["ΔE94 (CIE 1994)", fmt2(float(np.mean(de94_map))), status_color])
            color_diff_suite.append(["ΔE2000 (CIEDE2000)", fmt2(mean_de00_D65), status])
            if settings.use_delta_e_cmc:
                cmc_status = "PASS" if mean_de_cmc < settings.delta_e_threshold else ("CONDITIONAL" if mean_de_cmc <= settings.delta_e_conditional else "FAIL")
                color_diff_suite.append([f"ΔE CMC ({settings.cmc_l_c_ratio})", fmt2(mean_de_cmc), cmc_status])
            elements.append(make_table(color_diff_suite, colWidths=[2.5*inch, 1.5*inch, 1.5*inch]))
            elements.append(Spacer(1, 12))

        # Whiteness & Yellowness Indices
        if settings.enable_spectro_whiteness:
            elements.append(Paragraph("Whiteness & Yellowness Indices", StyleH2))
            wy_tbl = [["Index", "Reference", "Sample", "Threshold", "Status"]]
            w_status = "PASS" if whiteness_test >= settings.whiteness_min else "FAIL"
            y_status = "PASS" if yi_test <= settings.yellowness_max else "FAIL"
            wy_tbl.append(["CIE Whiteness (ISO 11475)", fmt2(whiteness_ref), fmt2(whiteness_test), f"≥ {settings.whiteness_min}", w_status])
            wy_tbl.append(["CIE Tint", fmt2(tint_ref), fmt2(tint_test), "—", "—"])
            wy_tbl.append(["Yellowness Index (ASTM E313)", fmt2(yi_ref), fmt2(yi_test), f"≤ {settings.yellowness_max}", y_status])
            elements.append(make_table(wy_tbl, colWidths=[2.2*inch, 1.2*inch, 1.2*inch, 1.0*inch, 0.8*inch]))
            elements.append(Spacer(1, 12))

        # Metamerism Analysis
        if settings.enable_spectro_metamerism:
            elements.append(Paragraph("Metamerism Analysis", StyleH2))
            elements.append(Paragraph("Color difference under various illuminants to assess metamerism.", StyleSmall))
            if metamerism_results:
                # Keep plot with heading
                metamerism_plot_group = []
                metamerism_plot_group.append(RLImage(metamerism_plot_path, width=5.5*inch, height=3.5*inch))
                metamerism_plot_group.append(Spacer(1, 6))
                elements.append(KeepTogether(metamerism_plot_group))

                # Illuminant ΔE Table
                meta_tbl = [["Illuminant", "ΔE2000", "Status"]]
                for m in metamerism_results:
                    m_status = "PASS" if m['delta_e'] < 2.0 else ("CONDITIONAL" if m['delta_e'] < 3.5 else "FAIL")
                    meta_tbl.append([m['illuminant'], fmt2(m['delta_e']), m_status])
                elements.append(make_table(meta_tbl, colWidths=[2.0*inch, 1.5*inch, 1.5*inch]))

                if worst_metamerism:
                    elements.append(Spacer(1, 6))
                    elements.append(Paragraph(f"<b>Worst-case metamerism:</b> {worst_metamerism['illuminant']} (ΔE = {fmt2(worst_metamerism['delta_e'])})", StyleBody))
            elements.append(Spacer(1, 12))

        # True Spectral Data Analysis (if available)
        if settings.enable_spectro_spectral_data and spectral_data_available:
            spectral_section = []
            spectral_section.append(Paragraph("True Spectral Reflectance Analysis", StyleH2))
            spectral_section.append(Paragraph(f"Spectral data provided: Reference ({settings.spectral_ref_path}), Sample ({settings.spectral_sample_path})", StyleSmall))
            spectral_section.append(RLImage(spectral_curve_path, width=6*inch, height=4*inch))
            spectral_section.append(Spacer(1, 6))
            elements.append(KeepTogether(spectral_section))

            # Spectral Peak/Valley Table
            if spectral_features_ref or spectral_features_sample:
                peak_valley_tbl = [["Sample", "Type", "Wavelength (nm)", "Reflectance (%)"]]
                for feat in spectral_features_ref[:3]:
                    peak_valley_tbl.append(["Reference", feat['type'], fmt1(feat['wavelength']), fmt1(feat['reflectance'])])
                for feat in spectral_features_sample[:3]:
                    peak_valley_tbl.append(["Sample", feat['type'], fmt1(feat['wavelength']), fmt1(feat['reflectance'])])
                elements.append(make_table(peak_valley_tbl, colWidths=[1.5*inch, 1.0*inch, 1.5*inch, 1.5*inch]))

            elements.append(Spacer(1, 6))
            elements.append(Paragraph("<i>Note: Tristimulus values computed from spectral data using CIE color matching functions.</i>", StyleSmall))
            elements.append(Spacer(1, 12))

        # Calibration Notes
        if settings.enable_spectro_calibration:
            elements.append(Paragraph("Calibration & Limitations", StyleH2))
            calib_notes = [["Parameter", "Status / Note"],
                           ["White Tile Calibration", "Simulated (not available for RGB images)"],
                           ["UV Control", settings.uv_control_note],
                           ["Data Source", "Spectral CSV" if spectral_data_available else "RGB → XYZ conversion"]]
            elements.append(make_table(calib_notes, colWidths=[2.5*inch, 3.5*inch]))
            elements.append(Spacer(1, 12))

    # ==== Conclusion & Decision (Keep together on same page)
    conclusion_elements = []
    conclusion_elements.append(Paragraph("Conclusion & Decision", StyleH2))
    dec_color = GREEN if decision.startswith("ACCEPT") else (ORANGE if "CONDITIONAL" in decision else RED)
    # Colored box
    conclusion_elements.append(Spacer(1, 4))
    conclusion_elements.append(Paragraph(
        f"<para backColor='{dec_color}'>"
        f"<font color='white'><b>Recommendation: {decision}</b></font></para>", StyleBadge))
    conclusion_elements.append(Spacer(1, 6))
    if "REJECT" in decision:
        concl = [
            "Significant deviation from reference; corrective action required.",
            "Review dyeing parameters, chemical concentrations, and fabric preparation.",
            "Consider re-processing and implement enhanced QC measures."
        ]
    elif "CONDITIONAL" in decision:
        concl = [
            "Sample is near limits; monitor closely.",
            "Fine-tune process parameters to improve stability."
        ]
    else:
        concl = [
            "Sample matches reference within acceptable tolerances.",
            "Maintain parameters and regular monitoring."
        ]
    conclusion_elements.append(Paragraph("• " + "<br/>• ".join(concl), StyleBody))
    conclusion_elements.append(Spacer(1, 12))

    # Wrap conclusion in KeepTogether to prevent page break
    elements.append(KeepTogether(conclusion_elements))

    # ==== Build the PDF
    try:
        doc.build(elements, onFirstPage=first_page_header, onLaterPages=header_footer)
        logger.info(f"PDF report generated successfully: {pdf_name}")
        return pdf_path
    except Exception as e:
        logger.error(f"Failed to build PDF: {str(e)}")
        raise RuntimeError(f"PDF generation failed: {str(e)}")

# ----------------------------
# Generate Analysis Settings Technical Report
# ----------------------------
def generate_analysis_settings_report(ref_path, test_path, ref, test, settings):
    """Generate a compact technical report with analysis settings (small text for technicians)"""
    from datetime import datetime, timedelta

    # Use UTC+3 timestamp (matching main report)
    tz_offset = timedelta(hours=TIMEZONE_OFFSET_HOURS)
    now_utc3 = datetime.utcnow() + tz_offset
    timestamp_str = now_utc3.strftime("%Y%m%d_%H%M%S")
    pdf_path = f"Analysis_Settings_Report_{timestamp_str}.pdf"

    # Store the main report name that would be generated
    main_report_name = f"QC_Report_{timestamp_str}.pdf"

    # Create PDF with smaller margins for compact layout
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=A4,
        leftMargin=30,
        rightMargin=30,
        topMargin=60,
        bottomMargin=60
    )

    elements = []

    # Define styles for small text (technical report)
    styles = getSampleStyleSheet()

    # Small title style
    StyleTitle = ParagraphStyle(
        'TechTitle',
        parent=styles['Heading1'],
        fontSize=14,
        textColor=BLUE2,
        spaceAfter=8,
        alignment=1,  # Center
        fontName='Helvetica-Bold'
    )

    # Small heading style
    StyleHeading = ParagraphStyle(
        'TechHeading',
        parent=styles['Heading2'],
        fontSize=10,
        textColor=BLUE1,
        spaceAfter=6,
        spaceBefore=10,
        fontName='Helvetica-Bold'
    )

    # Small body text style
    StyleBody = ParagraphStyle(
        'TechBody',
        parent=styles['BodyText'],
        fontSize=7,
        leading=9,
        spaceAfter=4,
        fontName='Helvetica'
    )

    # Title
    elements.append(Paragraph("⚙️ ANALYSIS SETTINGS TECHNICAL REPORT", StyleTitle))
    elements.append(Paragraph(f"<b>Operator:</b> {settings.operator_name} | <b>Timestamp (UTC+3):</b> {now_utc3.strftime('%Y-%m-%d %H:%M:%S')}",
                             ParagraphStyle('TechSubtitle', parent=StyleBody, fontSize=8, alignment=1, spaceAfter=10)))
    elements.append(Spacer(1, 10))

    # Logo (if available)
    logo_path = "logo_vertical_512x256.png"
    if os.path.exists(logo_path):
        try:
            logo = RLImage(logo_path, width=60, height=30)
            elements.append(logo)
            elements.append(Spacer(1, 8))
        except:
            pass

    # ===== REPORT METADATA =====
    elements.append(Paragraph("Report Information", StyleHeading))
    report_info_data = [
        ["Field", "Value"],
        ["Main Report Name", main_report_name],
        ["Technical Report Name", os.path.basename(pdf_path)],
        ["Generation Timestamp (UTC+3)", now_utc3.strftime('%Y-%m-%d %H:%M:%S')],
        ["Operator", settings.operator_name],
        ["Reference Image", os.path.basename(ref_path)],
        ["Sample Image", os.path.basename(test_path)],
        ["Image Dimensions", f"{ref.shape[1]} × {ref.shape[0]} pixels"],
    ]

    report_info_table = Table(report_info_data, colWidths=[2.5*inch, 4.5*inch])
    report_info_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), BLUE2),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 8),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
        ('TOPPADDING', (0, 0), (-1, 0), 6),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 3),
        ('TOPPADDING', (0, 1), (-1, -1), 3),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.Color(0.95, 0.95, 0.95)]),
        ('WORDWRAP', (0, 0), (-1, -1), True),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ]))
    elements.append(report_info_table)
    elements.append(Spacer(1, 10))

    # ===== REGION ANALYSIS DETAILS =====
    elements.append(Paragraph("Region Analysis Configuration", StyleHeading))

    if settings.use_crop:
        analysis_mode = f"Selected Region Only ({settings.crop_shape.upper()})"
        if settings.crop_shape == "circle":
            shape_details = f"Circle: Center ({settings.crop_center_x}, {settings.crop_center_y}), Diameter {settings.crop_diameter} px"
        else:  # rectangle
            shape_details = f"Rectangle: Center ({settings.crop_center_x}, {settings.crop_center_y}), Size {settings.crop_width} × {settings.crop_height} px"
    else:
        analysis_mode = "Entire Image (No Region Selection)"
        shape_details = "N/A - Full image analyzed"

    region_data = [
        ["Configuration", "Details"],
        ["Analysis Mode", analysis_mode],
        ["Shape Details", shape_details],
        ["Sample Points", f"{settings.num_sample_points} points"],
    ]

    region_table = Table(region_data, colWidths=[2.5*inch, 4.5*inch])
    region_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), GREEN),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 8),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
        ('TOPPADDING', (0, 0), (-1, 0), 6),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 3),
        ('TOPPADDING', (0, 1), (-1, -1), 3),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.Color(0.95, 0.95, 0.95)]),
        ('WORDWRAP', (0, 0), (-1, -1), True),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ]))
    elements.append(region_table)
    elements.append(Spacer(1, 10))

    # ===== ENABLED/DISABLED SECTIONS =====
    elements.append(Paragraph("Report Sections Included", StyleHeading))

    sections_status = [
        ["Section", "Status"],
        # Main sections
        ["📊 Analysis Settings", "✓ Enabled" if settings.enable_analysis_settings else "✗ Disabled"],
        ["🎨 Color Unit", "✓ Enabled" if settings.enable_color_unit else "✗ Disabled"],
        # Color sub-sections (only show if Color Unit enabled)
    ]

    if settings.enable_color_unit:
        sections_status.extend([
            ["  ├─ Input Images", "✓ Enabled" if settings.enable_color_input_images else "✗ Disabled"],
            ["  ├─ Color Measurements", "✓ Enabled" if settings.enable_color_measurements else "✗ Disabled"],
            ["  ├─ Color Difference (ΔE)", "✓ Enabled" if settings.enable_color_difference else "✗ Disabled"],
            ["  ├─ Statistical Analysis", "✓ Enabled" if settings.enable_color_statistical else "✗ Disabled"],
            ["  ├─ Spectral Proxy", "✓ Enabled" if settings.enable_color_spectral_proxy else "✗ Disabled"],
            ["  ├─ Visual Difference", "✓ Enabled" if settings.enable_color_visual_diff else "✗ Disabled"],
            ["  ├─ Lab* Detailed Analysis", "✓ Enabled" if settings.enable_color_lab_detailed else "✗ Disabled"],
            ["  ├─ Lab* Visualizations", "✓ Enabled" if settings.enable_color_lab_viz else "✗ Disabled"],
            ["  ├─ Quality Assessment", "✓ Enabled" if settings.enable_color_quality_assessment else "✗ Disabled"],
            ["  ├─ Scoring & Status", "✓ Enabled" if settings.enable_color_scoring else "✗ Disabled"],
            ["  └─ Recommendations", "✓ Enabled" if settings.enable_color_recommendations else "✗ Disabled"],
        ])

    sections_status.append(["🔲 Pattern Unit", "✓ Enabled" if settings.enable_pattern_unit else "✗ Disabled"])

    if settings.enable_pattern_unit:
        sections_status.extend([
            ["  ├─ SSIM Analysis", "✓ Enabled" if settings.enable_pattern_ssim else "✗ Disabled"],
            ["  ├─ Symmetry Analysis", "✓ Enabled" if settings.enable_pattern_symmetry else "✗ Disabled"],
            ["  ├─ Edge Definition", "✓ Enabled" if settings.enable_pattern_edge else "✗ Disabled"],
            ["  ├─ Repeat Period", "✓ Enabled" if settings.enable_pattern_repeat else "✗ Disabled"],
            ["  └─ Advanced Texture", "✓ Enabled" if settings.enable_pattern_advanced else "✗ Disabled"],
        ])

    sections_status.append(["🔬 Spectrophotometer", "✓ Enabled" if settings.enable_spectrophotometer else "✗ Disabled"])

    if settings.enable_spectrophotometer:
        sections_status.extend([
            ["  ├─ Instrument Config", "✓ Enabled" if settings.enable_spectro_config else "✗ Disabled"],
            ["  ├─ Color Diff Methods", "✓ Enabled" if settings.enable_spectro_color_diff_methods else "✗ Disabled"],
            ["  ├─ Whiteness/Yellowness", "✓ Enabled" if settings.enable_spectro_whiteness else "✗ Disabled"],
            ["  ├─ Metamerism", "✓ Enabled" if settings.enable_spectro_metamerism else "✗ Disabled"],
            ["  ├─ Spectral Data", "✓ Enabled" if settings.enable_spectro_spectral_data else "✗ Disabled"],
            ["  └─ Calibration Notes", "✓ Enabled" if settings.enable_spectro_calibration else "✗ Disabled"],
        ])

    sections_table = Table(sections_status, colWidths=[4.5*inch, 2.5*inch])
    sections_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), ORANGE),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 8),
        ('FONTNAME', (0, 1), (-1, -1), 'Courier'),  # Monospace for tree structure
        ('FONTSIZE', (0, 1), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
        ('TOPPADDING', (0, 0), (-1, 0), 6),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 2),
        ('TOPPADDING', (0, 1), (-1, -1), 2),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.Color(0.95, 0.95, 0.95)]),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ]))
    elements.append(sections_table)
    elements.append(Spacer(1, 10))

    # ===== TWO-COLUMN LAYOUT FOR SETTINGS =====
    elements.append(PageBreak())
    elements.append(Paragraph("Detailed Analysis Parameters", StyleHeading))

    # Create TWO-COLUMN layout for settings (left and right tables)
    # LEFT COLUMN - Color & Pattern Settings
    left_settings = [
        ["Parameter", "Value"],
        # Color Thresholds
        ["— COLOR THRESHOLDS —", ""],
        ["ΔE Threshold", f"{settings.delta_e_threshold}"],
        ["ΔE Conditional", f"{settings.delta_e_conditional}"],
        ["Lab L* Threshold", f"{settings.lab_l_threshold}"],
        ["Lab a*b* Threshold", f"{settings.lab_ab_threshold}"],
        ["Lab Overall Threshold", f"{settings.lab_overall_threshold}"],
        ["— PATTERN THRESHOLDS —", ""],
        ["SSIM Pass Threshold", f"{settings.ssim_pass_threshold}"],
        ["SSIM Conditional", f"{settings.ssim_conditional_threshold}"],
        ["— SCORING PARAMETERS —", ""],
        ["Color Score Multiplier", f"{settings.color_score_multiplier}"],
        ["Uniformity Std Multiplier", f"{settings.uniformity_std_multiplier}"],
        ["Color Score Threshold", f"{settings.color_score_threshold}"],
        ["Pattern Score Threshold", f"{settings.pattern_score_threshold}"],
        ["Overall Score Threshold", f"{settings.overall_score_threshold}"],
        ["— TEXTURE ANALYSIS —", ""],
        ["FFT Notch Filter", "Enabled" if settings.fft_enable_notch else "Disabled"],
        ["FFT Peaks", str(settings.fft_num_peaks)],
        ["Gabor Frequencies", settings.gabor_frequencies_str],
        ["Gabor Orientations", str(settings.gabor_num_orientations)],
        ["GLCM Distances", settings.glcm_distances_str],
        ["GLCM Angles", settings.glcm_angles_str],
    ]

    # RIGHT COLUMN - Spectrophotometer & Advanced Settings
    right_settings = [
        ["Parameter", "Value"],
        # Spectrophotometer
        ["— SPECTROPHOTOMETER —", ""],
        ["Observer Angle", f"{settings.observer_angle}°"],
        ["Geometry Mode", settings.geometry_mode],
        ["ΔE CMC", "Enabled" if settings.use_delta_e_cmc else "Disabled"],
        ["CMC l:c Ratio", settings.cmc_l_c_ratio],
        ["Whiteness Min", f"{settings.whiteness_min}"],
        ["Yellowness Max", f"{settings.yellowness_max}"],
        ["Metamerism Illuminants", ", ".join(settings.metamerism_illuminants)[:30] + "..."],
        ["Spectral Analysis", "Enabled" if settings.spectral_enable else "Disabled"],
        ["— ADVANCED PARAMS —", ""],
        ["LBP Points", str(settings.lbp_points)],
        ["LBP Radius", str(settings.lbp_radius)],
        ["Wavelet Type", settings.wavelet_type],
        ["Wavelet Levels", str(settings.wavelet_levels)],
        ["Defect Min Area", f"{settings.defect_min_area} px"],
        ["Saliency Strength", f"{settings.saliency_strength}"],
        ["Morph Kernel Size", f"{settings.morph_kernel_size}"],
    ]

    # Create left table
    left_table = Table(left_settings, colWidths=[1.8*inch, 1.4*inch])
    left_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), BLUE2),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('BACKGROUND', (0, 1), (-1, 1), NEUTRAL_L),  # Section headers
        ('BACKGROUND', (0, 8), (-1, 8), NEUTRAL_L),
        ('BACKGROUND', (0, 11), (-1, 11), NEUTRAL_L),
        ('BACKGROUND', (0, 17), (-1, 17), NEUTRAL_L),
        ('SPAN', (0, 1), (1, 1)),  # Span section headers
        ('SPAN', (0, 8), (1, 8)),
        ('SPAN', (0, 11), (1, 11)),
        ('SPAN', (0, 17), (1, 17)),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 7),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 5.5),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 4),
        ('TOPPADDING', (0, 0), (-1, 0), 4),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 2),
        ('TOPPADDING', (0, 1), (-1, -1), 2),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('WORDWRAP', (0, 0), (-1, -1), True),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ]))

    # Create right table
    right_table = Table(right_settings, colWidths=[1.8*inch, 1.4*inch])
    right_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), BLUE2),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('BACKGROUND', (0, 1), (-1, 1), NEUTRAL_L),  # Section headers
        ('BACKGROUND', (0, 10), (-1, 10), NEUTRAL_L),
        ('SPAN', (0, 1), (1, 1)),  # Span section headers
        ('SPAN', (0, 10), (1, 10)),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 7),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 5.5),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 4),
        ('TOPPADDING', (0, 0), (-1, 0), 4),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 2),
        ('TOPPADDING', (0, 1), (-1, -1), 2),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('WORDWRAP', (0, 0), (-1, -1), True),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ]))

    # Place tables side-by-side
    two_column_layout = Table([[left_table, right_table]], colWidths=[3.4*inch, 3.4*inch])
    two_column_layout.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING', (0, 0), (-1, -1), 0),
        ('RIGHTPADDING', (0, 0), (-1, -1), 0),
    ]))

    elements.append(two_column_layout)
    elements.append(Spacer(1, 12))

    # Input Images (smaller size for compact layout)
    elements.append(PageBreak())
    elements.append(Paragraph("Input Images", StyleHeading))

    # Reference image
    ref_temp = "temp_ref_tech.png"
    Image.fromarray(ref).save(ref_temp)
    ref_img = RLImage(ref_temp, width=3*inch, height=3*inch)
    elements.append(Paragraph("<b>Reference Image:</b>", StyleBody))
    elements.append(ref_img)
    elements.append(Spacer(1, 8))

    # Test image
    test_temp = "temp_test_tech.png"
    Image.fromarray(test).save(test_temp)
    test_img = RLImage(test_temp, width=3*inch, height=3*inch)
    elements.append(Paragraph("<b>Sample Image:</b>", StyleBody))
    elements.append(test_img)
    elements.append(Spacer(1, 8))

    # File info
    elements.append(Paragraph(f"<b>Reference File:</b> {os.path.basename(ref_path)}", StyleBody))
    elements.append(Paragraph(f"<b>Sample File:</b> {os.path.basename(test_path)}", StyleBody))
    elements.append(Paragraph(f"<b>Image Size:</b> {ref.shape[1]}×{ref.shape[0]} pixels", StyleBody))

    # Footer note
    elements.append(Spacer(1, 15))
    elements.append(Paragraph("<i>This is a technical report for internal use by technicians. Contains detailed analysis configuration.</i>",
                             ParagraphStyle('TechFootnote', parent=StyleBody, fontSize=6, textColor=colors.grey, alignment=1)))

    # Build PDF with simplified header/footer
    def tech_header_footer(canvas, doc):
        canvas.saveState()

        # Header line
        canvas.setStrokeColor(BLUE2)
        canvas.setLineWidth(1)
        canvas.line(30, A4[1] - 40, A4[0] - 30, A4[1] - 40)

        # Footer
        canvas.setFont('Helvetica', 7)
        canvas.setFillColor(colors.grey)
        canvas.drawString(30, 30, f"Analysis Settings Report | {settings.operator_name}")
        canvas.drawRightString(A4[0] - 30, 30, f"Page {doc.page}")

        # Logo in footer (if available)
        if os.path.exists(logo_path):
            try:
                canvas.drawImage(logo_path, A4[0] - 60, 20, width=24, height=12, preserveAspectRatio=True, mask='auto')
            except:
                pass

        canvas.restoreState()

    doc.build(elements, onFirstPage=tech_header_footer, onLaterPages=tech_header_footer)

    # Clean up temp files
    if os.path.exists(ref_temp):
        os.remove(ref_temp)
    if os.path.exists(test_temp):
        os.remove(test_temp)

    return pdf_path

# ----------------------------
# Interactive Workflow with Advanced Settings
# ----------------------------
def main():
    """
    Main interactive workflow for textile quality control.

    Provides an interactive interface for:
    - Image upload
    - Settings configuration
    - Analysis execution
    - Report generation
    """
    try:
        # Step 1: Upload images
        logger.info("Starting Textile Quality Control System")
        print("=" * 60)
        print("🎨 TEXTILE QUALITY CONTROL SYSTEM")
        print("=" * 60)

        try:
            ref_path, test_path = upload_two_images()
            ref = read_rgb(ref_path)
            test = read_rgb(test_path)
            ref, test = to_same_size(ref, test)
        except Exception as e:
            logger.error(f"Failed to load images: {str(e)}")
            print(f"\n❌ Error loading images: {str(e)}")
            print("Please check your files and try again.")
            return

        logger.info(f"Images loaded: {ref.shape[1]}x{ref.shape[0]} pixels")
        print("\n✅ Images uploaded successfully!")
        print(f"   Reference: {os.path.basename(ref_path)} ({ref.shape[1]}x{ref.shape[0]})")
        print(f"   Sample: {os.path.basename(test_path)} ({test.shape[1]}x{test.shape[0]})")
    except Exception as e:
        logger.error(f"Critical error in main(): {str(e)}")
        print(f"\n❌ Critical error: {str(e)}")
        return

    # Initialize default settings
    settings = QCSettings()
    settings_ui_widgets = None

    # Create output areas
    settings_output = Output()
    processing_output = Output()

    # Step 2: Show action buttons
    clear_output(wait=False)

    # Professional header
    header_html = """
    <div style='background: linear-gradient(135deg, #2980B9 0%, #3498DB 100%);
                padding: 30px; border-radius: 15px; text-align: center; margin: 20px 0;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
        <h1 style='color: white; margin: 0; font-family: Arial, sans-serif; font-size: 32px;'>
            🎨 Textile Quality Control System
        </h1>
        <p style='color: #ecf0f1; margin: 10px 0 0 0; font-size: 16px;'>
            Professional Color & Pattern Analysis
        </p>
    </div>
    """

    info_html = f"""
    <div style='background: #f8f9fa; padding: 20px; border-radius: 10px; margin: 20px 0;
                border-left: 5px solid #27AE60;'>
        <h3 style='margin-top: 0; color: #27AE60;'>✅ Images Loaded Successfully</h3>
        <p style='margin: 5px 0;'><strong>Reference:</strong> {os.path.basename(ref_path)} ({ref.shape[1]}×{ref.shape[0]} pixels)</p>
        <p style='margin: 5px 0;'><strong>Sample:</strong> {os.path.basename(test_path)} ({test.shape[1]}×{test.shape[0]} pixels)</p>
    </div>
    """

    display(HTMLWidget(value=header_html))
    display(HTMLWidget(value=info_html))

    # Action buttons
    btn_start = Button(
        description='🚀 Start Processing',
        button_style='success',
        layout=Layout(width='250px', height='50px'),
        style={'button_color': '#27AE60', 'font_weight': 'bold'}
    )

    btn_advanced = Button(
        description='⚙️ Advanced Settings',
        button_style='info',
        layout=Layout(width='250px', height='50px'),
        style={'button_color': '#2980B9', 'font_weight': 'bold'}
    )

    btn_report_sections = Button(
        description='📋 Report Sections',
        button_style='warning',
        layout=Layout(width='250px', height='50px'),
        style={'button_color': '#F39C12', 'font_weight': 'bold'}
    )

    buttons_box = HBox(
        [btn_start, btn_advanced, btn_report_sections],
        layout=Layout(justify_content='center', margin='20px 0')
    )

    display(buttons_box)
    display(settings_output)
    display(processing_output)

    # Handler for Advanced Settings button
    def on_advanced_clicked(b):
        nonlocal settings_ui_widgets
        with settings_output:
            settings_output.clear_output()

            # Create settings UI
            panel, widgets_dict = create_advanced_settings_ui(settings, ref, test)
            settings_ui_widgets = widgets_dict

            # Create Apply and Process button
            btn_apply_process = Button(
                description='✅ Apply Settings & Start Processing',
                button_style='success',
                layout=Layout(width='400px', height='50px', margin='20px 0'),
                style={'button_color': '#27AE60', 'font_weight': 'bold'}
            )

            def on_apply_process_clicked(b):
                # Extract settings from widgets
                extract_settings_from_widgets(settings, settings_ui_widgets)

                with processing_output:
                    processing_output.clear_output()
                    print("⏳ Processing with custom settings...")

                # Hide buttons and settings
                buttons_box.layout.display = 'none'
                settings_output.clear_output()

                # Run pipeline
                run_analysis(ref_path, test_path, ref, test, settings, processing_output)

            btn_apply_process.on_click(on_apply_process_clicked)

            display(panel)
            display(HBox([btn_apply_process], layout=Layout(justify_content='center')))

    # Handler for Start Processing button (with defaults)
    def on_start_clicked(b):
        buttons_box.layout.display = 'none'
        with processing_output:
            processing_output.clear_output()
            print("⏳ Processing with default settings...")

        run_analysis(ref_path, test_path, ref, test, settings, processing_output)

    # Handler for Report Sections button
    def on_report_sections_clicked(b):
        with settings_output:
            settings_output.clear_output()
            panel = create_report_sections_ui(settings)
            display(panel)

    btn_advanced.on_click(on_advanced_clicked)
    btn_report_sections.on_click(on_report_sections_clicked)
    btn_start.on_click(on_start_clicked)

def run_analysis(ref_path, test_path, ref, test, settings, output_widget):
    """
    Run the analysis pipeline and display results.

    Args:
        ref_path: Path to reference image
        test_path: Path to test image
        ref: Reference image array
        test: Test image array
        settings: QCSettings object
        output_widget: Output widget for displaying results
    """
    with output_widget:
        output_widget.clear_output(wait=True)

        # Show progress
        progress_html = """
        <div style='background: linear-gradient(135deg, #3498DB 0%, #2980B9 100%);
                    padding: 20px; border-radius: 10px; text-align: center; margin: 20px 0;'>
            <h2 style='color: white; margin: 0;'>⚙️ Analysis in Progress...</h2>
            <p style='color: #ecf0f1; margin: 10px 0 0 0;'>
                Performing color and pattern analysis. This may take a moment.
            </p>
        </div>
        """
        display(HTMLWidget(value=progress_html))

        # Run pipeline
        try:
            logger.info("Starting analysis pipeline")
            pdf_file = run_pipeline_and_build_pdf(ref_path, test_path, ref, test, settings)

            # Generate Analysis Settings Technical Report
            logger.info("Generating technical settings report")
            tech_pdf_file = generate_analysis_settings_report(ref_path, test_path, ref, test, settings)

            # Clear and show download buttons
            output_widget.clear_output(wait=True)

            # Encode main report
            with open(pdf_file, "rb") as f:
                b64_main = base64.b64encode(f.read()).decode("utf-8")
            download_name_main = os.path.basename(pdf_file)

            # Encode technical report
            with open(tech_pdf_file, "rb") as f:
                b64_tech = base64.b64encode(f.read()).decode("utf-8")
            download_name_tech = os.path.basename(tech_pdf_file)

            success_html = f"""
            <div style='background: linear-gradient(135deg, #27AE60 0%, #2ECC71 100%);
                        padding: 30px; border-radius: 15px; text-align: center; margin: 20px 0;
                        box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
                <h2 style='color: white; margin: 0; font-size: 28px;'>✅ Analysis Complete!</h2>
                <p style='color: #ecf0f1; margin: 15px 0; font-size: 16px;'>
                    Your comprehensive quality control reports have been generated.
                </p>
                <div style='margin-top: 25px; display: flex; justify-content: center; gap: 15px; flex-wrap: wrap;'>
                    <a download="{download_name_main}"
                       href="data:application/pdf;base64,{b64_main}"
                       style="background: white; color: #27AE60; padding: 15px 35px;
                              border-radius: 25px; text-decoration: none;
                              font-family: Arial, sans-serif; font-size: 18px; font-weight: bold;
                              display: inline-block; box-shadow: 0 2px 4px rgba(0,0,0,0.2);
                              transition: transform 0.2s;">
                        📥 Download Report
                    </a>
                    <a download="{download_name_tech}"
                       href="data:application/pdf;base64,{b64_tech}"
                       style="background: white; color: #F39C12; padding: 15px 35px;
                              border-radius: 25px; text-decoration: none;
                              font-family: Arial, sans-serif; font-size: 16px; font-weight: bold;
                              display: inline-block; box-shadow: 0 2px 4px rgba(0,0,0,0.2);
                              transition: transform 0.2s;">
                        ⚙️ Download Analysis Settings Report
                    </a>
                </div>
                <p style='color: #ecf0f1; margin: 15px 0 0 0; font-size: 14px;'>
                    Main Report: {download_name_main}<br/>
                    Technical Settings Report: {download_name_tech}
                </p>
            </div>
            """
            display(HTML(success_html))

        except Exception as e:
            output_widget.clear_output(wait=True)
            error_html = f"""
            <div style='background: #E74C3C; padding: 20px; border-radius: 10px;
                        text-align: center; margin: 20px 0;'>
                <h2 style='color: white; margin: 0;'>❌ Error</h2>
                <p style='color: white; margin: 10px 0;'>
                    An error occurred during processing:<br/>
                    <code style='background: rgba(0,0,0,0.2); padding: 5px 10px;
                                 border-radius: 5px; display: inline-block; margin-top: 10px;'>
                        {str(e)}
                    </code>
                </p>
            </div>
            """
            display(HTML(error_html))
            raise

# ----------------------------
# Run the main workflow
# ----------------------------
main()
