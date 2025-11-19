"""Quality Control Settings dataclass."""

from dataclasses import dataclass, field
import numpy as np

@dataclass
class QCSettings:
    """Quality Control Settings for textile analysis."""
    
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
    gabor_frequencies_str: str = "0.1, 0.2, 0.3"
    gabor_num_orientations: int = 8

    # GLCM parameters
    glcm_distances: list = field(default_factory=lambda: [1, 3, 5])
    glcm_distances_str: str = "1, 3, 5"
    glcm_angles: list = field(default_factory=lambda: [0, 45, 90, 135])
    glcm_angles_str: str = "0, 45, 90, 135"

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
    pattern_min_area: int = 100
    pattern_max_area: int = 5000
    pattern_similarity_threshold: float = 0.85
    blob_min_circularity: float = 0.5
    blob_min_convexity: float = 0.8
    keypoint_detector: str = "ORB"  # "SIFT", "ORB", "AKAZE"
    grid_cell_size: int = 50
    pattern_count_tolerance: int = 5
    pattern_match_threshold: float = 0.7

    # ===== REPORT SECTIONS CONTROL =====
    # Main sections
    enable_analysis_settings: bool = False
    enable_color_unit: bool = True
    enable_pattern_unit: bool = True
    enable_pattern_repetition: bool = True
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

    def to_dict(self):
        """Convert settings to dictionary."""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, np.ndarray):
                result[key] = value.tolist() if value.size > 0 else []
            else:
                result[key] = value
        return result

    @classmethod
    def from_dict(cls, data: dict):
        """Create settings from dictionary."""
        settings = cls()
        for key, value in data.items():
            if hasattr(settings, key):
                if key in ['spectral_ref_wavelengths', 'spectral_ref_reflectance', 
                          'spectral_sample_wavelengths', 'spectral_sample_reflectance']:
                    setattr(settings, key, np.array(value) if value else np.array([]))
                else:
                    setattr(settings, key, value)
        return settings

