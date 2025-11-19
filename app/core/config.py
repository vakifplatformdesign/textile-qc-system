"""Application configuration and constants."""

import os
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch

# Software Information
SOFTWARE_VERSION = "1.1.0"
COMPANY_NAME = "Textile Engineering Solutions"
COMPANY_SUBTITLE = "Professional Color Analysis Solutions"
REPORT_TITLE = "Color Analysis Report"

# Page Settings
PAGE_SIZE = A4
MARGIN_L = 50
MARGIN_R = 50
MARGIN_T = 50
MARGIN_B = 50
DPI = 300
FRAME_MARGIN = 9  # 3mm frame margin

# Timezone
TIMEZONE_OFFSET_HOURS = 3  # Turkey timezone offset (UTC+3)

# Colors - Brand Identity
BLUE1 = colors.HexColor("#2980B9")
BLUE2 = colors.HexColor("#3498DB")
GREEN = colors.HexColor("#27AE60")
RED = colors.HexColor("#E74C3C")
ORANGE = colors.HexColor("#F39C12")
NEUTRAL_DARK = colors.HexColor("#2C3E50")
NEUTRAL = colors.HexColor("#7F8C8D")
NEUTRAL_L = colors.HexColor("#BDC3C7")

STATUS_COLORS = {"PASS": GREEN, "FAIL": RED, "CONDITIONAL": ORANGE}

# Logo files
PRIMARY_LOGO = "llogo_square_with_name_1024x1024.png"
FALLBACK_LOGOS = ["logo_square_with_name_1024x1024.png", "logo_square_no_name_1024x1024.png"]

# File Upload Settings
MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff', 'tif'}
UPLOAD_FOLDER = 'data/uploads'
TEMP_FOLDER = 'data/temp'

# Processing Settings
MAX_IMAGE_SIZE = 10000  # Max dimension in pixels
MIN_IMAGE_SIZE = 100    # Min dimension in pixels

class Config:
    """Base configuration."""
    SECRET_KEY = os.getenv('SECRET_KEY', 'textile-qc-secret-key-change-in-production')
    DEBUG = os.getenv('DEBUG', 'False') == 'True'
    UPLOAD_FOLDER = UPLOAD_FOLDER
    TEMP_FOLDER = TEMP_FOLDER
    MAX_CONTENT_LENGTH = MAX_CONTENT_LENGTH

class ProductionConfig(Config):
    """Production configuration."""
    DEBUG = False
    SECRET_KEY = os.getenv('SECRET_KEY')  # Must be set in production

class DevelopmentConfig(Config):
    """Development configuration."""
    DEBUG = True

