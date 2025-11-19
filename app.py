"""Main Flask Application for Textile QC System."""

import os
import sys
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
import json
import logging

# Add project root to path to import from colabcoderefernce
sys.path.insert(0, os.path.dirname(__file__))

# Import core modules
from app.core.config import Config, DevelopmentConfig, ProductionConfig
from app.core.settings import QCSettings
from app.utils.image_io import read_rgb, validate_image_file
from app.utils.image_processing import to_same_size, apply_crop
from app.utils.helpers import get_local_time

# Import original report generation functions (keeping PDF output exactly the same)
from colabcoderefernce import run_pipeline_and_build_pdf, generate_analysis_settings_report

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Load configuration based on environment
if os.getenv('FLASK_ENV') == 'production':
    app.config.from_object(ProductionConfig)
else:
    app.config.from_object(DevelopmentConfig)

# Ensure upload directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['TEMP_FOLDER'], exist_ok=True)

def allowed_file(filename):
    """Check if file has allowed extension."""
    from app.core.config import ALLOWED_EXTENSIONS
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Render main page."""
    return render_template('index.html')

@app.route('/api/analyze', methods=['POST'])
def analyze():
    """
    Main analysis endpoint.
    
    Expects:
        - reference_image: File
        - test_image: File  
        - settings: JSON string (optional)
    
    Returns:
        JSON with download URLs for both PDFs
    """
    try:
        # Validate file uploads
        if 'reference_image' not in request.files or 'test_image' not in request.files:
            return jsonify({'error': 'Both reference and test images are required'}), 400
        
        ref_file = request.files['reference_image']
        test_file = request.files['test_image']
        
        if ref_file.filename == '' or test_file.filename == '':
            return jsonify({'error': 'No files selected'}), 400
        
        if not (allowed_file(ref_file.filename) and allowed_file(test_file.filename)):
            return jsonify({'error': 'Invalid file format'}), 400
        
        # Save uploaded files
        ref_filename = secure_filename(ref_file.filename)
        test_filename = secure_filename(test_file.filename)
        
        ref_path = os.path.join(app.config['UPLOAD_FOLDER'], ref_filename)
        test_path = os.path.join(app.config['UPLOAD_FOLDER'], test_filename)
        
        ref_file.save(ref_path)
        test_file.save(test_path)
        
        logger.info(f"Files uploaded: {ref_filename}, {test_filename}")
        
        # Parse settings
        settings = QCSettings()
        if 'settings' in request.form:
            try:
                settings_dict = json.loads(request.form['settings'])
                settings = QCSettings.from_dict(settings_dict)
                logger.info("Custom settings loaded")
            except Exception as e:
                logger.warning(f"Error parsing settings, using defaults: {e}")
        
        # Load and validate images
        try:
            ref_img = read_rgb(ref_path)
            test_img = read_rgb(test_path)
            ref_img, test_img = to_same_size(ref_img, test_img)
            
            # Apply crop if enabled
            if settings.use_crop:
                ref_img = apply_crop(ref_img, settings)
                test_img = apply_crop(test_img, settings)
            
            logger.info(f"Images loaded: {ref_img.shape}")
        except Exception as e:
            return jsonify({'error': f'Error loading images: {str(e)}'}), 400
        
        # Generate reports using original code (keeps PDF exactly the same)
        try:
            logger.info("Starting PDF generation...")
            
            main_report_path = run_pipeline_and_build_pdf(
                ref_path, test_path, ref_img, test_img, settings
            )
            
            tech_report_path = generate_analysis_settings_report(
                ref_path, test_path, ref_img, test_img, settings
            )
            
            logger.info("Reports generated successfully")
            
            # Return success with file information
            return jsonify({
                'success': True,
                'main_report': os.path.basename(main_report_path),
                'tech_report': os.path.basename(tech_report_path),
                'main_report_path': main_report_path,
                'tech_report_path': tech_report_path,
                'message': 'Analysis completed successfully'
            })
            
        except Exception as e:
            logger.error(f"Error generating reports: {e}", exc_info=True)
            return jsonify({'error': f'Error generating reports: {str(e)}'}), 500
        
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500

@app.route('/api/download/<report_type>/<filename>')
def download_report(report_type, filename):
    """Download generated report."""
    try:
        # Security: only allow downloading from /content directory
        if report_type == 'main':
            filepath = os.path.join('/content', filename)
        else:
            filepath = os.path.join(os.getcwd(), filename)
        
        if not os.path.exists(filepath):
            return jsonify({'error': 'File not found'}), 404
        
        return send_file(
            filepath,
            mimetype='application/pdf',
            as_attachment=True,
            download_name=filename
        )
    except Exception as e:
        logger.error(f"Error downloading file: {e}")
        return jsonify({'error': 'Error downloading file'}), 500

@app.route('/api/settings/defaults', methods=['GET'])
def get_default_settings():
    """Get default QC settings as JSON."""
    settings = QCSettings()
    return jsonify(settings.to_dict())

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'version': '1.1.0',
        'service': 'Textile QC Analysis'
    })

# Error handlers
@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=app.config['DEBUG'])

