# Textile Quality Control System v1.1.0

Professional web application for textile color and pattern analysis with comprehensive PDF reporting.

## Features

- 🎨 Advanced color analysis (ΔE76, ΔE94, ΔE2000, ΔE CMC)
- 📐 Pattern analysis (SSIM, symmetry, edge definition)
- 🔬 Spectrophotometer simulation
- 📊 Comprehensive PDF reports
- ⚙️ Customizable analysis settings
- 🌐 Modern web interface with drag-and-drop upload
- 📱 Responsive design

## Quick Start

### Local Development

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Application**
   ```bash
   python app.py
   ```

3. **Access the Application**
   Open your browser and navigate to `http://localhost:5000`

### Production Deployment (Render)

1. **Push to GitHub**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin YOUR_GITHUB_REPO_URL
   git push -u origin main
   ```

2. **Deploy on Render**
   - Go to [Render Dashboard](https://dashboard.render.com/)
   - Click "New +" → "Web Service"
   - Connect your GitHub repository
   - Configure:
     - **Name**: textile-qc-system
     - **Environment**: Python 3
     - **Build Command**: `pip install -r requirements.txt`
     - **Start Command**: `gunicorn app:app`
   - Click "Create Web Service"

3. **Environment Variables** (Optional)
   - `FLASK_ENV`: `production`
   - `SECRET_KEY`: Generate a secure random key

## Project Structure

```
SPECTROPHOTOMETER/
├── app/                      # Application modules
│   ├── core/                # Core configuration
│   ├── models/              # Data models
│   ├── services/            # Analysis services
│   ├── utils/               # Utility functions
│   ├── visualization/       # Chart generation
│   ├── report/              # PDF generation
│   └── web/                 # Web handlers
├── static/                  # Static assets
│   ├── css/                # Stylesheets
│   ├── js/                 # JavaScript
│   └── images/             # Images and logos
├── templates/               # HTML templates
├── data/                    # Data directories
│   ├── uploads/            # Uploaded images
│   └── temp/               # Temporary files
├── app.py                   # Main Flask application
├── colabcoderefernce.py    # Original analysis code (PDF generation)
├── requirements.txt         # Python dependencies
├── Procfile                # Render deployment config
└── README.md               # This file
```

## Usage

1. **Upload Images**
   - Click on the upload boxes to select your reference and test images
   - Supported formats: JPG, PNG, BMP, TIFF

2. **Configure Settings** (Optional)
   - Click "Advanced Settings" to customize analysis parameters
   - Adjust color thresholds, pattern parameters, and report sections
   - Changes are applied when you close the modal

3. **Run Analysis**
   - Click "Start Analysis" button
   - Wait for processing (usually 30-60 seconds)
   - Download both PDF reports when complete

4. **View Reports**
   - **Main Report**: Comprehensive analysis results
   - **Settings Report**: Technical parameters used

## Configuration

### Color Analysis Settings
- ΔE Threshold (PASS): Default 2.0
- ΔE Conditional: Default 3.5
- Observer Angle: 2° or 10°
- Geometry Mode: d/8 SCI, d/8 SCE, or 45/0

### Pattern Analysis Settings
- SSIM PASS Threshold: Default 0.95
- SSIM Conditional: Default 0.90
- Edge Definition, Symmetry, Repeat Period

### Report Sections
Enable/disable specific sections to optimize processing time:
- Color Unit
- Pattern Unit
- Pattern Repetition Unit
- Spectrophotometer Simulation

## Technical Details

### Technologies
- **Backend**: Flask (Python)
- **Image Processing**: OpenCV, scikit-image
- **Scientific Computing**: NumPy, SciPy
- **PDF Generation**: ReportLab
- **Deployment**: Gunicorn, Render

### Analysis Methods
- Color Space Conversions: sRGB → XYZ → Lab*
- Delta E Formulas: CIE76, CIE94, CIEDE2000, CMC
- Chromatic Adaptation: Bradford transform
- Pattern Metrics: SSIM, FFT analysis, texture features

## Troubleshooting

### Common Issues

**Images not uploading**
- Check file format (must be JPG, PNG, BMP, or TIFF)
- Ensure file size is under 50MB
- Try a different browser

**Analysis fails**
- Verify both images are uploaded
- Check that images are valid (not corrupted)
- Try with smaller image sizes

**Reports not downloading**
- Check browser pop-up blocker settings
- Ensure sufficient disk space
- Try a different browser

### Logs

Check application logs for detailed error messages:
```bash
# Local development
Check terminal output

# Render deployment
View logs in Render dashboard
```

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review application logs
3. Verify all dependencies are installed correctly

## License

Proprietary - Textile Engineering Solutions

## Version History

### v1.1.0 (Current)
- Modular architecture with Flask web framework
- Modern responsive web interface
- Drag-and-drop image upload
- Interactive settings modal
- Real-time loading indicators
- Optimized processing with conditional execution
- Deployment-ready for Render

### v1.0.0
- Initial Google Colab version
- Monolithic code structure
- Command-line interface

