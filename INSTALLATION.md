# Installation Instructions for Enhanced Features

## Required Dependencies

To enable all features including PDF report generation, install the following packages:

```bash
pip install reportlab pyyaml
```

### Package Details:
- **reportlab**: Professional PDF generation library
- **pyyaml**: YAML configuration file parsing (for clinical thresholds)

## Installation Steps

1. **Activate your virtual environment** (if using one):
   ```bash
   # On Windows
   torch\Scripts\activate
   
   # On Linux/Mac
   source torch/bin/activate
   ```

2. **Install dependencies**:
   ```bash
   pip install reportlab pyyaml
   ```

3. **Verify installation**:
   ```bash
   python -c "import reportlab; import yaml; print('✓ All dependencies installed')"
   ```

## Features Status

### ✅ Currently Working (No Additional Dependencies):
- Clinical Decision Support System
- Risk Assessment Engine
- Clinical Recommendations
- Enhanced CSV Reports with risk levels
- Growth pattern detection (IUGR, AGA, macrosomia, microcephaly)
- Follow-up planning

### 📄 Requires `reportlab` (Optional):
- Professional PDF Clinical Reports
- Multi-page formatted reports
- Charts and graphs in PDF
- Custom hospital letterhead

## Running the Application

### Without PDF Support:
```bash
python web_app/app.py
```
The app will run normally with CSV reports. You'll see a warning:
```
⚠️  reportlab not installed. PDF reports disabled. Install with: pip install reportlab pyyaml
```

### With Full PDF Support:
```bash
# Install dependencies first
pip install reportlab pyyaml

# Run the app
python web_app/app.py
```

## What's New

### Clinical Decision Support:
- **Automated Risk Assessment**: Classifies measurements as normal, borderline, high-risk, or critical
- **Growth Pattern Detection**: Identifies IUGR, macrosomia, microcephaly, and AGA patterns
- **Clinical Recommendations**: Evidence-based recommendations based on INTERGROWTH-21st standards
- **Follow-up Planning**: Automated scheduling based on risk level

### Enhanced Reports:
- **CSV Reports**: Now include risk assessment, recommendations, and follow-up plans
- **PDF Reports** (when reportlab is installed): Professional clinical-grade PDF reports

## Troubleshooting

### Error: `ModuleNotFoundError: No module named 'reportlab'`
**Solution**: Install reportlab:
```bash
pip install reportlab pyyaml
```

### Error: `FileNotFoundError: clinical_thresholds.yaml`
**Solution**: The file should be at `c:\Projects\Zidane\config\clinical_thresholds.yaml`. Verify it exists.

### Error: `No module named 'yaml'`
**Solution**: Install PyYAML:
```bash
pip install pyyaml
```

## File Structure

```
c:\Projects\Zidane\
├── clinical_rules.py              # Risk assessment engine
├── recommendations.py             # Recommendation generator
├── report_generator.py            # PDF report generator (optional)
├── config/
│   └── clinical_thresholds.yaml   # Clinical thresholds configuration
└── web_app/
    ├── app.py                     # Flask application (updated)
    └── reports/                   # Generated PDF reports (created automatically)
```

## Next Steps

1. Install dependencies: `pip install reportlab pyyaml`
2. Restart the Flask server
3. Upload an ultrasound image
4. Download the enhanced clinical report (CSV or PDF)

Enjoy your enhanced clinical decision support system! 🏥✨
