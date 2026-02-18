# ⚡ CradleMetrics Quick Start Guide

Welcome to **CradleMetrics**, the advanced AI-driven fetal ultrasound analysis platform. Follow these steps to get the system up and running in minutes.

## 1. Prerequisites
Ensure you have the following installed:
- **Python 3.9+**
- **CUDA-enabled GPU** (Recommended for SAM performance)
- **Git**

## 2. Installation
Clone the repository and install the dependencies:

```powershell
# Install requirements
pip install -r requirements.txt

# Install optional PDF reporting dependencies
pip install reportlab PyYAML
```

## 3. Launching the Web Interface
The easiest way to use CradleMetrics is through the professional medical dashboard.

```powershell
# Run the application
python web_app/app.py
```
*Wait for the models to load (YOLOv8 and SAM VIT-B).*

## 4. Analysis Workflow
1. **Open Dashboard**: Navigate to `http://localhost:5000` in your browser.
2. **Upload Scan**: Drag and drop a fetal ultrasound image (HC, AC, BPD, or FL planes).
3. **Set GA (Optional)**: Input the known Gestational Age for more precise clinical flags.
4. **Run Analysis**: Click **"Start Analysis"**.
5. **View Results**: 
   - Review the **Segmentation Output** on the visualization card.
   - Check the **Biometric Indices** (Measurements vs INTERGROWTH-21st percentiles).
   - Read the **Clinical Profile** for growth status and consistency.
6. **Preview & Download**:
   - Click **"Preview Report"** for an aesthetic clinical summary.
   - Click **"Download Clinical Report"** to save a permanent PDF record.

## 5. Command Line Interface (CLI)
For batch processing or testing without the UI:

```powershell
python automatic_pipeline.py --image path/to/scan.jpg --output results/
```

---
**Need help?** Check out the full [CRADLEMETRICS.md](file:///C:/Projects/Zidane/CRADLEMETRICS.md) for architectural details.
