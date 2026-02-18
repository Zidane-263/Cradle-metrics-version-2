#!/usr/bin/env python3
"""
Flask Web Application for Fetal Ultrasound Analysis
Professional medical-grade web interface
"""

import os
import sys
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import uuid
import json
import math

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from automatic_pipeline import AutomaticPipeline
from clinical_rules import ClinicalRulesEngine
from recommendations import RecommendationGenerator
from clinical_history import ClinicalHistoryManager

# Optional PDF report generator (requires reportlab)
try:
    from report_generator import ClinicalReportGenerator
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    print("⚠️  reportlab not installed. PDF reports disabled. Install with: pip install reportlab pyyaml")

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configuration
app.config['UPLOAD_FOLDER'] = str(project_root / 'web_app' / 'uploads')
app.config['RESULTS_FOLDER'] = str(project_root / 'web_app' / 'results')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'bmp', 'tif', 'tiff'}

# Create folders
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

# Initialize pipeline (lazy loading)
pipeline = None
clinical_engine = None
quality_assessor = None
rec_generator = None
report_generator = None
history_manager = None


def get_pipeline():
    """Get or create pipeline instance"""
    global pipeline
    if pipeline is None:
        # Use absolute paths
        yolo_model = str(project_root / 'runs' / 'detect' / 'fetal_detection' / 'weights' / 'best.pt')
        sam_model = str(project_root / 'sam_vit_b_01ec64.pth')
        
        pipeline = AutomaticPipeline(
            yolo_model_path=yolo_model,
            sam_checkpoint=sam_model,
            pixel_to_mm=2.5,
            enable_clinical=True
        )
    return pipeline


def get_clinical_engine():
    """Get or create clinical rules engine"""
    global clinical_engine
    if clinical_engine is None:
        clinical_engine = ClinicalRulesEngine()
    return clinical_engine


def get_rec_generator():
    """Get or create recommendation generator"""
    global rec_generator
    if rec_generator is None:
        rec_generator = RecommendationGenerator()
    return rec_generator

def get_quality_assessor():
    """Get or create quality assessor"""
    global quality_assessor
    if quality_assessor is None:
        from clinical_rules import AnatomicalQualityAssessor
        quality_assessor = AnatomicalQualityAssessor()
    return quality_assessor


def get_report_generator():
    """Get or create PDF report generator"""
    global report_generator
    if not PDF_AVAILABLE:
        return None
    if report_generator is None:
        reports_dir = str(project_root / 'web_app' / 'reports')
        report_generator = ClinicalReportGenerator(output_dir=reports_dir)
    return report_generator


def get_history_manager():
    """Get or create clinical history manager"""
    global history_manager
    if history_manager is None:
        history_manager = ClinicalHistoryManager()
    return history_manager


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


@app.route('/')
def index():
    """Landing page"""
    return render_template('landing.html')


@app.route('/analyze')
def analyze():
    """Analysis page"""
    file_id = request.args.get('file_id')
    return render_template('index.html', file_id=file_id)

@app.route('/patients')
def patients_page():
    """Render patient directory page"""
    file_id = request.args.get('file_id')
    return render_template('patients.html', file_id=file_id)


@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Handle file upload"""
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400
    
    # Generate unique filename
    file_id = str(uuid.uuid4())
    ext = file.filename.rsplit('.', 1)[1].lower()
    filename = f"{file_id}.{ext}"
    
    # Save file
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    return jsonify({
        'success': True,
        'file_id': file_id,
        'filename': filename
    })


@app.route('/api/process', methods=['POST'])
def process_image():
    """Process uploaded image"""
    
    data = request.json
    file_id = data.get('file_id')
    ga_weeks = data.get('ga_weeks')
    
    # NEW: Advanced Clinical Inputs
    afi = data.get('afi')
    ua_pi = data.get('ua_pi')
    mca_pi = data.get('mca_pi')
    
    if not file_id:
        return jsonify({'error': 'No file_id provided'}), 400
    
    # Find file
    upload_folder = Path(app.config['UPLOAD_FOLDER'])
    files = list(upload_folder.glob(f"{file_id}.*"))
    
    if not files:
        return jsonify({'error': 'File not found'}), 404
    
    filepath = files[0]
    
    try:
        # Get pipeline
        pipe = get_pipeline()
        
        # Update GA if provided
        if ga_weeks:
            pipe.ga_weeks = float(ga_weeks)
        
        # Initialize clinical variables
        analysis_data = {}
        percentiles = {}
        assessment = {}
        recommendations = []
        
        # NEW: Measure actual execution time
        import time
        start_time = time.time()
        
        # Process image
        results_dir = os.path.join(app.config['RESULTS_FOLDER'], file_id)
        os.makedirs(results_dir, exist_ok=True)
        
        # Save a copy of the original image to the results folder for easy access
        import shutil
        original_copy_path = os.path.join(results_dir, f"original{filepath.suffix}")
        shutil.copy2(filepath, original_copy_path)
        
        results = pipe.process_image(
            str(filepath),
            output_dir=results_dir
        )
        
        # Extract results
        response = {
            'success': True,
            'file_id': file_id,
            'detections': []
        }
        
        # Add detections
        for det in results['detections']:
            response['detections'].append({
                'label': det['label'],
                'confidence': round(det['confidence'], 3)
            })
        
        # Add measurements
        metrics = results['segmentation']['metrics']
        spatial = metrics.get('spatial_metrics', {})
        
        measurements = {}
        if 'head_circumference' in spatial:
            measurements['HC'] = round(spatial['head_circumference'], 1)
        if 'abdomen_circumference' in spatial:
            measurements['AC'] = round(spatial['abdomen_circumference'], 1)
        if 'biparietal_diameter' in spatial:
            measurements['BPD'] = round(spatial['biparietal_diameter'], 1)
        if 'femur_length' in spatial:
            measurements['FL'] = round(spatial['femur_length'], 1)
        
        # Add limb lengths
        if 'limb_lengths' in spatial:
            for limb, length in spatial['limb_lengths'].items():
                label = limb.upper() if len(limb) <= 3 else limb.capitalize()
                measurements[label] = round(length, 1)
        
        # Add geometric/aspect metrics
        if 'head_aspect_ratio' in spatial:
            measurements['head_aspect_ratio'] = spatial['head_aspect_ratio']
        if 'abdomen_aspect_ratio' in spatial:
            measurements['abdomen_aspect_ratio'] = spatial['abdomen_aspect_ratio']
            
        response['measurements'] = measurements
        response['unit'] = metrics.get('unit', 'mm') # Default to mm for professional view
        
        # Add clinical assessment
        if pipe.enable_clinical and measurements:
            clinical = pipe.clinical_assessor.assess_all_measurements(
                measurements, pipe.ga_weeks
            )
            
            response['clinical'] = {
                'estimated_ga': clinical['consensus_ga'],
                'ga_uncertainty': clinical['ga_uncertainty'],
                'ga_consistency': clinical['ga_consistency'],
                'growth_status': clinical['overall_assessment']['status'],
                'flags': clinical['overall_assessment']['flags']
            }
            
            # Add percentiles
            percentiles = {}
            for metric, data in clinical['measurements'].items():
                percentiles[metric] = {
                    'percentile': round(data['percentile'], 1),
                    'classification': data['classification'],
                    'flag': data['flag']
                }
            response['percentiles'] = percentiles
        
        # Find result image
        result_images = list(Path(results_dir).glob('*_result.png'))
        if result_images:
            response['result_image'] = f"/api/results/{file_id}/{result_images[0].name}"
        
        # Enhanced Clinical Assessment using new rules engine
        if measurements:
            # Prepare data for clinical engine
            analysis_data = {}
            for key, value in measurements.items():
                perc_data = percentiles.get(key, {})
                analysis_data[key] = {
                    'value': value,
                    'percentile': perc_data.get('percentile') if perc_data else None
                }
            
            # Add GA
            if 'clinical' in response:
                analysis_data['GA'] = {'value': response['clinical']['estimated_ga']}
            
            # Add AFI and Doppler to analysis_data
            if afi is not None:
                analysis_data['AFI'] = {'value': float(afi)}
            if ua_pi is not None:
                analysis_data['UA_PI'] = {'value': float(ua_pi)}
            if mca_pi is not None:
                analysis_data['MCA_PI'] = {'value': float(mca_pi)}
            
            # Generate comprehensive assessment
            engine = get_clinical_engine()
            assessment = engine.generate_comprehensive_assessment(analysis_data)
            
            # Generate recommendations
            rec_gen = get_rec_generator()
            recommendations = rec_gen.generate_recommendations(assessment)
            report_data = rec_gen.format_for_report(assessment, recommendations)
            
            # Add to response
            response['risk_assessment'] = {
                'overall_risk': assessment['overall_risk'],
                'risk_color': assessment.get('measurements', {}).get('HC', {}).get('color', '#10b981'),
                'growth_pattern': assessment.get('growth_pattern', {}),
                'alerts': assessment.get('alerts', []),
                'priority': assessment.get('priority', 0),
                'efw': assessment.get('efw'),
                'ci': assessment.get('ci'),
                'afi': assessment.get('afi'),
                'doppler': assessment.get('doppler')
            }
            
            # Anatomical Quality Assessment
            q_assessor = get_quality_assessor()
            q_results = q_assessor.assess_quality(response)
            response['quality_score'] = q_results
            
            response['recommendations'] = recommendations
            response['clinical_summary'] = report_data['summary']
            response['follow_up'] = report_data['follow_up']
            
        # Add actual processing time
        response['processing_time'] = round(time.time() - start_time, 2)
            
        # Growth History & Velocity
        patient_id = data.get('patient_id', 'default_patient')
        history_mgr = get_history_manager()
        
        # Save this record
        history_mgr.save_record(patient_id, response)
        
        # Calculate and add velocity
        velocity_data = history_mgr.calculate_velocity(patient_id, response)
        if velocity_data:
            response['growth_velocity'] = velocity_data
            
        # Attach quality score and processing time to assessment for report consistency
        assessment['quality_score'] = response.get('quality_score', {})
        assessment['processing_time'] = response.get('processing_time', 0)
        
        # Store assessment for PDF generation
        response['_assessment'] = assessment  # Internal use only
        
        # Save results to JSON for report generation
        with open(os.path.join(results_dir, 'results.json'), 'w') as f:
            json.dump(response, f, default=str)
            
        return jsonify(response)
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/results/<file_id>/<filename>')
def get_result(file_id, filename):
    """Serve result images"""
    results_dir = os.path.join(app.config['RESULTS_FOLDER'], file_id)
    return send_from_directory(results_dir, filename)


@app.route('/report/<file_id>')
def report_preview(file_id):
    """Render aesthetic report preview page"""
    results_dir = Path(app.config['RESULTS_FOLDER']) / file_id
    json_path = results_dir / 'results.json'
    
    if not json_path.exists():
        return "Report data not found", 404
        
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    return render_template('report_preview.html', data=data, file_id=file_id)

@app.route('/api/results/<file_id>')
def get_results_data(file_id):
    """API endpoint to get analysis results as JSON"""
    results_dir = Path(app.config['RESULTS_FOLDER']) / file_id
    json_path = results_dir / 'results.json'
    
    if not json_path.exists():
        return jsonify({'success': False, 'error': 'Results not found'}), 404
        
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    return jsonify({'success': True, 'data': data})

@app.route('/api/history/<patient_id>')
def get_patient_history(patient_id):
    """API endpoint to get scan history for a patient"""
    history_mgr = get_history_manager()
    history = history_mgr.get_patient_history(patient_id)
    return jsonify({
        'success': True,
        'patient_id': patient_id,
        'history': history,
        'count': len(history)
    })


@app.route('/api/patients')
def list_patients():
    """List all patients in history"""
    history_mgr = get_history_manager()
    patients = history_mgr.get_all_patients()
    return jsonify({'patients': patients})

@app.route('/api/trends/<patient_id>/<metric>')
def get_trends(patient_id, metric):
    """Get trend data and reference percentile curves"""
    history_mgr = get_history_manager()
    patient_data = history_mgr.get_trend_data(patient_id, metric)
    
    # Generate reference curves (10th, 50th, 90th)
    from utils.intergrowth21 import Intergrowth21
    calc = Intergrowth21()
    
    reference_curves = {
        '10th': [],
        '50th': [],
        '90th': []
    }
    
    # Range of weeks for reference (14 to 40)
    for ga in range(14, 41):
        try:
            if metric.upper() in ['EFW']:
                # For EFW use the Hadlock formula in clinical_rules
                from clinical_rules import ClinicalRulesEngine
                engine = ClinicalRulesEngine()
                # Simplified reference for EFW if ga is provided
                mean_ln_efw = 0.578 + (0.332 * ga) - (0.00354 * (ga ** 2))
                mean_efw = math.exp(mean_ln_efw)
                sd = mean_efw * 0.15
                reference_curves['50th'].append({'x': ga, 'y': round(mean_efw, 1)})
                reference_curves['10th'].append({'x': ga, 'y': round(mean_efw - 1.28 * sd, 1)})
                reference_curves['90th'].append({'x': ga, 'y': round(mean_efw + 1.28 * sd, 1)})
            elif metric.upper() in calc.valid_metrics:
                expected = calc.get_expected_value(ga, metric.upper())
                sd = calc.sd[metric.upper()]
                reference_curves['50th'].append({'x': ga, 'y': round(expected, 1)})
                reference_curves['10th'].append({'x': ga, 'y': round(expected - 1.28 * sd, 1)})
                reference_curves['90th'].append({'x': ga, 'y': round(expected + 1.28 * sd, 1)})
        except:
            continue
            
    # Filter out non-finite values (NaN, Inf) and sort by GA
    patient_data = [p for p in patient_data if math.isfinite(p['value'])]
    patient_data.sort(key=lambda x: x['ga'])
    
    return jsonify({
        'patient_data': [{'x': p['ga'], 'y': p['value'], 'date': p['date']} for p in patient_data],
        'reference': reference_curves,
        'metric': metric,
        'unit': 'g' if metric.upper() == 'EFW' else 'mm'
    })

@app.route('/api/report/<file_id>')
def generate_report(file_id):
    """Generate and serve clinical report (PDF preferred, CSV fallback)"""
    results_dir = Path(app.config['RESULTS_FOLDER']) / file_id
    json_path = results_dir / 'results.json'
    
    if not json_path.exists():
        return jsonify({'error': 'Results not found'}), 404
        
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Try PDF generation first
    if PDF_AVAILABLE:
        try:
            report_gen = get_report_generator()
            if report_gen:
                # Prepare data for PDF generator
                # Reconstruct analysis_data to match what report_generator expects
                measurements = data.get('measurements', {})
                percentiles = data.get('percentiles', {})
                analysis_data = {}
                
                for key, value in measurements.items():
                    perc_data = percentiles.get(key, {})
                    analysis_data[key] = {
                        'value': value,
                        'percentile': perc_data.get('percentile') if perc_data else None
                    }
                
                # Add GA if present
                clinical = data.get('clinical', {})
                if 'estimated_ga' in clinical:
                    analysis_data['GA'] = {'value': clinical['estimated_ga']}

                # Add processing time to analysis_data for PDF
                analysis_data['processing_time'] = data.get('processing_time', '---')

                assessment = data.get('_assessment', {})
                # Ensure quality_score is present in assessment if it exists at top level
                if 'quality_score' in data and 'quality_score' not in assessment:
                    assessment['quality_score'] = data['quality_score']
                
                recommendations = data.get('recommendations', [])
                
                # Generate PDF
                pdf_path = report_gen.generate_report(
                    analysis_data, 
                    assessment, 
                    recommendations,
                    output_filename=f"CradleMetrics_Report_{file_id}.pdf"
                )
                
                return send_from_directory(
                    os.path.dirname(pdf_path), 
                    os.path.basename(pdf_path),
                    as_attachment=True
                )
        except Exception as e:
            print(f"⚠️ PDF generation failed: {str(e)}. Falling back to CSV.")
    
    # Fallback to CSV
    import csv
    import io
    from flask import make_response
    
    output = io.StringIO()
    writer = csv.writer(output)
    
    # ... rest of the existing CSV generation logic ...
    writer.writerow(['Fetal Ultrasound Clinical Report'])
    writer.writerow(['ID', file_id])
    writer.writerow([])
    writer.writerow(['Biometric Measurements'])
    writer.writerow(['Parameter', 'Value', 'Unit', 'Percentile', 'Classification'])
    
    measurements = data.get('measurements', {})
    percentiles = data.get('percentiles', {})
    unit = data.get('unit', 'mm')
    
    for key, value in measurements.items():
        perc = percentiles.get(key, {})
        writer.writerow([
            key, 
            value, 
            unit, 
            perc.get('percentile', 'N/A'), 
            perc.get('classification', 'N/A')
        ])
    
    writer.writerow([])
    writer.writerow(['Clinical Assessment'])
    clinical = data.get('clinical', {})
    writer.writerow(['Estimated GA', f"{clinical.get('estimated_ga', 'N/A')} weeks"])
    writer.writerow(['GA Consistency', clinical.get('ga_consistency', 'N/A')])
    writer.writerow(['Growth Status', clinical.get('growth_status', 'N/A')])
    
    if 'risk_assessment' in data:
        writer.writerow([])
        writer.writerow(['Risk Assessment'])
        risk = data['risk_assessment']
        writer.writerow(['Overall Risk Level', risk.get('overall_risk', 'N/A').upper().replace('_', ' ')])
        writer.writerow(['Growth Pattern', risk.get('growth_pattern', {}).get('pattern', 'N/A')])
        writer.writerow(['Priority', risk.get('priority', 0)])
        
        if risk.get('efw'):
            efw = risk['efw']
            writer.writerow(['Estimated Fetal Weight', f"{efw.get('value')} {efw.get('unit')}", f"{efw.get('percentile', 'N/A')}%"])
        if risk.get('ci'):
            ci = risk['ci']
            writer.writerow(['Cephalic Index', f"{ci.get('value')} {ci.get('unit')}", ci.get('status', 'N/A')])
        if risk.get('afi'):
            afi = risk['afi']
            writer.writerow(['Amniotic Fluid Index', f"{afi.get('value')} {afi.get('unit')}", afi.get('classification', 'N/A')])
        if risk.get('doppler'):
            doppler = risk['doppler']
            writer.writerow(['Doppler (CPR)', doppler.get('cpr', 'N/A'), doppler.get('classification', 'N/A')])
            writer.writerow(['UA-PI', doppler.get('ua_pi', 'N/A')])
            writer.writerow(['MCA-PI', doppler.get('mca_pi', 'N/A')])
            
    # Anatomical Intelligence & Performance
    writer.writerow([])
    writer.writerow(['System Intelligence & Performance'])
    q = data.get('quality_score', {})
    writer.writerow(['Anatomical Quality Score', f"{q.get('score', 'N/A')}/100", q.get('status', 'N/A')])
    writer.writerow(['Plane Accuracy Score', q.get('plane_accuracy', 'N/A')])
    writer.writerow(['AI Confidence Score', q.get('avg_confidence', 'N/A')])
    writer.writerow(['Processing Speed', f"{data.get('processing_time', 'N/A')}s"])
            
    if 'clinical_summary' in data:
        writer.writerow([])
        writer.writerow(['Clinical Summary'])
        writer.writerow([data['clinical_summary']])
    
    if 'recommendations' in data and data['recommendations']:
        writer.writerow([])
        writer.writerow(['Clinical Recommendations'])
        writer.writerow(['Priority', 'Category', 'Recommendation'])
        for rec in data['recommendations']:
            writer.writerow([
                rec.get('priority', 'N/A').upper(),
                rec.get('category', 'N/A'),
                rec.get('text', 'N/A')
            ])
    
    if 'follow_up' in data:
        writer.writerow([])
        writer.writerow(['Follow-up Plan'])
        follow_up = data['follow_up']
        writer.writerow(['Next Scan', follow_up.get('next_scan', 'N/A')])
        writer.writerow(['Frequency', follow_up.get('frequency', 'N/A')])
        writer.writerow(['Specialist', follow_up.get('specialist', 'N/A')])
        if 'additional' in follow_up:
            writer.writerow(['Additional Tests', ', '.join(follow_up['additional'])])
    
    if clinical.get('flags'):
        writer.writerow([])
        writer.writerow(['Clinical Flags'])
        for flag in clinical['flags']:
            writer.writerow([flag])
            
    response = make_response(output.getvalue())
    response.headers["Content-Disposition"] = f"attachment; filename=CradleMetrics_Report_{file_id}.csv"
    response.headers["Content-type"] = "text/csv"
    return response


@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy'})


if __name__ == '__main__':
    print("="*70)
    print("🏥 Fetal Ultrasound Analysis - Web Interface")
    print("="*70)
    print("\n🌐 Starting server...")
    print("📍 Open your browser and go to: http://localhost:5000")
    print("\n⚠️  Press CTRL+C to stop the server\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
