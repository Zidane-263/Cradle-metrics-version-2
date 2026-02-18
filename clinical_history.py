import json
import os
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path

class ClinicalHistoryManager:
    """Manages persistent storage of clinical scan results for longitudinal tracking"""
    
    def __init__(self, storage_dir: str = None):
        if storage_dir is None:
            # Default to a history folder in the project root
            storage_dir = os.path.join(os.path.dirname(__file__), 'clinical_history')
        
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.index_file = self.storage_dir / 'index.json'
        self._ensure_index()

    def _ensure_index(self):
        """Ensure the index file exists"""
        if not self.index_file.exists():
            with open(self.index_file, 'w') as f:
                json.dump({'patients': {}}, f)

    def save_record(self, patient_id: str, scan_data: Dict):
        """
        Save a scan record for a patient
        
        Args:
            patient_id: Unique identifier for the patient
            scan_data: Dictionary containing measurements and assessment
        """
        record_id = scan_data.get('file_id', str(datetime.now().timestamp()))
        timestamp = datetime.now().isoformat()
        
        record = {
            'record_id': record_id,
            'timestamp': timestamp,
            'data': scan_data
        }
        
        # Save individual record file
        patient_dir = self.storage_dir / patient_id
        patient_dir.mkdir(exist_ok=True)
        
        record_file = patient_dir / f"{record_id}.json"
        with open(record_file, 'w') as f:
            json.dump(record, f, indent=2, default=str)
            
        # Update index
        with open(self.index_file, 'r') as f:
            index = json.load(f)
            
        if patient_id not in index['patients']:
            index['patients'][patient_id] = []
            
        # Add to index if not already present
        if record_id not in [r['record_id'] for r in index['patients'][patient_id]]:
            index['patients'][patient_id].append({
                'record_id': record_id,
                'timestamp': timestamp,
                'ga': scan_data.get('clinical', {}).get('estimated_ga')
            })
            
        with open(self.index_file, 'w') as f:
            json.dump(index, f, indent=2)

    def get_patient_history(self, patient_id: str) -> List[Dict]:
        """Retrieve all scan records for a patient, sorted by date"""
        patient_dir = self.storage_dir / patient_id
        if not patient_dir.exists():
            return []
            
        records = []
        for file in patient_dir.glob('*.json'):
            with open(file, 'r') as f:
                records.append(json.load(f))
                
        # Sort by timestamp
        records.sort(key=lambda x: x['timestamp'])
        return records

    def calculate_velocity(self, patient_id: str, current_record: Dict) -> Dict:
        """
        Calculate growth velocity based on previous and current scan
        """
        history = self.get_patient_history(patient_id)
        if len(history) < 2:
            return None
            
        # Find current and previous records in the sorted history
        current_id = current_record.get('file_id') or current_record.get('record_id')
        curr_record_full = None
        prev_record = None
        
        for i in range(len(history)):
            if history[i]['record_id'] == current_id:
                curr_record_full = history[i]
                if i > 0:
                    prev_record = history[i-1]
                break
        
        if not curr_record_full or not prev_record:
            return None
            
        # Time difference in weeks
        t1 = datetime.fromisoformat(prev_record['timestamp'])
        t2 = datetime.fromisoformat(curr_record_full['timestamp'])
        dt_weeks = (t2 - t1).total_seconds() / (60 * 60 * 24 * 7)
        
        if dt_weeks < 0.1: # Less than ~17 hours, velocity calculation is unreliable
            return None
            
        velocity = {}
        prev_data = prev_record['data'].get('measurements', {})
        curr_data = curr_record_full['data'].get('measurements', {})
        
        for key in ['HC', 'AC', 'BPD', 'FL']:
            if key in prev_data and key in curr_data:
                growth = curr_data[key] - prev_data[key]
                velocity[key] = round(growth / dt_weeks, 2)
                
        return {
            'dt_weeks': round(dt_weeks, 1),
            'velocity': velocity,
            'unit': 'mm/week'
        }

    def get_trend_data(self, patient_id: str, metric_type: str) -> List[Dict]:
        """Get history data points for a specific metric for charting"""
        history = self.get_patient_history(patient_id)
        trend = []
        
        for record in history:
            ga = record['data'].get('clinical', {}).get('estimated_ga')
            # Check risk_assessment for EFW/CI or measurements for primary biometrics
            value = None
            if metric_type.upper() in ['EFW', 'CI']:
                ra = record['data'].get('risk_assessment')
                if ra and isinstance(ra, dict):
                    metric_data = ra.get(metric_type.lower())
                    if metric_data and isinstance(metric_data, dict):
                        value = metric_data.get('value')
            else:
                value = record['data'].get('measurements', {}).get(metric_type.upper())
                
            if ga is not None and value is not None:
                trend.append({
                    'ga': round(ga, 1),
                    'value': round(value, 2),
                    'date': record['timestamp']
                })
        return trend

    def get_all_patients(self) -> List[Dict]:
        """Get a list of all patients with their latest status summary"""
        with open(self.index_file, 'r') as f:
            index = json.load(f)
            
        patient_list = []
        for pid, records in index['patients'].items():
            if not records: continue
            
            # Sort records by timestamp to find latest
            records.sort(key=lambda x: x['timestamp'], reverse=True)
            latest = records[0]
            
            patient_list.append({
                'patient_id': pid,
                'last_scan': latest['timestamp'],
                'current_ga': latest.get('ga'),
                'total_scans': len(records)
            })
            
        return patient_list
