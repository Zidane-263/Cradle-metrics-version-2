import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from clinical_rules import ClinicalRulesEngine
from clinical_history import ClinicalHistoryManager

def test_biometrics():
    print("--- Testing Clinical Biometrics ---")
    engine = ClinicalRulesEngine()
    
    # Sample Case: Normal Term Fetus
    # HC: 320mm, AC: 300mm, BPD: 90mm, FL: 70mm
    hc, ac, bpd, fl = 320, 300, 90, 70
    
    efw = engine.calculate_efw(hc, ac, bpd, fl)
    print(f"Calculated EFW (Hadlock 4): {efw:.2f}g")
    
    ci = engine.calculate_ci(bpd, hc=hc)
    print(f"Calculated Cephalic Index: {ci:.2f}%")
    
    # Assess comprehensive
    sample_data = {
        'HC': {'value': hc, 'percentile': 50},
        'AC': {'value': ac, 'percentile': 50},
        'BPD': {'value': bpd, 'percentile': 50},
        'FL': {'value': fl, 'percentile': 50}
    }
    assessment = engine.generate_comprehensive_assessment(sample_data)
    print(f"Assessment EFW: {assessment.get('efw', {}).get('value')}g")
    print(f"Assessment CI: {assessment.get('ci', {}).get('value')}%")
    
    return assessment

def test_history(assessment):
    print("\n--- Testing Clinical History ---")
    history_mgr = ClinicalHistoryManager(storage_dir='./test_history')
    
    patient_id = "test_patient_001"
    
    # Simulated scan response
    scan_response = {
        'file_id': 'test_scan_1',
        'measurements': {'HC': 320, 'AC': 300, 'BPD': 90, 'FL': 70},
        'risk_assessment': assessment
    }
    
    history_mgr.save_record(patient_id, scan_response)
    print(f"Saved record for {patient_id}")
    
    # Retrieve history
    history = history_mgr.get_patient_history(patient_id)
    print(f"History count: {len(history)}")
    
    if len(history) > 0:
        print(f"First record ID: {history[0]['record_id']}")
        
    return True

if __name__ == "__main__":
    try:
        assessment = test_biometrics()
        test_history(assessment)
        print("\n✅ Verification successful!")
    except Exception as e:
        print(f"\n❌ Verification failed: {str(e)}")
        sys.exit(1)
