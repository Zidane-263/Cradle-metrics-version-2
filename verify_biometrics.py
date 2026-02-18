
import sys
from pathlib import Path
import math

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from clinical_rules import ClinicalRulesEngine

def test_clinical_rules():
    engine = ClinicalRulesEngine()
    
    print("Testing Clinical Rules Engine...")
    
    # Test Measurements (28 weeks GA)
    # Typical values for 28 weeks: HC ~260, AC ~240, BPD ~70, FL ~54
    hc, ac, bpd, fl = 260, 240, 70, 54
    ga = 28.0
    
    # 1. Test EFW Calculation
    efw = engine.calculate_efw(hc, ac, bpd, fl)
    print(f"EFW for 28w (HC={hc}, AC={ac}, BPD={bpd}, FL={fl}): {efw:.1f}g")
    
    # Expected Hadlock 4 for these values is ~1200-1300g
    if 1100 < efw < 1400:
        print("✅ EFW calculation within expected range.")
    else:
        print("❌ EFW calculation outside expected range!")
        
    # 2. Test EFW Percentile
    efw_perc = engine.calculate_efw_percentile(efw, ga)
    print(f"EFW Percentile for {efw:.1f}g at {ga}w: {efw_perc}%")
    
    if 10 < efw_perc < 90:
        print("✅ EFW percentile is normal (as expected for these values).")
    else:
        print(f"⚠️ EFW percentile is unusual: {efw_perc}")
        
    # 3. Test Cephalic Index
    ci = engine.calculate_ci(bpd, hc=hc)
    print(f"Cephalic Index (BPD={bpd}, HC={hc}): {ci:.1f}%")
    
    # Normal CI is 70-85%
    if 70 <= ci <= 85:
        print("✅ Cephalic Index within normal range.")
    else:
        print("⚠️ Cephalic Index outside normal range.")
        
    # 4. Test Comprehensive Assessment
    # We need to simulate the percentiles that the pipeline would provide
    sample_data = {
        'HC': {'value': hc, 'percentile': 50},
        'AC': {'value': ac, 'percentile': 50},
        'BPD': {'value': bpd, 'percentile': 50},
        'FL': {'value': fl, 'percentile': 50},
        'GA': {'value': ga}
    }
    
    assessment = engine.generate_comprehensive_assessment(sample_data)
    
    print("\nComprehensive Assessment Results:")
    print(f"Overall Risk: {assessment['overall_risk']}")
    print(f"EFW: {assessment['efw']['value']} {assessment['efw']['unit']} ({assessment['efw'].get('percentile', 'N/A')}%)")
    print(f"CI: {assessment['ci']['value']} {assessment['ci']['unit']} ({assessment['ci']['status']})")
    
    if 'efw' in assessment and 'ci' in assessment:
        print("✅ Comprehensive assessment includes advanced biometrics.")
    else:
        print("❌ Comprehensive assessment missing advanced biometrics!")

if __name__ == "__main__":
    test_clinical_rules()
