"""
Clinical Recommendations Generator
Provides evidence-based recommendations based on risk assessment
"""

from typing import Dict, List
import yaml
import os


class RecommendationGenerator:
    """Generate clinical recommendations based on assessment results"""
    
    def __init__(self, config_path: str = None):
        """Initialize with clinical thresholds configuration"""
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), 'config', 'clinical_thresholds.yaml')
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.recommendations = self.config['recommendations']
    
    def generate_recommendations(self, assessment: Dict) -> List[Dict]:
        """
        Generate clinical recommendations based on assessment
        
        Args:
            assessment: Clinical assessment from ClinicalRulesEngine
        
        Returns:
            List of recommendation dicts with priority, category, and text
        """
        recommendations = []
        
        # Growth pattern recommendations
        if 'growth_pattern' in assessment and 'pattern' in assessment['growth_pattern']:
            pattern = assessment['growth_pattern']['pattern']
            if pattern in self.recommendations:
                for rec in self.recommendations[pattern]:
                    recommendations.append({
                        'priority': 'high' if pattern in ['IUGR', 'microcephaly'] else 'medium',
                        'category': 'Growth Pattern',
                        'text': rec,
                        'pattern': pattern
                    })
        
        # Borderline measurements recommendations
        borderline_count = sum(
            1 for m in assessment.get('measurements', {}).values()
            if m.get('risk_level') == 'borderline'
        )
        if borderline_count > 0:
            for rec in self.recommendations['borderline_measurements']:
                recommendations.append({
                    'priority': 'medium',
                    'category': 'Monitoring',
                    'text': rec
                })
        
        # Normal case recommendations
        if assessment.get('overall_risk') == 'normal':
            for rec in self.recommendations['normal']:
                recommendations.append({
                    'priority': 'low',
                    'category': 'Routine Care',
                    'text': rec
                })
        
        # Gestational age specific recommendations
        if 'gestational_age' in assessment:
            ga_risk = assessment['gestational_age'].get('risk_level')
            if ga_risk in ['high_risk', 'critical']:
                recommendations.append({
                    'priority': 'high',
                    'category': 'Gestational Age',
                    'text': 'Consult obstetrics for delivery planning'
                })
        
        # Sort by priority
        priority_order = {'high': 0, 'medium': 1, 'low': 2}
        recommendations.sort(key=lambda x: priority_order.get(x['priority'], 3))
        
        return recommendations
    
    def generate_follow_up_plan(self, assessment: Dict) -> Dict:
        """Generate follow-up schedule based on risk level"""
        risk_level = assessment.get('overall_risk', 'normal')
        
        follow_up_schedules = {
            'critical': {
                'next_scan': '1 week',
                'frequency': 'Weekly until delivery',
                'specialist': 'Maternal-Fetal Medicine (urgent)',
                'additional': ['Doppler studies', 'Non-stress tests']
            },
            'high_risk': {
                'next_scan': '2 weeks',
                'frequency': 'Every 2 weeks',
                'specialist': 'Maternal-Fetal Medicine',
                'additional': ['Growth velocity tracking', 'Amniotic fluid assessment']
            },
            'borderline': {
                'next_scan': '3-4 weeks',
                'frequency': 'Every 3-4 weeks',
                'specialist': 'Consider specialist consultation',
                'additional': ['Repeat measurements', 'Trend analysis']
            },
            'normal': {
                'next_scan': 'Per routine protocol',
                'frequency': 'Standard prenatal schedule',
                'specialist': 'Not required',
                'additional': ['Continue routine prenatal care']
            }
        }
        
        return follow_up_schedules.get(risk_level, follow_up_schedules['normal'])
    
    def generate_patient_summary(self, assessment: Dict, recommendations: List[Dict]) -> str:
        """Generate patient-friendly summary"""
        risk_level = assessment.get('overall_risk', 'normal')
        
        summaries = {
            'normal': "Your baby's measurements are within the normal range. Continue with routine prenatal care.",
            'borderline': "Some measurements are slightly outside the typical range. Your healthcare provider will monitor these closely.",
            'high_risk': "Some measurements require closer monitoring. Your healthcare provider will discuss next steps with you.",
            'critical': "Immediate medical review is recommended. Please contact your healthcare provider."
        }
        
        summary = summaries.get(risk_level, summaries['normal'])
        
        # Add growth pattern info
        if 'growth_pattern' in assessment and 'pattern' in assessment['growth_pattern']:
            pattern = assessment['growth_pattern']['pattern']
            if pattern == 'AGA':
                summary += " Growth pattern is appropriate for gestational age."
            elif pattern != 'borderline':
                summary += f" Growth pattern shows {pattern}."
        
        return summary
    
    def format_for_report(self, assessment: Dict, recommendations: List[Dict]) -> Dict:
        """Format assessment and recommendations for clinical report"""
        return {
            'summary': self.generate_patient_summary(assessment, recommendations),
            'risk_level': assessment.get('overall_risk', 'normal'),
            'recommendations': recommendations,
            'follow_up': self.generate_follow_up_plan(assessment),
            'alerts': assessment.get('alerts', []),
            'clinical_notes': self._generate_clinical_notes(assessment)
        }
    
    def _generate_clinical_notes(self, assessment: Dict) -> List[str]:
        """Generate clinical notes for healthcare providers"""
        notes = []
        
        # Overall assessment
        risk = assessment.get('overall_risk', 'normal')
        notes.append(f"Overall risk assessment: {risk.upper().replace('_', ' ')}")
        
        # Growth pattern
        if 'growth_pattern' in assessment:
            pattern = assessment['growth_pattern'].get('pattern', 'Unknown')
            desc = assessment['growth_pattern'].get('description', '')
            notes.append(f"Growth pattern: {pattern} - {desc}")
        
        # Individual measurements
        for key, result in assessment.get('measurements', {}).items():
            notes.append(f"{key}: {result.get('message', 'No data')}")
        
        # Gestational age
        if 'gestational_age' in assessment:
            notes.append(assessment['gestational_age'].get('message', ''))
        
        return notes


# Example usage
if __name__ == "__main__":
    from clinical_rules import ClinicalRulesEngine
    
    # Initialize
    engine = ClinicalRulesEngine()
    rec_gen = RecommendationGenerator()
    
    # Sample data
    sample_data = {
        'HC': {'value': 245, 'percentile': 50},
        'AC': {'value': 289, 'percentile': 52},
        'BPD': {'value': 75, 'percentile': 48},
        'FL': {'value': 65, 'percentile': 51},
        'GA': {'value': 28.3}
    }
    
    # Generate assessment and recommendations
    assessment = engine.generate_comprehensive_assessment(sample_data)
    recommendations = rec_gen.generate_recommendations(assessment)
    report = rec_gen.format_for_report(assessment, recommendations)
    
    print("Clinical Report:")
    print(f"Summary: {report['summary']}")
    print(f"\nRecommendations ({len(recommendations)}):")
    for rec in recommendations:
        print(f"  [{rec['priority'].upper()}] {rec['text']}")
