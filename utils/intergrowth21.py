#!/usr/bin/env python3
"""
INTERGROWTH-21st Fetal Growth Standards Calculator (Simplified)
Based on published INTERGROWTH-21st reference tables
"""

import numpy as np
from scipy import stats, interpolate
from typing import Dict, Optional, Tuple
from utils.intergrowth_data import INTERGROWTH_REFERENCE, INTERGROWTH_SD, PERCENTILE_Z_SCORES


class Intergrowth21:
    """
    Simplified INTERGROWTH-21st calculator using reference tables
    """
    
    def __init__(self):
        """Initialize calculator"""
        self.reference = INTERGROWTH_REFERENCE
        self.sd = INTERGROWTH_SD
        self.valid_metrics = ['HC', 'AC', 'FL', 'BPD']
        
        # Create interpolation functions for each metric
        self.interpolators = {}
        for metric in self.valid_metrics:
            gas = sorted(self.reference.keys())
            values = [self.reference[ga][metric] for ga in gas]
            self.interpolators[metric] = interpolate.interp1d(
                gas, values, kind='cubic', fill_value='extrapolate'
            )
    
    def get_expected_value(self, ga_weeks: float, metric_type: str) -> float:
        """Get expected (50th percentile) value for GA"""
        if metric_type not in self.valid_metrics:
            raise ValueError(f"Invalid metric. Must be one of {self.valid_metrics}")
        
        # Use interpolation for smooth values between reference points
        return float(self.interpolators[metric_type](ga_weeks))
    
    def calculate_z_score(self, measurement: float, ga_weeks: float, 
                         metric_type: str) -> float:
        """Calculate Z-score"""
        expected = self.get_expected_value(ga_weeks, metric_type)
        sd = self.sd[metric_type]
        z_score = (measurement - expected) / sd
        return z_score
    
    def calculate_percentile(self, measurement: float, ga_weeks: float, 
                            metric_type: str) -> float:
        """Calculate percentile"""
        z_score = self.calculate_z_score(measurement, ga_weeks, metric_type)
        percentile = stats.norm.cdf(z_score) * 100
        return max(0, min(100, percentile))  # Clamp to 0-100
    
    def estimate_ga_from_measurement(self, measurement: float, 
                                    metric_type: str) -> float:
        """Estimate GA from measurement"""
        if metric_type not in self.valid_metrics:
            raise ValueError(f"Invalid metric. Must be one of {self.valid_metrics}")
        
        # Search for closest match
        best_ga = None
        min_diff = float('inf')
        
        for ga in np.arange(14, 42, 0.5):
            expected = self.get_expected_value(ga, metric_type)
            diff = abs(measurement - expected)
            if diff < min_diff:
                min_diff = diff
                best_ga = ga
        
        return round(best_ga, 1) if best_ga else 28.0
    
    def classify_growth(self, percentile: float) -> Dict[str, str]:
        """Classify growth based on percentile"""
        if percentile < 3:
            return {
                'classification': 'SGA',
                'full_name': 'Small for Gestational Age',
                'severity': 'Severe',
                'flag': '⚠️'
            }
        elif percentile < 10:
            return {
                'classification': 'SGA',
                'full_name': 'Small for Gestational Age',
                'severity': 'Mild',
                'flag': '⚠️'
            }
        elif percentile <= 90:
            return {
                'classification': 'AGA',
                'full_name': 'Appropriate for Gestational Age',
                'severity': 'Normal',
                'flag': '✓'
            }
        elif percentile <= 97:
            return {
                'classification': 'LGA',
                'full_name': 'Large for Gestational Age',
                'severity': 'Mild',
                'flag': '⚠️'
            }
        else:
            return {
                'classification': 'LGA',
                'full_name': 'Large for Gestational Age',
                'severity': 'Severe',
                'flag': '⚠️'
            }
    
    def get_expected_range(self, ga_weeks: float, metric_type: str, 
                          percentile_range: Tuple[float, float] = (10, 90)) -> Tuple[float, float]:
        """Get expected range for GA"""
        expected = self.get_expected_value(ga_weeks, metric_type)
        sd = self.sd[metric_type]
        
        z_lower = stats.norm.ppf(percentile_range[0] / 100)
        z_upper = stats.norm.ppf(percentile_range[1] / 100)
        
        lower = expected + z_lower * sd
        upper = expected + z_upper * sd
        
        return (lower, upper)
    
    def assess_measurement(self, measurement: float, ga_weeks: Optional[float], 
                          metric_type: str) -> Dict:
        """Complete assessment of a measurement"""
        # Estimate GA if not provided
        estimated_ga = self.estimate_ga_from_measurement(measurement, metric_type)
        
        # Use provided GA or estimated
        ga_for_calc = ga_weeks if ga_weeks is not None else estimated_ga
        
        # Calculate metrics
        percentile = self.calculate_percentile(measurement, ga_for_calc, metric_type)
        z_score = self.calculate_z_score(measurement, ga_for_calc, metric_type)
        growth = self.classify_growth(percentile)
        expected_range = self.get_expected_range(ga_for_calc, metric_type)
        
        return {
            'measurement': measurement,
            'metric_type': metric_type,
            'ga_weeks': ga_for_calc,
            'ga_provided': ga_weeks is not None,
            'estimated_ga': estimated_ga,
            'percentile': round(percentile, 1),
            'z_score': round(z_score, 2),
            'classification': growth['classification'],
            'classification_full': growth['full_name'],
            'severity': growth['severity'],
            'flag': growth['flag'],
            'expected_range': (round(expected_range[0], 1), round(expected_range[1], 1)),
            'within_normal': 10 <= percentile <= 90
        }


def example_usage():
    """Example usage"""
    calc = Intergrowth21()
    
    # Example measurements
    measurements = {
        'HC': 245.3,
        'AC': 289.7,
        'BPD': 85.2,
        'FL': 65.4
    }
    
    known_ga = 28
    
    print("="*70)
    print("INTERGROWTH-21st Assessment")
    print("="*70)
    
    for metric, value in measurements.items():
        assessment = calc.assess_measurement(value, known_ga, metric)
        
        print(f"\n{metric}: {value}mm")
        print(f"  Percentile: {assessment['percentile']}th")
        print(f"  Z-score: {assessment['z_score']}")
        print(f"  Classification: {assessment['classification']} {assessment['flag']}")
        print(f"  Estimated GA: {assessment['estimated_ga']} weeks")
        print(f"  Expected range (10-90th): {assessment['expected_range'][0]:.1f}-{assessment['expected_range'][1]:.1f}mm")


if __name__ == "__main__":
    example_usage()
