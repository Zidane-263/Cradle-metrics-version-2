#!/usr/bin/env python3
"""
INTERGROWTH-21st Reference Data
Simplified lookup tables based on published INTERGROWTH-21st standards
"""

# INTERGROWTH-21st 50th percentile values (mean) for key gestational ages
# Source: INTERGROWTH-21st published tables
# Values in millimeters

INTERGROWTH_REFERENCE = {
    # Gestational Age (weeks): {HC, AC, FL, BPD}
    14: {'HC': 96, 'AC': 79, 'FL': 11, 'BPD': 26},
    16: {'HC': 124, 'AC': 103, 'FL': 18, 'BPD': 35},
    18: {'HC': 151, 'AC': 131, 'FL': 25, 'BPD': 42},
    20: {'HC': 176, 'AC': 158, 'FL': 32, 'BPD': 49},
    22: {'HC': 200, 'AC': 184, 'FL': 38, 'BPD': 55},
    24: {'HC': 222, 'AC': 209, 'FL': 44, 'BPD': 61},
    26: {'HC': 243, 'AC': 232, 'FL': 49, 'BPD': 66},
    28: {'HC': 263, 'AC': 254, 'FL': 54, 'BPD': 71},
    30: {'HC': 282, 'AC': 275, 'FL': 58, 'BPD': 75},
    32: {'HC': 300, 'AC': 295, 'FL': 62, 'BPD': 79},
    34: {'HC': 317, 'AC': 314, 'FL': 66, 'BPD': 83},
    36: {'HC': 333, 'AC': 332, 'FL': 69, 'BPD': 87},
    38: {'HC': 348, 'AC': 349, 'FL': 72, 'BPD': 90},
    40: {'HC': 362, 'AC': 365, 'FL': 75, 'BPD': 93},
}

# Standard deviations (approximate, for percentile calculation)
INTERGROWTH_SD = {
    'HC': 12,  # mm
    'AC': 15,  # mm
    'FL': 4,   # mm
    'BPD': 4,  # mm
}

# Percentile Z-scores
PERCENTILE_Z_SCORES = {
    3: -1.88,
    5: -1.64,
    10: -1.28,
    25: -0.67,
    50: 0.00,
    75: 0.67,
    90: 1.28,
    95: 1.64,
    97: 1.88,
}
