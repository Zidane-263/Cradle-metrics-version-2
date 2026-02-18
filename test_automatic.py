#!/usr/bin/env python3
"""
Simple test script using config file
Run automatic analysis on image specified in config.yaml
"""

import yaml
from pathlib import Path
from automatic_pipeline import AutomaticPipeline


def load_config(config_path: str = 'config.yaml') -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    """Run automatic analysis using config file"""
    
    # Load config
    print("Loading configuration from config.yaml...")
    config = load_config()
    
    # Create pipeline
    pipeline = AutomaticPipeline(
        yolo_model_path=config['YOLO_MODEL'],
        sam_checkpoint=config['SAM_MODEL'],
        pixel_to_mm=config['PIXEL_TO_MM'],
        confidence_threshold=config['CONFIDENCE_THRESHOLD'],
        enable_clinical=config.get('ENABLE_CLINICAL_ASSESSMENT', True),
        ga_weeks=config.get('GESTATIONAL_AGE_WEEKS')
    )
    
    # Process image
    results = pipeline.process_image(
        image_path=config['TEST_IMAGE'],
        output_dir=config['OUTPUT_DIR']
    )
    
    print(f"\n✅ Done! Check results in: {config['OUTPUT_DIR']}/")


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\n💡 Make sure to:")
        print("   1. Update TEST_IMAGE path in config.yaml")
        print("   2. Wait for YOLO training to complete")
        print("   3. Ensure SAM model is downloaded")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise
