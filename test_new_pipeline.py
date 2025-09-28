#!/usr/bin/env python3
"""
Test script for the new seizure proximity pipeline
"""

import numpy as np
from data_processing_pipeline import HRVFeatureProcessor

def test_feature_extraction():
    """Test the new minute-level feature extraction."""
    
    print("Testing HRV Feature Processor with Seizure Proximity Bins")
    print("=" * 60)
    
    # Initialize processor
    processor = HRVFeatureProcessor()
    
    print("Configuration:")
    print(f"  Window size: {processor.window_size_minutes}min")
    print(f"  Overlap: {processor.overlap_minutes}min")
    print(f"  Feature windows: {list(processor.feature_windows.keys())}")
    print(f"  Bin ranges: {processor.bin_ranges}")
    
    # Test seizure proximity bin calculation
    print("\nTesting Seizure Proximity Bins:")
    test_cases = [
        (120, [420]),    # 2min -> seizure at 7min = 5min distance -> Bin 2
        (600, [900]),    # 10min -> seizure at 15min = 5min distance -> Bin 2
        (1200, [3000]),  # 20min -> seizure at 50min = 30min distance -> Bin 3
        (3600, []),      # 60min -> no seizures -> Bin 4
    ]
    
    for minute_time, seizures in test_cases:
        bin_result = processor._calculate_seizure_proximity_bin(minute_time, seizures)
        distance = abs(seizures[0] - minute_time) / 60 if seizures else '>60'
        active_bin = [i+1 for i, v in enumerate(bin_result.values()) if v == 1][0]
        print(f"  Minute {minute_time/60:4.1f}min, distance {distance:>5}min -> Bin {active_bin} {bin_result}")
    
    print("\nFeature Groups:")
    for window, features in processor.feature_windows.items():
        print(f"  {window:>5}: {features}")

if __name__ == "__main__":
    test_feature_extraction()