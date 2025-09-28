#!/usr/bin/env python3
"""
Test the new seizure validation logic for patient selection.
"""

import pandas as pd
import sys
import os

# Add the current directory to path to import our modules
sys.path.append(os.getcwd())

from data_processing_pipeline import DataProcessingPipeline

def test_seizure_validation():
    """Test the seizure validation logic."""
    
    print("Testing Seizure Validation Logic")
    print("=" * 40)
    
    # Create a mock processor to test the validation method
    processor = DataProcessingPipeline(data_root=".", output_dir=".", use_parallel=False)
    
    # Test case 1: Valid seizures
    print("\nTest Case 1: All valid seizures")
    seizures_1 = pd.DataFrame({
        'onset': [30*60, 90*60, 150*60],  # 30min, 90min, 150min
        'duration': [60, 60, 60]  # 1 minute each
    })
    
    valid_1 = processor._validate_seizures(seizures_1, 4*3600)  # 4 hour recording
    print(f"Input seizures: {len(seizures_1)}")
    print(f"Valid seizures: {len(valid_1)}")
    print(f"Seizure times: {[t/60 for t in seizures_1['onset']]}")
    print(f"Expected: All 3 seizures should be valid (all >20min from start, >30min apart)")
    
    # Test case 2: Early seizure (invalid)
    print("\nTest Case 2: Early seizure should be excluded")
    seizures_2 = pd.DataFrame({
        'onset': [10*60, 50*60, 120*60],  # 10min (INVALID), 50min, 120min
        'duration': [60, 60, 60]
    })
    
    valid_2 = processor._validate_seizures(seizures_2, 4*3600)
    print(f"Input seizures: {len(seizures_2)}")
    print(f"Valid seizures: {len(valid_2)}")
    print(f"Seizure times: {[t/60 for t in seizures_2['onset']]}")
    print(f"Valid seizure times: {[t/60 for t in valid_2['onset']] if len(valid_2) > 0 else 'None'}")
    print(f"Expected: 2 valid seizures (first at 10min should be excluded)")
    
    # Test case 3: Close seizures (second should be invalid)
    print("\nTest Case 3: Close seizures - post-ictal exclusion")
    seizures_3 = pd.DataFrame({
        'onset': [30*60, 45*60, 90*60],  # 30min, 45min (INVALID - 15min after first), 90min
        'duration': [60, 60, 60]
    })
    
    valid_3 = processor._validate_seizures(seizures_3, 4*3600)
    print(f"Input seizures: {len(seizures_3)}")
    print(f"Valid seizures: {len(valid_3)}")
    print(f"Seizure times: {[t/60 for t in seizures_3['onset']]}")
    print(f"Valid seizure times: {[t/60 for t in valid_3['onset']] if len(valid_3) > 0 else 'None'}")
    print(f"Expected: 2 valid seizures (middle one at 45min should be excluded - within 30min post-ictal)")
    
    # Test case 4: Complex scenario
    print("\nTest Case 4: Complex scenario")
    seizures_4 = pd.DataFrame({
        'onset': [5*60, 25*60, 40*60, 80*60, 100*60, 160*60],  # Mix of valid/invalid
        'duration': [60, 60, 60, 60, 60, 60]
    })
    
    valid_4 = processor._validate_seizures(seizures_4, 4*3600)
    print(f"Input seizures: {len(seizures_4)}")
    print(f"Valid seizures: {len(valid_4)}")
    print(f"Seizure times: {[t/60 for t in seizures_4['onset']]}")
    print(f"Valid seizure times: {[t/60 for t in valid_4['onset']] if len(valid_4) > 0 else 'None'}")
    print(f"Expected analysis:")
    print(f"  - 5min: INVALID (too early)")
    print(f"  - 25min: VALID (>20min from start)")
    print(f"  - 40min: INVALID (15min after 25min seizure, within 30min post-ictal)")
    print(f"  - 80min: VALID (55min after 25min seizure)")
    print(f"  - 100min: INVALID (20min after 80min seizure, within 30min post-ictal)")
    print(f"  - 160min: VALID (80min after 80min seizure)")
    
    # Test case 5: Empty DataFrame
    print("\nTest Case 5: No seizures")
    seizures_5 = pd.DataFrame()
    valid_5 = processor._validate_seizures(seizures_5, 4*3600)
    print(f"Input seizures: {len(seizures_5)}")
    print(f"Valid seizures: {len(valid_5)}")
    print(f"Expected: 0 valid seizures (empty input)")
    
    print("\n" + "="*40)
    print("Seizure Validation Test Complete!")

if __name__ == "__main__":
    test_seizure_validation()