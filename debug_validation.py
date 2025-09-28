#!/usr/bin/env python3
"""
Debug the seizure validation logic for Test Case 4.
"""

import pandas as pd
import sys
import os

sys.path.append(os.getcwd())
from data_processing_pipeline import DataProcessingPipeline

def debug_validation():
    """Debug the complex validation case."""
    
    print("Debugging Test Case 4")
    print("=" * 30)
    
    processor = DataProcessingPipeline(data_root=".", output_dir=".", use_parallel=False)
    
    # Test case 4: Complex scenario
    seizures = pd.DataFrame({
        'onset': [5*60, 25*60, 40*60, 80*60, 100*60, 160*60],  # Mix of valid/invalid
        'duration': [60, 60, 60, 60, 60, 60]
    })
    
    print("Input seizures (in minutes):")
    for i, row in seizures.iterrows():
        print(f"  Seizure {i}: {row['onset']/60:.1f} minutes")
    
    # Manually step through the validation logic
    print("\nManual validation analysis:")
    
    seizures_sorted = seizures.sort_values('onset').reset_index(drop=True)
    for idx, seizure in seizures_sorted.iterrows():
        onset_time = seizure['onset']
        onset_min = onset_time / 60.0
        is_valid = True
        reason = ""
        
        print(f"\nAnalyzing seizure {idx} at {onset_min:.1f} minutes:")
        
        # Criterion 1: Must be at least 20 minutes from start
        if onset_time < 20 * 60:
            is_valid = False
            reason = f"Too early ({onset_min:.1f}min < 20min)"
        else:
            print(f"  ✓ Time criterion: {onset_min:.1f}min >= 20min")
        
        if is_valid:
            # Criterion 2: Must not be within 30-minute post-ictal phase
            for prev_idx, prev_seizure in seizures_sorted.iterrows():
                if prev_idx >= idx:  # Only check previous seizures
                    break
                    
                prev_onset = prev_seizure['onset']
                prev_onset_min = prev_onset / 60.0
                time_since_prev = (onset_time - prev_onset) / 60.0
                
                print(f"  Checking against seizure {prev_idx} at {prev_onset_min:.1f}min:")
                print(f"    Time difference: {time_since_prev:.1f} minutes")
                
                if 0 < time_since_prev <= 30:
                    is_valid = False
                    reason = f"Within post-ictal period of seizure {prev_idx} ({time_since_prev:.1f}min <= 30min)"
                    print(f"    ✗ {reason}")
                    break
                else:
                    print(f"    ✓ Outside post-ictal period ({time_since_prev:.1f}min > 30min)")
        
        result = "VALID" if is_valid else "INVALID"
        print(f"  Result: {result}")
        if reason:
            print(f"  Reason: {reason}")
    
    # Test the actual function
    print(f"\nActual function result:")
    valid_seizures = processor._validate_seizures(seizures, 4*3600)
    print(f"Valid seizure times: {[t/60 for t in valid_seizures['onset']] if len(valid_seizures) > 0 else 'None'}")

if __name__ == "__main__":
    debug_validation()