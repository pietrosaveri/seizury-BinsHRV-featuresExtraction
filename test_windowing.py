#!/usr/bin/env python3
"""
Test script to verify the overlapping windowing and TTs calculation.
"""

import sys
sys.path.append('.')

from data_processing_pipeline import HRVFeatureProcessor
import pandas as pd

def test_windowing_and_tts():
    """Test the windowing and TTs calculation logic."""
    
    # Create a processor instance
    processor = HRVFeatureProcessor(
        sampling_rate=256,
        window_size_minutes=60,
        overlap_minutes=30
    )
    
    # Test case 1: Recording with seizure at 2 hours (7200 seconds)
    print("Test Case 1: Seizure at 2 hours (7200s)")
    print("=" * 50)
    
    seizure_timestamps = [7200.0]  # Seizure at 2 hours
    total_duration = 14400.0  # 4 hours total
    
    # Calculate expected windows
    window_size_seconds = 60 * 60  # 60 minutes
    step_size = 30 * 60  # 30 minutes
    
    windows = []
    for i in range(int((total_duration - window_size_seconds) / step_size) + 1):
        start_time = i * step_size
        end_time = start_time + window_size_seconds
        if end_time <= total_duration:
            windows.append((start_time, end_time))
    
    print(f"Expected windows for {total_duration/3600:.1f} hour recording:")
    for i, (start, end) in enumerate(windows):
        print(f"  Window {i+1}: {start/3600:.1f}h - {end/3600:.1f}h ({start}s - {end}s)")
    
    print(f"\nSeizure timestamp: {seizure_timestamps[0]/3600:.1f}h ({seizure_timestamps[0]}s)")
    print("\nTTs Analysis:")
    
    for i, (start_time, end_time) in enumerate(windows):
        label, tts = processor._calculate_window_label_and_tts(
            start_time, end_time, seizure_timestamps
        )
        
        contains_seizure = start_time <= seizure_timestamps[0] <= end_time
        
        print(f"  Window {i+1} ({start_time/3600:.1f}h-{end_time/3600:.1f}h):")
        print(f"    Start: {start_time/3600:.1f}h ({start_time}s)")
        print(f"    Contains seizure: {contains_seizure}")
        print(f"    Label: {label} ({'No seizure' if label==0 else 'Pre-seizure' if label==1 else 'Seizure'})")
        print(f"    TTs from start: {tts:.1f}s ({tts/60:.1f}min)")
        print()
    
    # Test case 2: Multiple seizures
    print("\nTest Case 2: Multiple seizures")
    print("=" * 50)
    
    seizure_timestamps = [3600.0, 10800.0]  # Seizures at 1h and 3h
    print(f"Seizure timestamps: {[ts/3600 for ts in seizure_timestamps]} hours")
    
    print("\nTTs Analysis for multiple seizures:")
    for i, (start_time, end_time) in enumerate(windows):
        label, tts = processor._calculate_window_label_and_tts(
            start_time, end_time, seizure_timestamps
        )
        
        contains_seizures = [ts for ts in seizure_timestamps if start_time <= ts <= end_time]
        
        print(f"  Window {i+1} ({start_time/3600:.1f}h-{end_time/3600:.1f}h):")
        print(f"    Contains seizures: {[ts/3600 for ts in contains_seizures]} hours")
        print(f"    Label: {label} ({'No seizure' if label==0 else 'Pre-seizure' if label==1 else 'Seizure'})")
        print(f"    TTs from start to closest: {tts:.1f}s ({tts/60:.1f}min)")
        print()

if __name__ == "__main__":
    test_windowing_and_tts()