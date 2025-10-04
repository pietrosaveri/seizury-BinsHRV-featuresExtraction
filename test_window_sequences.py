#!/usr/bin/env python3
"""
Test the updated LSTM sequence builder with 60-minute windows and dense supervision.
"""

import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.getcwd())
from lstm_sequences import LSTMSequenceBuilder

def create_test_data():
    """Create test data that mimics the pipeline output with overlapping windows."""
    
    data = []
    
    # Window 1: minutes 0-59
    for minute in range(60):
        data.append({
            'subject_id': 'sub-001',
            'recording_id': 'rec-001',
            'window_start_time': 0.0,
            'window_end_time': 3600.0,
            'minute_time': minute * 60.0,
            'minute_in_window': minute,
            'RRMean_3': 800 + np.random.randn() * 10,
            'SDNN_5': 50 + np.random.randn() * 5,
            'LF_POWER_10': 100 + np.random.randn() * 20,
            'bin_1': 1 if 55 <= minute < 60 else 0,
            'bin_2': 0,
            'bin_3': 0, 
            'bin_4': 1 if minute < 55 else 0,
            'is_ictal': 0,
            'is_postictal': 0,
            'recent_seizure_flag': 0,
            'training_mask': 1
        })
    
    # Window 2: minutes 30-89 (includes seizure at minute 60)
    for minute in range(30, 90):
        data.append({
            'subject_id': 'sub-001',
            'recording_id': 'rec-001',
            'window_start_time': 1800.0,
            'window_end_time': 5400.0,
            'minute_time': minute * 60.0,
            'minute_in_window': minute - 30,
            'RRMean_3': 800 + np.random.randn() * 10,
            'SDNN_5': 50 + np.random.randn() * 5,
            'LF_POWER_10': 100 + np.random.randn() * 20,
            'bin_1': 1 if 55 <= minute < 60 else 0,
            'bin_2': 0,
            'bin_3': 0,
            'bin_4': 1 if minute < 55 or minute > 90 else 0,
            'is_ictal': 1 if minute == 60 else 0,
            'is_postictal': 1 if 60 < minute <= 90 else 0,
            'recent_seizure_flag': 1 if 60 < minute <= 90 else 0,
            'training_mask': 1 if minute < 60 or minute > 90 else 0
        })
    
    return pd.DataFrame(data)

def test_updated_sequences():
    """Test the updated sequence builder."""
    print("Testing Updated LSTM Sequence Builder")
    print("=" * 50)
    
    df = create_test_data()
    print(f"Created test data: {len(df)} rows")
    print(f"Windows: {df.groupby(['window_start_time', 'window_end_time']).size().to_dict()}")
    
    # Initialize with new defaults
    builder = LSTMSequenceBuilder(
        seq_len=60,
        stride=30,
        normalize_features=False
    )
    
    sequences, labels, timestamps, train_masks = builder.create_sequences_from_recording(df)
    
    print(f"\nResults:")
    print(f"Sequences: {sequences.shape}")
    print(f"Labels: {labels.shape}")
    print(f"Timestamps: {timestamps.shape}")
    print(f"Train masks: {train_masks.shape}")
    
    if len(sequences) > 0:
        print(f"\n✓ Created {len(sequences)} sequences")
        if len(labels.shape) == 3:
            print(f"✓ Dense supervision: {labels.shape[1]} predictions per sequence")
        
        # Check training masks
        total_mins = train_masks.size
        trainable_mins = train_masks.sum()
        print(f"✓ Training masks: {trainable_mins}/{total_mins} trainable minutes")
    
    print("\nTest Complete!")

if __name__ == "__main__":
    test_updated_sequences()