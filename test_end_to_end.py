#!/usr/bin/env python3
"""
Quick test to verify the new ictal/postictal masking works with LSTM sequence builder.
"""

import pandas as pd
import numpy as np
from data_processing_pipeline import HRVFeatureProcessor
from lstm_sequences import LSTMSequenceBuilder

def test_end_to_end():
    """Test the full pipeline from feature extraction to sequence building."""
    
    print("Testing End-to-End Pipeline with Ictal/Postictal Masking")
    print("=" * 60)
    
    # Step 1: Create fake feature DataFrame
    print("1. Creating fake feature data...")
    
    # Create 120 minutes of data (2 hours)
    n_minutes = 120
    fake_data = []
    
    for minute in range(n_minutes):
        minute_time = minute * 60.0  # Convert to seconds
        
        # Add some fake HRV features
        row = {
            'subject_id': 'sub-test',
            'recording_id': 'test_recording',
            'window_start_time': 0.0,
            'window_end_time': 3600.0,
            'minute_time': minute_time,
            'minute_in_window': minute % 60,
            'RRMean_3': 800 + np.random.randn() * 50,
            'SDNN_3': 50 + np.random.randn() * 10,
            'LF_NORM_10': 0.5 + np.random.randn() * 0.1,
            'is_padded': False
        }
        
        # Add fake seizure events: seizure at minute 60 (1 hour)
        if minute == 60:  # During seizure
            row.update({
                'bin_1': 0, 'bin_2': 0, 'bin_3': 0, 'bin_4': 0,
                'is_ictal': 1, 'is_postictal': 0, 'recent_seizure_flag': 0,
                'training_mask': 0
            })
        elif 60 < minute <= 90:  # Post-ictal (30 minutes after seizure start)
            row.update({
                'bin_1': 0, 'bin_2': 0, 'bin_3': 0, 'bin_4': 0,
                'is_ictal': 0, 'is_postictal': 1, 'recent_seizure_flag': 1,
                'training_mask': 0
            })
        elif 55 <= minute < 60:  # Pre-ictal
            row.update({
                'bin_1': 1, 'bin_2': 0, 'bin_3': 0, 'bin_4': 0,
                'is_ictal': 0, 'is_postictal': 0, 'recent_seizure_flag': 0,
                'training_mask': 1
            })
        else:  # Normal
            row.update({
                'bin_1': 0, 'bin_2': 0, 'bin_3': 0, 'bin_4': 1,
                'is_ictal': 0, 'is_postictal': 0, 'recent_seizure_flag': 0,
                'training_mask': 1
            })
        
        fake_data.append(row)
    
    df = pd.DataFrame(fake_data)
    print(f"Created {len(df)} minutes of fake data")
    
    # Step 2: Test sequence building
    print("\n2. Testing sequence building...")
    
    builder = LSTMSequenceBuilder(seq_len=10, stride=1, normalize_features=False)
    sequences, labels, timestamps, train_masks = builder.create_sequences_from_recording(df)
    
    print(f"Created {len(sequences)} sequences")
    print(f"Sequence shape: {sequences.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Training masks shape: {train_masks.shape}")
    
    # Step 3: Analyze results
    print("\n3. Analyzing training masks...")
    
    n_trainable = train_masks.sum()
    n_excluded = len(train_masks) - n_trainable
    
    print(f"Trainable sequences: {n_trainable}/{len(train_masks)} ({n_trainable/len(train_masks)*100:.1f}%)")
    print(f"Excluded sequences: {n_excluded} ({n_excluded/len(train_masks)*100:.1f}%)")
    
    # Check some specific sequences around the seizure
    seizure_minute = 60
    analysis_start = max(0, seizure_minute - 15)
    analysis_end = min(len(timestamps), seizure_minute + 35)
    
    print(f"\nSequences around seizure time (target minute {seizure_minute}):")
    print(f"{'Seq#':<4} {'Target Min':<10} {'Bin Active':<10} {'Trainable':<10}")
    print("-" * 40)
    
    for i in range(min(20, len(timestamps))):  # Show first 20 sequences
        target_minute = int(timestamps[i] / 60)
        active_bin = np.argmax(labels[i]) + 1 if np.any(labels[i]) else 'None'
        trainable = 'Yes' if train_masks[i] else 'No'
        
        print(f"{i:<4} {target_minute:<10} {active_bin:<10} {trainable:<10}")
    
    print(f"\nKey Observations:")
    print(f"- Sequences targeting ictal minutes should be excluded (training_mask=False)")
    print(f"- Sequences targeting post-ictal minutes should be excluded")
    print(f"- Only sequences targeting pre-ictal and normal minutes should be trainable")

if __name__ == "__main__":
    test_end_to_end()