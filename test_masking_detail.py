#!/usr/bin/env python3
"""
Detailed test to verify ictal/postictal masking around seizure times.
"""

import pandas as pd
import numpy as np
from lstm_sequences import LSTMSequenceBuilder

def test_masking_around_seizure():
    """Test training masks specifically around seizure events."""
    
    print("Testing Training Masks Around Seizure Events")
    print("=" * 50)
    
    # Create 120 minutes of data with seizure at minute 60
    n_minutes = 120
    fake_data = []
    
    for minute in range(n_minutes):
        minute_time = minute * 60.0
        
        row = {
            'subject_id': 'sub-test',
            'recording_id': 'test_recording', 
            'window_start_time': 0.0,
            'window_end_time': 7200.0,  # 2 hours
            'minute_time': minute_time,
            'minute_in_window': minute,
            'RRMean_3': 800 + np.random.randn() * 10,
            'SDNN_3': 50 + np.random.randn() * 5,
            'LF_NORM_10': 0.5 + np.random.randn() * 0.05,
            'is_padded': False
        }
        
        # Seizure event at minute 60, lasting 1 minute
        if minute == 60:  # Ictal period
            row.update({
                'bin_1': 0, 'bin_2': 0, 'bin_3': 0, 'bin_4': 0,
                'is_ictal': 1, 'is_postictal': 0, 'recent_seizure_flag': 0,
                'training_mask': 0  # EXCLUDED from training
            })
        elif 60 < minute <= 90:  # Post-ictal (30 minutes)
            row.update({
                'bin_1': 0, 'bin_2': 0, 'bin_3': 0, 'bin_4': 0,
                'is_ictal': 0, 'is_postictal': 1, 'recent_seizure_flag': 1,
                'training_mask': 0  # EXCLUDED from training
            })
        elif 55 <= minute < 60:  # Pre-ictal (5 minutes before)
            row.update({
                'bin_1': 1, 'bin_2': 0, 'bin_3': 0, 'bin_4': 0,
                'is_ictal': 0, 'is_postictal': 0, 'recent_seizure_flag': 0,
                'training_mask': 1  # INCLUDED in training
            })
        elif 30 <= minute < 55:  # Pre-ictal (30-5 minutes before)
            row.update({
                'bin_1': 0, 'bin_2': 1, 'bin_3': 0, 'bin_4': 0,
                'is_ictal': 0, 'is_postictal': 0, 'recent_seizure_flag': 0,
                'training_mask': 1  # INCLUDED in training
            })
        else:  # Normal periods
            row.update({
                'bin_1': 0, 'bin_2': 0, 'bin_3': 0, 'bin_4': 1,
                'is_ictal': 0, 'is_postictal': 0, 'recent_seizure_flag': 0,
                'training_mask': 1  # INCLUDED in training
            })
        
        fake_data.append(row)
    
    df = pd.DataFrame(fake_data)
    
    # Build sequences
    builder = LSTMSequenceBuilder(seq_len=10, stride=1, normalize_features=False)
    sequences, labels, timestamps, train_masks = builder.create_sequences_from_recording(df)
    
    print(f"Created {len(sequences)} sequences from {n_minutes} minutes of data")
    
    # Analyze sequences around the seizure
    seizure_minute = 60
    
    print(f"\nDetailed Analysis Around Seizure (minute {seizure_minute}):")
    print(f"{'Seq#':<4} {'Target Min':<11} {'Bin':<8} {'Ictal':<6} {'PostIctal':<10} {'Trainable':<10} {'Notes'}")
    print("-" * 80)
    
    for i, timestamp in enumerate(timestamps):
        target_minute = int(timestamp / 60)
        
        # Only show sequences around the seizure
        if abs(target_minute - seizure_minute) <= 35:
            active_bin = np.argmax(labels[i]) + 1 if np.any(labels[i]) else 0
            trainable = 'Yes' if train_masks[i] else 'No'
            
            # Get status from original dataframe
            df_row = df[df['minute_time'] == timestamp].iloc[0]
            ictal = 'Yes' if df_row['is_ictal'] else 'No'
            postictal = 'Yes' if df_row['is_postictal'] else 'No'
            
            # Add notes
            if target_minute == 60:
                notes = "SEIZURE MINUTE"
            elif 60 < target_minute <= 90:
                notes = "Post-ictal refractory"
            elif 55 <= target_minute < 60:
                notes = "Pre-ictal (0-5 min)"
            elif 30 <= target_minute < 55:
                notes = "Pre-ictal (5-30 min)"
            else:
                notes = "Normal"
            
            print(f"{i:<4} {target_minute:<11} {active_bin:<8} {ictal:<6} {postictal:<10} {trainable:<10} {notes}")
    
    # Summary statistics
    n_trainable = train_masks.sum()
    n_excluded = len(train_masks) - n_trainable
    
    print(f"\n" + "="*50)
    print(f"SUMMARY:")
    print(f"Total sequences: {len(train_masks)}")
    print(f"Trainable sequences: {n_trainable} ({n_trainable/len(train_masks)*100:.1f}%)")
    print(f"Excluded sequences: {n_excluded} ({n_excluded/len(train_masks)*100:.1f}%)")
    
    # Check that ictal and post-ictal sequences are properly excluded
    ictal_sequences = 0
    postictal_sequences = 0
    
    for i, timestamp in enumerate(timestamps):
        target_minute = int(timestamp / 60)
        if target_minute == 60:  # Ictal
            ictal_sequences += 1
            if train_masks[i] == 1:
                print(f"ERROR: Ictal sequence {i} (minute {target_minute}) is marked as trainable!")
        elif 60 < target_minute <= 90:  # Post-ictal
            postictal_sequences += 1
            if train_masks[i] == 1:
                print(f"ERROR: Post-ictal sequence {i} (minute {target_minute}) is marked as trainable!")
    
    print(f"\nValidation Results:")
    print(f"- Ictal sequences found: {ictal_sequences} (all should be excluded)")
    print(f"- Post-ictal sequences found: {postictal_sequences} (all should be excluded)")
    print(f"✓ Training mask validation complete")

if __name__ == "__main__":
    test_masking_around_seizure()