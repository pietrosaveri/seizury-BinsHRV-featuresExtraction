#!/usr/bin/env python3
"""
Summary of LSTM Sequence Builder Updates

IMPORTANT CHANGES to match 60-minute window approach:

1. DEFAULT PARAMETERS UPDATED:
   OLD: seq_len=36, stride=1, history=180s (3 minutes)
   NEW: seq_len=60, stride=30, history=3600s (60 minutes)

2. DENSE SUPERVISION IMPLEMENTED:
   OLD: (n_sequences, 4) - single prediction per sequence
   NEW: (n_sequences, 60, 4) - prediction for every minute in 60-minute window

3. WINDOW-BASED PROCESSING:
   - Automatically detects overlapping windows from pipeline data
   - Groups by (window_start_time, window_end_time) 
   - Creates one sequence per 60-minute window
   - Maintains 30-minute overlap between sequences

4. TRAINING MASK HANDLING:
   - Preserves ictal/post-ictal exclusions at minute level
   - Shape: (n_windows, 60) for dense supervision
   - Statistics calculated on individual minutes, not sequences

5. IMPROVED ARCHITECTURE:
   - Window-based sequences for pipeline data (preferred)
   - Sliding window fallback for backward compatibility
   - Proper class weight calculation for dense format
   - Enhanced metadata tracking

KEY BENEFITS:
✅ Matches 60-minute pipeline windows exactly
✅ No data waste (uses all 60 minutes per window) 
✅ Dense supervision for rich training signal
✅ Proper overlap handling (30-minute stride)
✅ Maintains medical masking constraints
✅ Ready for modern seizure prediction models

USAGE:
builder = LSTMSequenceBuilder()  # Uses new defaults
sequences, labels, timestamps, masks = builder.create_sequences_from_recording(df)

# Output shapes:
# sequences: (n_windows, 60, n_features) 
# labels: (n_windows, 60, 4) - dense bin predictions
# timestamps: (n_windows, 60) - time for each minute
# masks: (n_windows, 60) - trainable minute mask
"""

if __name__ == "__main__":
    print(__doc__)