# HRV-Based Seizure Proximity Probability Prediction

This pipeline processes ECG data to create minute-level feature packages for predicting seizure proximity probability distributions using multi-temporal HRV features.

## Overview

Instead of predicting exact time-to-seizure, the model predicts probability distributions over how far each minute is from the nearest seizure. This approach provides more robust seizure monitoring by giving probabilistic assessments of seizure risk.

## Approach

### 1. Minute-Level Feature Packages

Each minute in the recording gets a comprehensive feature package containing:
- **Multi-temporal HRV features** calculated from different historical windows
- **Seizure proximity bin label** based on distance to nearest seizure
- **Feature availability mask** indicating which features could be calculated

### 2. Multi-Temporal Feature Windows

Features are calculated from different historical time windows to capture both short-term and long-term cardiac variability patterns:

#### 3-Minute Historical Features
Calculated from 3 minutes of history before the current minute:
- `RRMean_3` - Mean RR interval (ms)
- `RRMin_3` - Minimum RR interval (ms)  
- `RRMax_3` - Maximum RR interval (ms)
- `RRVar_3` - RR interval variance (ms²)
- `RMSSD_3` - Root mean square of successive differences (ms)
- `SDNN_3` - Standard deviation of RR intervals (ms)
- `SDSD_3` - Standard deviation of successive differences (ms)
- `NN50_3` - Number of successive RR intervals differing by >50ms
- `pNN50_3` - Percentage of NN50
- `SampEn_3` - Sample entropy

#### 5-Minute Historical Features
Calculated from 5 minutes of history before the current minute:
- `ApEn_5` - Approximate entropy
- `SD1_5` - Poincaré plot SD1 (short-term variability, ms)
- `SD2_5` - Poincaré plot SD2 (long-term variability, ms)
- `SD1toSD2_5` - SD1/SD2 ratio

#### 10-Minute Historical Features
Calculated from 10 minutes of history before the current minute:
- `TOTAL_POWER_10` - Total power in all frequency bands (ms²)
- `LF_NORM_10` - Normalized low frequency power (%)
- `HF_NORM_10` - Normalized high frequency power (%)
- `LF_POWER_10` - Low frequency power (0.04-0.15 Hz, ms²)
- `HF_POWER_10` - High frequency power (0.15-0.4 Hz, ms²)
- `LF_TO_HF_10` - LF/HF ratio
- `VLF_POWER_10` - Very low frequency power (0.003-0.04 Hz, ms²)
- `VLF_NORM_10` - Normalized VLF power (%)

### 3. Seizure Proximity Bins

Each minute is assigned to one of four bins based on distance to the closest seizure:

- **Bin 1**: 0-5 minutes to seizure (immediate seizure risk)
- **Bin 2**: 5-30 minutes to seizure (elevated seizure risk)
- **Bin 3**: 30-60 minutes to seizure (moderate seizure risk)
- **Bin 4**: >60 minutes to seizure (low seizure risk / no seizure in sight)

Bins are represented as one-hot encoded vectors:
- Bin 1: `[1, 0, 0, 0]`
- Bin 2: `[0, 1, 0, 0]`
- Bin 3: `[0, 0, 1, 0]`
- Bin 4: `[0, 0, 0, 1]`

### 4. Overlapping Windows for Training

Minute packages are organized into 60-minute overlapping windows:
- **Window size**: 60 minutes
- **Overlap**: 30 minutes (50% overlap)
- **Dense supervision**: Each minute within the window gets its own bin label
- **Training**: Loss is averaged across all 60 minutes in each window

### 5. Feature Availability Masking

When insufficient historical data is available (e.g., at recording start):
- Missing features are set to `0.0`
- A binary mask indicates feature availability
- Model learns to handle missing features appropriately

## Output Format

Each row in the output CSV represents one minute within a 60-minute window:

```csv
subject_id,recording_id,window_start_time,window_end_time,minute_time,minute_in_window,
RRMean_3,RRMin_3,RRMax_3,RRVar_3,RMSSD_3,SDNN_3,SDSD_3,NN50_3,pNN50_3,SampEn_3,
ApEn_5,SD1_5,SD2_5,SD1toSD2_5,
TOTAL_POWER_10,LF_NORM_10,HF_NORM_10,LF_POWER_10,HF_POWER_10,LF_TO_HF_10,VLF_POWER_10,VLF_NORM_10,
bin_1,bin_2,bin_3,bin_4,mask
```

## Model Training

The LSTM model predicts probability distributions:
- **Input**: Sequences of minute-level feature packages
- **Output**: 4-class probability distribution over seizure proximity bins
- **Loss**: Cross-entropy loss averaged across all minutes in each window
- **Evaluation**: Probability calibration and seizure detection performance

## Usage

```bash
# Process dataset with seizure proximity bin labeling
python3 data_processing_pipeline.py --top-n-patients 10

# Build LSTM sequences (if using the sequence builder)
python3 lstm_sequences.py --input-dir hrv_features --output-dir sequences
```

## Advantages

1. **Probabilistic predictions**: More informative than binary seizure/non-seizure
2. **Multi-temporal features**: Captures both short and long-term cardiac patterns
3. **Dense supervision**: Every minute contributes to training
4. **Robust to missing data**: Explicit masking of unavailable features
5. **Clinical relevance**: Bins correspond to meaningful time horizons for intervention

## Files

- `data_processing_pipeline.py`: Main processing pipeline
- `hrv_features.py`: HRV feature extraction (unchanged)
- `ecg_processing.py`: ECG R-peak detection and tachogram extraction (unchanged)
- `lstm_sequences.py`: LSTM sequence preparation (may need updates)
- `README.md`: This documentation