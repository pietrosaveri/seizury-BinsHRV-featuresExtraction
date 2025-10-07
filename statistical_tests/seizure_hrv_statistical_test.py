#!/usr/bin/env python3
"""
Seizure HRV Statistical Analysis

This script performs comprehensive HRV feature analysis to determine if HRV features
have a statistical correlation with seizures using PERMANOVA testing.

Analysis Pipeline:
1. Select runs with sz_foc_ia_nm seizure events (≥50 min after recording start, first seizure only)
2. Cut recording at seizure onset (exclude seizure period)
3. Define 25-minute pre-seizure window and control window
4. Apply comprehensive preprocessing: merge short RR, outlier removal, mean removal, detrending
5. Extract windowed HRV features (3, 5, 25-minute windows) for each minute
6. Compare pre-seizure vs control windows using PERMANOVA

Author: AI Assistant
Date: 2024
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import warnings
from scipy import signal, interpolate
from scipy.interpolate import PchipInterpolator, Akima1DInterpolator
from scipy.signal import lombscargle
from sklearn.preprocessing import StandardScaler
from skbio import DistanceMatrix
from skbio.stats.distance import permanova
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Add parent directory to path to import our modules
parent_dir = Path(__file__).absolute().parent.parent
sys.path.append(str(parent_dir))

# Import our custom modules
from data_processing_pipeline import DataDiscovery, AnnotationProcessor
from ecg_processing import ECGProcessor
from hrv_features import HRVFeatureExtractor

class SeizureHRVAnalyzer:
    """
    Comprehensive HRV analysis for seizure prediction research.
    
    Implements the complete pipeline from data selection through statistical testing.
    """
    
    def __init__(self, data_root: str, pre_seizure_window_min: int = 25):
        """
        Initialize the seizure HRV analyzer.
        
        Args:
            data_root: Path to dataset root directory
            pre_seizure_window_min: Pre-seizure window duration in minutes (default: 25)
        """
        self.data_root = data_root
        self.pre_seizure_window_min = pre_seizure_window_min
        self.pre_seizure_window_sec = pre_seizure_window_min * 60
        
        # Initialize processors
        self.discovery = DataDiscovery(data_root)
        self.annotation_processor = AnnotationProcessor()
        self.ecg_processor = ECGProcessor()
        self.hrv_extractor = HRVFeatureExtractor(resampling_rate=4.0)
        
        # Analysis parameters
        self.min_seizure_time_sec = 3000  # 50 minutes
        self.rr_merge_threshold_ms = 250
        
        # Results storage
        self.results = {}
        self.analysis_metadata = {}
        
    def find_valid_seizure_runs(self) -> List[Dict]:
        """
        Find all runs with valid sz_foc_ia_nm seizures.
        
        Criteria:
        1. Event type = "sz_foc_ia_nm"
        2. Seizure occurs ≥50 minutes after recording start
        3. Only first seizure per recording is considered
        
        Returns:
            List of dictionaries containing run info and seizure details
        """
        print("="*80)
        print("FINDING VALID SEIZURE RUNS")
        print("="*80)
        
        # Scan dataset and get matched runs
        self.discovery.scan_dataset()
        matched_runs = self.discovery.match_runs()
        
        valid_seizure_runs = []
        
        print(f"Scanning {len(matched_runs)} runs for valid sz_foc_ia_nm seizures...")
        
        for run in matched_runs:
            if not run['annotation_file']:
                continue
            
            # Debug: Check if EDF/ECG file exists
            if 'ecg_file' not in run or not run['ecg_file']:
                print(f"  ⚠ Skipping {run.get('subject', 'unknown')}/ses-{run.get('session', 'unknown')}/run-{run.get('run', 'unknown')}: No ECG file")
                continue
                
            try:
                # Load annotations
                annotations = self.annotation_processor.load_annotations(run['annotation_file'])
                
                # Fallback to direct CSV if needed
                if annotations.empty:
                    try:
                        annotations = pd.read_csv(run['annotation_file'], sep='\\t')
                    except Exception:
                        continue
                
                if annotations.empty or 'eventType' not in annotations.columns:
                    continue
                
                # Filter for sz_foc_ia_nm events
                sz_events = annotations[annotations['eventType'] == 'sz_foc_ia_nm'].copy()
                
                # Debug: Check what event types we have
                unique_events = annotations['eventType'].unique()
                print(f"    Event types in file: {unique_events}")
                
                if sz_events.empty or 'onset' not in sz_events.columns:
                    if sz_events.empty:
                        print(f"    No sz_foc_ia_nm events found (have {len(annotations)} total events)")
                    else:
                        print(f"    No 'onset' column in sz_foc_ia_nm events")
                    continue
                
                # Apply time filter (≥50 minutes)
                print(f"    Found {len(sz_events)} sz_foc_ia_nm events")
                print(f"    Onset times: {sz_events['onset'].tolist()}")
                print(f"    Min seizure time threshold: {self.min_seizure_time_sec} seconds ({self.min_seizure_time_sec/60:.1f} min)")
                
                valid_sz_events = sz_events[sz_events['onset'] >= self.min_seizure_time_sec].copy()
                
                if valid_sz_events.empty:
                    print(f"    No seizures after {self.min_seizure_time_sec/60:.1f} minutes")
                    continue

                
                
                # Get first seizure (earliest onset)
                first_seizure = valid_sz_events.loc[valid_sz_events['onset'].idxmin()]
                
                # Verify we have enough data before seizure for analysis
                seizure_time = first_seizure['onset']
                if seizure_time < (self.pre_seizure_window_sec + 600):  # Need extra buffer
                    continue

                
                # Store valid run info (use ecg_file, not edf_file)
                run_info = {
                    'subject': run['subject'],
                    'session': run['session'], 
                    'run': run['run'],
                    'ecg_file': run['ecg_file'],  # Fixed: use ecg_file not edf_file
                    'annotation_file': run['annotation_file'],
                    'seizure_onset': seizure_time,
                    'seizure_duration': first_seizure.get('duration', 0),
                    'seizure_onset_minutes': seizure_time / 60.0
                }
                
                valid_seizure_runs.append(run_info)
                
                print(f"  ✓ {run['subject']}/ses-{run['session']}/run-{run['run']}: "
                      f"sz_foc_ia_nm at {seizure_time/60:.1f} min")
                
            except Exception as e:
                print(f"  ✗ Error processing {run.get('subject', 'unknown')}: {e}")
                continue
        
        print(f"\\nFound {len(valid_seizure_runs)} valid seizure runs")
        
        # Store for later use
        self.valid_seizure_runs = valid_seizure_runs
        return valid_seizure_runs
    
    def extract_rr_intervals(self, run_info: Dict, end_time_sec: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract RR intervals from ECG file up to specified end time.
        
        Args:
            run_info: Dictionary containing run information
            end_time_sec: End time in seconds (if None, use seizure onset)
            
        Returns:
            Tuple of (rr_intervals, rr_times) in seconds
        """
        if end_time_sec is None:
            end_time_sec = run_info['seizure_onset']
        
        try:
            # Load ECG data using MNE (same approach as data_processing_pipeline.py)
            import mne
            
            raw_ecg = mne.io.read_raw_edf(run_info['ecg_file'], preload=True, verbose=False)
            
            # Resample if needed
            target_fs = 256  # Standard sampling rate for ECG processing
            if raw_ecg.info['sfreq'] != target_fs:
                raw_ecg.resample(target_fs, verbose=False)
            
            # Get ECG data (assume single channel or take first channel)
            ecg_data = raw_ecg.get_data()[0]
            fs = raw_ecg.info['sfreq']
            
            # Clear raw object to free memory
            del raw_ecg
            
            if len(ecg_data) == 0:
                return np.array([]), np.array([])
            
            # Extract segment up to end time
            max_samples = int(end_time_sec * fs)
            if max_samples >= len(ecg_data):
                max_samples = len(ecg_data) - 1
            
            ecg_segment = ecg_data[:max_samples]
            
            # Process ECG to get tachogram
            tachogram_result = self.ecg_processor.process_ecg_to_tachogram(ecg_segment)
            
            if len(tachogram_result['filtered_rr']) == 0:
                return np.array([]), np.array([])
            
            # Get RR intervals and times
            rr_intervals = tachogram_result['filtered_rr']
            rr_times = tachogram_result['filtered_times']
            
            # Filter by end time
            time_mask = rr_times <= end_time_sec
            rr_intervals = rr_intervals[time_mask]
            rr_times = rr_times[time_mask]
            
            return rr_intervals, rr_times
            
        except Exception as e:
            print(f"    Error extracting RR intervals: {e}")
            import traceback
            traceback.print_exc()
            return np.array([]), np.array([])
    
    # Copy preprocessing functions from PSD comparison notebook
    def merge_short_rr_intervals(self, rr_intervals: np.ndarray, rr_times: np.ndarray, 
                                threshold_ms: float = 250) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Merge RR intervals that are <= threshold_ms with the adjacent shortest RR interval.
        EXACTLY as implemented in psd_method_comparison.ipynb
        """
        if len(rr_intervals) < 2:
            return rr_intervals, rr_times, {'n_merged': 0}
        
        threshold_s = threshold_ms / 1000.0  # Convert to seconds
        merged_rr = rr_intervals.copy()
        merged_times = rr_times.copy()
        n_merged = 0
        
        i = 0
        while i < len(merged_rr):
            if merged_rr[i] <= threshold_s:
                # Find the adjacent RR interval to merge with (choose the shortest)
                left_idx = i - 1 if i > 0 else None
                right_idx = i + 1 if i < len(merged_rr) - 1 else None
                
                if left_idx is not None and right_idx is not None:
                    # Both neighbors exist, choose the shortest
                    if merged_rr[left_idx] <= merged_rr[right_idx]:
                        merge_idx = left_idx
                    else:
                        merge_idx = right_idx
                elif left_idx is not None:
                    # Only left neighbor exists
                    merge_idx = left_idx
                elif right_idx is not None:
                    # Only right neighbor exists
                    merge_idx = right_idx
                else:
                    # No neighbors (shouldn't happen with len >= 2)
                    i += 1
                    continue
                
                # Merge the intervals
                merged_rr[merge_idx] = merged_rr[merge_idx] + merged_rr[i]
                
                # Remove the short interval
                merged_rr = np.delete(merged_rr, i)
                merged_times = np.delete(merged_times, i)
                
                n_merged += 1
                
                # Don't increment i since we removed an element
                if merge_idx > i:
                    # If we merged with right neighbor, continue from same position
                    continue
                else:
                    # If we merged with left neighbor, we need to adjust position
                    i = max(0, i - 1)
            else:
                i += 1
        
        # Recalculate timeline based on merged RR intervals if any merging occurred
        if n_merged > 0:
            # Recalculate timeline to reflect the merged intervals
            corrected_merged_times = np.zeros_like(merged_times)
            corrected_merged_times[0] = merged_times[0]  # Keep the same starting time
            
            # Calculate cumulative sum of merged RR intervals for proper timeline
            for i in range(1, len(merged_rr)):
                corrected_merged_times[i] = corrected_merged_times[i-1] + merged_rr[i-1]
            
            merged_times = corrected_merged_times
        
        merge_info = {
            'n_merged': n_merged,
            'threshold_ms': threshold_ms,
            'original_count': len(rr_intervals),
            'final_count': len(merged_rr)
        }
        
        return merged_rr, merged_times, merge_info
    
    def remove_outliers_iqr(self, rr_intervals: np.ndarray, rr_times: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Remove outliers from RR intervals using the Interquartile Range (IQR) method.
        EXACTLY as implemented in psd_method_comparison.ipynb
        """
        if len(rr_intervals) < 4:
            return rr_intervals, rr_times, np.ones(len(rr_intervals), dtype=bool)
        
        # Calculate quartiles
        Q1 = np.percentile(rr_intervals, 25)
        Q3 = np.percentile(rr_intervals, 75)
        IQR = Q3 - Q1
        
        # Define outlier bounds
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Create mask for outliers
        outlier_mask_low = rr_intervals < lower_bound
        outlier_mask_high = rr_intervals > upper_bound
        outlier_mask = outlier_mask_low | outlier_mask_high
        
        # Replace outliers with appropriate quartile values
        cleaned_rr = rr_intervals.copy()
        cleaned_rr[outlier_mask_low] = Q1  # Replace low outliers with Q1
        cleaned_rr[outlier_mask_high] = Q3  # Replace high outliers with Q3
        
        # Recalculate timeline based on corrected RR intervals
        cleaned_times = np.zeros_like(rr_times)
        cleaned_times[0] = rr_times[0]  # Keep the same starting time
        
        # Calculate cumulative sum of corrected RR intervals for proper timeline
        for i in range(1, len(cleaned_rr)):
            cleaned_times[i] = cleaned_times[i-1] + cleaned_rr[i-1]
        
        return cleaned_rr, cleaned_times, ~outlier_mask  # Return non-outlier mask
    
    def preprocess_rr_intervals(self, rr_intervals: np.ndarray, rr_times: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Comprehensive preprocessing of RR intervals for spectral analysis.
        EXACTLY as implemented in psd_method_comparison.ipynb
        """
        # Step 1: Merge short RR intervals before outlier detection
        merged_rr, merged_times, merge_info = self.merge_short_rr_intervals(
            rr_intervals, rr_times, self.rr_merge_threshold_ms)
        
        # Step 2: Handle outliers by replacing with quartiles
        clean_rr, clean_times, non_outlier_mask = self.remove_outliers_iqr(merged_rr, merged_times)
        
        if len(clean_rr) < 4:
            return np.array([]), np.array([]), {}
        
        # Step 3: Remove mean (centering)
        original_mean = np.mean(clean_rr)
        centered_rr = clean_rr - original_mean
        
        # Recalculate timeline after mean removal
        centered_times = np.zeros_like(clean_times)
        centered_times[0] = clean_times[0]  # Keep the same starting time
        
        # Calculate cumulative sum of centered RR intervals for proper timeline
        rr_for_timeline = centered_rr + original_mean  # Restore original durations for timeline
        for i in range(1, len(centered_rr)):
            centered_times[i] = centered_times[i-1] + rr_for_timeline[i-1]
        
        # Step 4: High-pass filter detrending (remove frequencies < 0.003 Hz)
        from scipy import signal as scipy_signal
        
        # First interpolate to uniform sampling for filtering
        time_span = centered_times[-1] - centered_times[0]
        n_samples = len(centered_times)
        original_fs = n_samples / time_span
        
        # Use at least 4 Hz for proper filtering
        target_fs = max(4.0, original_fs * 2)
        
        # Create uniform time grid for filtering
        uniform_times_filter = np.linspace(centered_times[0], centered_times[-1], 
                                         int(time_span * target_fs))
        
        # Interpolate data to uniform grid
        interp_for_filter = np.interp(uniform_times_filter, centered_times, centered_rr)
        
        # Design high-pass filter to remove frequencies < 0.003 Hz
        cutoff_freq = 0.003  # Hz
        nyquist_freq = target_fs / 2
        normalized_cutoff = cutoff_freq / nyquist_freq
        
        # Ensure normalized frequency is valid
        if normalized_cutoff >= 1.0:
            detrended_rr = scipy_signal.detrend(centered_rr, type='linear')
            trend_method = "linear (fallback)"
        else:
            try:
                sos = scipy_signal.butter(4, normalized_cutoff, btype='high', output='sos')
                filtered_uniform = scipy_signal.sosfiltfilt(sos, interp_for_filter)
                detrended_rr = np.interp(centered_times, uniform_times_filter, filtered_uniform)
                trend_method = f"high-pass filter (cutoff: {cutoff_freq} Hz)"
            except Exception:
                detrended_rr = scipy_signal.detrend(centered_rr, type='linear')
                trend_method = "linear (fallback)"
        
        # Preprocessing summary
        preprocessing_info = {
            'n_merged': merge_info['n_merged'],
            'merge_percentage': merge_info['n_merged'] / len(rr_intervals) * 100,
            'n_outliers_replaced': np.sum(~non_outlier_mask),
            'outlier_percentage': np.sum(~non_outlier_mask) / len(merged_rr) * 100,
            'original_mean': original_mean,
            'detrend_method': trend_method,
            'cutoff_frequency': cutoff_freq,
            'final_mean': np.mean(detrended_rr),
            'final_std': np.std(detrended_rr)
        }
        
        return detrended_rr, centered_times, preprocessing_info
    
    def interpolate_rr_intervals_akima(self, rr_intervals: np.ndarray, rr_times: np.ndarray, 
                                      sampling_rate: float = 4.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Interpolate RR intervals using Akima method with comprehensive preprocessing.
        EXACTLY as implemented in psd_method_comparison.ipynb
        """
        # Apply comprehensive preprocessing (same as Lomb-Scargle)
        processed_rr, processed_times, preproc_info = self.preprocess_rr_intervals(rr_intervals, rr_times)
        
        # Handle empty or insufficient input after preprocessing
        if len(processed_rr) == 0 or len(processed_times) == 0:
            return np.array([]), np.array([])
            
        if len(processed_rr) < 2 or len(processed_times) < 2:
            return np.array([]), np.array([])
        
        # Create uniform time vector at resampling rate
        start_time = processed_times[0]
        end_time = processed_times[-1]
        time_span = end_time - start_time
        
        if time_span <= 0:
            return np.array([]), np.array([])
            
        uniform_times = np.arange(start_time, end_time, 1/sampling_rate)
        
        if len(uniform_times) < 2:
            return np.array([]), np.array([])
        
        try:
            # Akima: Akima spline interpolation
            interp_func = Akima1DInterpolator(processed_times, processed_rr)
            interpolated_rr = interp_func(uniform_times, extrapolate=True)
            
            # Remove any invalid values
            valid_mask = np.isfinite(interpolated_rr)
            if not np.any(valid_mask):
                return np.array([]), np.array([])
                
            return interpolated_rr[valid_mask], uniform_times[valid_mask]
            
        except Exception as e:
            return np.array([]), np.array([])


if __name__ == "__main__":
    # Example usage
    DATA_ROOT = "/Volumes/Seizury/ds005873"
    
    if not os.path.exists(DATA_ROOT):
        print(f"Error: Data root directory not found: {DATA_ROOT}")
        print("Please update the DATA_ROOT variable.")
        sys.exit(1)
    
    analyzer = SeizureHRVAnalyzer(DATA_ROOT, pre_seizure_window_min=25)
    
    # Find valid seizure runs
    valid_runs = analyzer.find_valid_seizure_runs()
    
    if len(valid_runs) == 0:
        print("No valid seizure runs found!")
        sys.exit(1)
    
    print(f"\\nReady to analyze {len(valid_runs)} seizure runs...")
    print("Next steps: Implement feature extraction and statistical testing")