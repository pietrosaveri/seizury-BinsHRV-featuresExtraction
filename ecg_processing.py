#!/usr/bin/env python3
"""
ECG R-peak Detection and Tachogram Extraction Module

This module implements robust R-peak detection and RR interval extraction 
from ECG signals using multiple detection algorithms and artifact filtering.
"""

import numpy as np
import pandas as pd
from scipy import signal
from scipy.interpolate import interp1d
import warnings
from typing import Tuple, Optional, Dict, List

try:
    import neurokit2 as nk
    HAS_NEUROKIT = True
except ImportError:
    HAS_NEUROKIT = False
    
try:
    import biosppy.signals.ecg as biosppy_ecg
    HAS_BIOSPPY = True  
except ImportError:
    HAS_BIOSPPY = False


class ECGProcessor:
    """
    ECG signal processing for R-peak detection and tachogram extraction.
    """
    
    def __init__(self, sampling_rate: int = 256, 
                 lowcut: float = 0.5, highcut: float = 40.0,
                 method: str = 'auto'):
        """
        Initialize ECG processor.
        
        Args:
            sampling_rate: Sampling rate in Hz
            lowcut: Low cutoff frequency for bandpass filter (Hz)
            highcut: High cutoff frequency for bandpass filter (Hz)  
            method: R-peak detection method ('auto', 'neurokit', 'biosppy', 'basic')
        """
        self.sampling_rate = sampling_rate
        self.lowcut = lowcut
        self.highcut = highcut
        self.method = method
        
        # Artifact filtering thresholds
        self.min_rr_sec = 0.2  # Minimum physiological RR interval (300 bpm)
        self.max_rr_sec = 2.0  # Maximum physiological RR interval (30 bpm)
        
    def bandpass_filter(self, ecg_signal: np.ndarray) -> np.ndarray:
        """
        Apply bandpass filter to ECG signal.
        
        Args:
            ecg_signal: Raw ECG signal
            
        Returns:
            Filtered ECG signal
        """
        nyquist = self.sampling_rate / 2.0
        low = self.lowcut / nyquist
        high = self.highcut / nyquist
        
        # Design Butterworth bandpass filter
        b, a = signal.butter(4, [low, high], btype='band')
        
        # Apply zero-phase filtering
        filtered_signal = signal.filtfilt(b, a, ecg_signal)
        
        return filtered_signal
    
    def detect_r_peaks_neurokit(self, ecg_signal: np.ndarray) -> np.ndarray:
        """
        Detect R-peaks using NeuroKit2.
        
        Args:
            ecg_signal: Filtered ECG signal
            
        Returns:
            Array of R-peak indices
        """
        if not HAS_NEUROKIT:
            raise ImportError("NeuroKit2 not available. Install with: pip install neurokit2")
        
        # Clean the signal first
        cleaned_ecg = nk.ecg_clean(ecg_signal, sampling_rate=self.sampling_rate)
        
        # Detect R-peaks
        _, rpeaks = nk.ecg_peaks(cleaned_ecg, sampling_rate=self.sampling_rate, 
                                method='neurokit')
        
        return rpeaks['ECG_R_Peaks']
    
    def detect_r_peaks_biosppy(self, ecg_signal: np.ndarray) -> np.ndarray:
        """
        Detect R-peaks using BioSPPy.
        
        Args:
            ecg_signal: Filtered ECG signal
            
        Returns:
            Array of R-peak indices
        """
        if not HAS_BIOSPPY:
            raise ImportError("BioSPPy not available. Install with: pip install biosppy")
        
        # Process ECG signal
        out = biosppy_ecg.ecg(signal=ecg_signal, sampling_rate=self.sampling_rate, show=False)
        
        return out['rpeaks']
    
    def detect_r_peaks_basic(self, ecg_signal: np.ndarray) -> np.ndarray:
        """
        Basic R-peak detection using scipy.signal.find_peaks.
        
        Args:
            ecg_signal: Filtered ECG signal
            
        Returns:
            Array of R-peak indices
        """
        # Normalize signal
        ecg_norm = (ecg_signal - np.mean(ecg_signal)) / np.std(ecg_signal)
        
        # Set minimum distance between peaks (physiological constraint)
        min_distance = int(0.3 * self.sampling_rate)  # 300ms minimum RR interval
        
        # Find peaks
        peaks, properties = signal.find_peaks(ecg_norm, 
                                            height=np.std(ecg_norm) * 0.5,
                                            distance=min_distance)
        
        return peaks
    
    def detect_r_peaks(self, ecg_signal: np.ndarray) -> np.ndarray:
        """
        Detect R-peaks using the specified method.
        
        Args:
            ecg_signal: Raw ECG signal
            
        Returns:
            Array of R-peak indices
        """
        # Apply bandpass filter
        filtered_ecg = self.bandpass_filter(ecg_signal)
        
        # Choose detection method
        if self.method == 'auto':
            # Try methods in order of preference
            if HAS_NEUROKIT:
                method = 'neurokit'
            elif HAS_BIOSPPY:
                method = 'biosppy'
            else:
                method = 'basic'
        else:
            method = self.method
        
        # Detect R-peaks
        if method == 'neurokit':
            r_peaks = self.detect_r_peaks_neurokit(filtered_ecg)
        elif method == 'biosppy':
            r_peaks = self.detect_r_peaks_biosppy(filtered_ecg)
        elif method == 'basic':
            r_peaks = self.detect_r_peaks_basic(filtered_ecg)
        else:
            raise ValueError(f"Unknown detection method: {method}")
        
        return r_peaks
    
    def compute_rr_intervals(self, r_peaks: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute RR intervals from R-peak indices.
        
        Args:
            r_peaks: Array of R-peak indices
            
        Returns:
            Tuple of (rr_intervals_sec, rr_times_sec)
        """
        if len(r_peaks) < 2:
            return np.array([]), np.array([])
        
        # Convert indices to times
        r_times = r_peaks / self.sampling_rate
        
        # Compute RR intervals (differences between consecutive R-peaks)
        rr_intervals = np.diff(r_times)
        
        # RR interval times (assign to the second R-peak of each interval)
        rr_times = r_times[1:]
        
        return rr_intervals, rr_times
    
    def filter_artifacts(self, rr_intervals: np.ndarray, rr_times: np.ndarray,
                        method: str = 'threshold') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Filter artifacts from RR interval series.
        
        Args:
            rr_intervals: RR intervals in seconds
            rr_times: Times corresponding to RR intervals
            method: Artifact filtering method ('threshold', 'median', 'iqr')
            
        Returns:
            Tuple of (filtered_rr, filtered_times, artifact_mask)
        """
        if len(rr_intervals) == 0:
            return np.array([]), np.array([]), np.array([])
        
        artifact_mask = np.zeros(len(rr_intervals), dtype=bool)
        
        if method == 'threshold':
            # Simple threshold-based filtering
            artifact_mask = (rr_intervals < self.min_rr_sec) | (rr_intervals > self.max_rr_sec)
            
        elif method == 'median':
            # Median-based filtering (remove outliers beyond 3 MAD)
            median_rr = np.median(rr_intervals)
            mad = np.median(np.abs(rr_intervals - median_rr))
            threshold = 3 * mad
            artifact_mask = np.abs(rr_intervals - median_rr) > threshold
            
        elif method == 'iqr':
            # Interquartile range-based filtering
            q25, q75 = np.percentile(rr_intervals, [25, 75])
            iqr = q75 - q25
            lower_bound = q25 - 1.5 * iqr
            upper_bound = q75 + 1.5 * iqr
            artifact_mask = (rr_intervals < lower_bound) | (rr_intervals > upper_bound)
        
        # Apply physiological constraints as well
        phys_mask = (rr_intervals < self.min_rr_sec) | (rr_intervals > self.max_rr_sec)
        artifact_mask = artifact_mask | phys_mask
        
        # Filter out artifacts
        clean_mask = ~artifact_mask
        filtered_rr = rr_intervals[clean_mask]
        filtered_times = rr_times[clean_mask]
        
        return filtered_rr, filtered_times, artifact_mask
    
    def interpolate_artifacts(self, rr_intervals: np.ndarray, rr_times: np.ndarray,
                            artifact_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Interpolate artifacts in RR interval series.
        
        Args:
            rr_intervals: Original RR intervals
            rr_times: Original RR times
            artifact_mask: Boolean mask indicating artifacts
            
        Returns:
            Tuple of (interpolated_rr, interpolated_times)
        """
        if len(rr_intervals) == 0 or np.all(artifact_mask):
            return np.array([]), np.array([])
        
        # Use clean data for interpolation
        clean_mask = ~artifact_mask
        clean_rr = rr_intervals[clean_mask]
        clean_times = rr_times[clean_mask]
        
        if len(clean_rr) < 2:
            return clean_rr, clean_times
        
        # Interpolate missing values
        interp_func = interp1d(clean_times, clean_rr, kind='linear', 
                              bounds_error=False, fill_value='extrapolate')
        
        # Interpolate at original time points
        interpolated_rr = interp_func(rr_times)
        
        return interpolated_rr, rr_times
    
    def process_ecg_to_tachogram(self, ecg_signal: np.ndarray, 
                               filter_method: str = 'threshold',
                               interpolate: bool = False) -> Dict:
        """
        Complete processing pipeline from ECG to tachogram.
        
        Args:
            ecg_signal: Raw ECG signal (single channel)
            filter_method: Artifact filtering method
            interpolate: Whether to interpolate artifacts
            
        Returns:
            Dictionary with processing results
        """
        # Detect R-peaks
        r_peaks = self.detect_r_peaks(ecg_signal)
        
        if len(r_peaks) < 2:
            return {
                'r_peaks': r_peaks,
                'r_times': np.array([]),
                'rr_intervals': np.array([]),
                'rr_times': np.array([]),
                'filtered_rr': np.array([]),
                'filtered_times': np.array([]),
                'artifact_mask': np.array([]),
                'stats': {'n_peaks': len(r_peaks), 'n_rr': 0, 'artifact_rate': 0.0}
            }
        
        # Compute RR intervals
        rr_intervals, rr_times = self.compute_rr_intervals(r_peaks)
        
        # Filter artifacts
        filtered_rr, filtered_times, artifact_mask = self.filter_artifacts(
            rr_intervals, rr_times, method=filter_method)
        
        # Optionally interpolate artifacts
        if interpolate and np.any(artifact_mask):
            interpolated_rr, interpolated_times = self.interpolate_artifacts(
                rr_intervals, rr_times, artifact_mask)
        else:
            interpolated_rr, interpolated_times = filtered_rr, filtered_times
        
        # Calculate statistics
        stats = {
            'n_peaks': len(r_peaks),
            'n_rr': len(rr_intervals),
            'n_artifacts': np.sum(artifact_mask),
            'artifact_rate': np.sum(artifact_mask) / len(rr_intervals) if len(rr_intervals) > 0 else 0.0,
            'mean_rr_ms': np.mean(filtered_rr) * 1000 if len(filtered_rr) > 0 else 0.0,
            'std_rr_ms': np.std(filtered_rr) * 1000 if len(filtered_rr) > 0 else 0.0,
            'mean_hr_bpm': 60.0 / np.mean(filtered_rr) if len(filtered_rr) > 0 and np.mean(filtered_rr) > 0 else 0.0
        }
        
        return {
            'r_peaks': r_peaks,
            'r_times': r_peaks / self.sampling_rate,
            'rr_intervals': rr_intervals,
            'rr_times': rr_times,
            'filtered_rr': interpolated_rr,
            'filtered_times': interpolated_times,
            'artifact_mask': artifact_mask,
            'stats': stats
        }
    
    def extract_tachogram_window(self, rr_intervals: np.ndarray, rr_times: np.ndarray,
                               window_start: float, window_end: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract RR intervals within a specific time window.
        
        Args:
            rr_intervals: RR intervals in seconds
            rr_times: Times corresponding to RR intervals  
            window_start: Window start time in seconds
            window_end: Window end time in seconds
            
        Returns:
            Tuple of (windowed_rr, windowed_times)
        """
        if len(rr_times) == 0:
            return np.array([]), np.array([])
        
        # Find RR intervals within the window
        mask = (rr_times >= window_start) & (rr_times <= window_end)
        
        windowed_rr = rr_intervals[mask]
        windowed_times = rr_times[mask]
        
        return windowed_rr, windowed_times


def test_ecg_processor():
    """Test function for ECG processor with synthetic data."""
    
    # Generate synthetic ECG signal
    fs = 256
    duration = 60  # 60 seconds
    t = np.arange(0, duration, 1/fs)
    
    # Simulate ECG with R-peaks every ~0.8 seconds (75 bpm)
    hr = 75  # beats per minute
    rr_interval = 60 / hr  # seconds
    
    ecg = np.zeros_like(t)
    r_times = np.arange(rr_interval, duration, rr_interval)
    
    # Add R-wave spikes
    for r_time in r_times:
        idx = int(r_time * fs)
        if idx < len(ecg):
            ecg[idx-2:idx+3] = [0.1, 0.5, 1.0, 0.5, 0.1]  # Simple R-wave shape
    
    # Add noise
    ecg += 0.1 * np.random.randn(len(ecg))
    
    # Test processor
    processor = ECGProcessor(sampling_rate=fs)
    result = processor.process_ecg_to_tachogram(ecg)
    
    print("ECG Processor Test Results:")
    print(f"Number of R-peaks detected: {result['stats']['n_peaks']}")
    print(f"Number of RR intervals: {result['stats']['n_rr']}")
    print(f"Artifact rate: {result['stats']['artifact_rate']:.1%}")
    print(f"Mean RR interval: {result['stats']['mean_rr_ms']:.1f} ms")
    print(f"Mean heart rate: {result['stats']['mean_hr_bpm']:.1f} bpm")
    
    return result


if __name__ == "__main__":
    test_ecg_processor()