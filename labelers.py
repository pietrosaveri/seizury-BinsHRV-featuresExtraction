#!/usr/bin/env python3
"""
Fixed SPH (Seizure Prediction Horizon) Labeling Module

This module implements a Fixed SPH labeler that targets seizure prediction 
at exactly 3 minutes (180 seconds) before seizure onset with a narrow 
tolerance window.
"""

import numpy as np
import pandas as pd
from typing import Tuple


class FixedSPHLabeler:
    """
    Fixed SPH (Seizure Prediction Horizon) labeler.
    
    Targets seizure prediction at exactly SPH seconds before seizure onset
    with a configurable tolerance window around the target.
    
    Label definitions:
    - 0: Normal (all other times)
    - 1: Pre-seizure target window (t_target ± label_width/2)  
    - 2: Ictal (during seizure)
    """
    
    def __init__(self, sampling_rate: int = 256, sph_seconds: int = 180, 
                 label_width_seconds: int = 30, skip_overlap: bool = True):
        """
        Initialize FixedSPHLabeler.
        
        Args:
            sampling_rate: Sampling rate in Hz
            sph_seconds: Seizure prediction horizon in seconds (default: 180s = 3 min)
            label_width_seconds: Width of positive label window in seconds (default: 30s)
            skip_overlap: If True, skip marking positive labels that overlap with ictal period
        """
        self.sampling_rate = sampling_rate
        self.sph_seconds = sph_seconds
        self.label_width_seconds = label_width_seconds
        self.skip_overlap = skip_overlap
        
    def create_labels(self, raw_data, seizure_events: pd.DataFrame) -> np.ndarray:
        """
        Create Fixed SPH labels for seizure prediction.
        
        Args:
            raw_data: MNE Raw object or data with timing information
            seizure_events: DataFrame with columns 'onset' and 'duration'
            
        Returns:
            numpy array of labels (0=normal, 1=pre-seizure target, 2=ictal)
        """
        n_samples = len(raw_data.times)
        labels = np.zeros(n_samples, dtype=int)  # 0 = Normal
        
        if seizure_events.empty:
            return labels
            
        for _, event in seizure_events.iterrows():
            onset_time = event['onset']
            duration = event.get('duration', 0)
            
            # Convert times to sample indices
            onset_sample = int(onset_time * self.sampling_rate)
            duration_samples = int(duration * self.sampling_rate)
            end_sample = onset_sample + duration_samples
            
            # Label seizure period (label 2 - ictal)
            seizure_start = max(0, onset_sample)
            seizure_end = min(n_samples, end_sample)
            labels[seizure_start:seizure_end] = 2
            
            # Calculate SPH target time: t_target = onset_time - sph_seconds
            t_target = onset_time - self.sph_seconds
            
            if t_target >= 0:
                # Calculate SPH target window: t_target ± label_width/2
                target_start_time = t_target - self.label_width_seconds / 2
                target_end_time = t_target + self.label_width_seconds / 2
                
                if target_start_time >= 0:
                    target_start_sample = int(target_start_time * self.sampling_rate)
                    target_end_sample = int(target_end_time * self.sampling_rate)
                    
                    target_start_sample = max(0, target_start_sample)
                    target_end_sample = min(n_samples, target_end_sample)
                    
                    # Check for overlap with ictal period
                    if self.skip_overlap:
                        # Only label if target window doesn't overlap with seizure
                        if target_end_sample <= seizure_start:
                            labels[target_start_sample:target_end_sample] = 1
                    else:
                        # Label regardless of overlap (may overwrite ictal labels)
                        labels[target_start_sample:target_end_sample] = 1
                        # Re-apply ictal labels to maintain priority
                        labels[seizure_start:seizure_end] = 2
        
        return labels
    
    def get_label_statistics(self, labels: np.ndarray, recording_duration: float) -> dict:
        """
        Calculate statistics for the labeling.
        
        Args:
            labels: Label array 
            recording_duration: Duration of recording in seconds
            
        Returns:
            Dictionary with label statistics
        """
        label_counts = np.bincount(labels, minlength=3)
        total_samples = len(labels)
        
        stats = {
            'total_duration_sec': recording_duration,
            'total_samples': total_samples,
            'sampling_rate': self.sampling_rate,
            'sph_seconds': self.sph_seconds,
            'label_width_seconds': self.label_width_seconds,
            'normal_samples': int(label_counts[0]),
            'pre_seizure_samples': int(label_counts[1]), 
            'ictal_samples': int(label_counts[2]),
            'normal_percent': label_counts[0] / total_samples * 100,
            'pre_seizure_percent': label_counts[1] / total_samples * 100,
            'ictal_percent': label_counts[2] / total_samples * 100,
            'normal_duration_sec': label_counts[0] / self.sampling_rate,
            'pre_seizure_duration_sec': label_counts[1] / self.sampling_rate,
            'ictal_duration_sec': label_counts[2] / self.sampling_rate,
        }
        
        return stats
    
    def validate_labeling(self, labels: np.ndarray, seizure_events: pd.DataFrame) -> dict:
        """
        Validate that labeling was performed correctly.
        
        Args:
            labels: Label array
            seizure_events: Seizure events DataFrame
            
        Returns:
            Dictionary with validation results
        """
        validation = {
            'n_seizures': len(seizure_events),
            'target_windows_created': 0,
            'targets_with_labels': 0,
            'ictal_periods_labeled': 0
        }
        
        for _, event in seizure_events.iterrows():
            onset_time = event['onset']
            duration = event.get('duration', 0)
            
            # Check if target window was created
            t_target = onset_time - self.sph_seconds
            if t_target >= 0:
                target_start_time = t_target - self.label_width_seconds / 2
                if target_start_time >= 0:
                    validation['target_windows_created'] += 1
                    
                    # Check if target window has positive labels
                    target_start_sample = int(target_start_time * self.sampling_rate)
                    target_end_sample = int((t_target + self.label_width_seconds / 2) * self.sampling_rate)
                    target_end_sample = min(len(labels), target_end_sample)
                    
                    if target_end_sample > target_start_sample:
                        target_labels = labels[target_start_sample:target_end_sample]
                        if np.any(target_labels == 1):
                            validation['targets_with_labels'] += 1
            
            # Check if ictal period was labeled
            onset_sample = int(onset_time * self.sampling_rate)
            duration_samples = int(duration * self.sampling_rate)
            end_sample = min(len(labels), onset_sample + duration_samples)
            
            if end_sample > onset_sample:
                ictal_labels = labels[onset_sample:end_sample]
                if np.any(ictal_labels == 2):
                    validation['ictal_periods_labeled'] += 1
        
        return validation


class WindowCenterLabeler:
    """
    Utility class for assigning window labels based on window center sampling.
    
    This replaces majority voting with center-point sampling for more precise
    temporal alignment.
    """
    
    @staticmethod
    def assign_window_labels(data: np.ndarray, labels: np.ndarray, 
                           window_size: float, stride: float, 
                           sampling_rate: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create windowed data with labels assigned by sampling at window center.
        
        Args:
            data: Input data array (n_channels, n_samples)
            labels: Label array (n_samples,)
            window_size: Window size in seconds
            stride: Stride between windows in seconds
            sampling_rate: Sampling rate in Hz
            
        Returns:
            Tuple of (windowed_data, windowed_labels, window_center_times)
        """
        if len(data.shape) == 1:
            data = data.reshape(1, -1)  # Single channel
        
        n_channels, n_samples = data.shape
        samples_per_window = int(window_size * sampling_rate)
        step_size = int(stride * sampling_rate)
        
        # Calculate number of windows
        n_windows = (n_samples - samples_per_window) // step_size + 1
        
        if n_windows <= 0:
            return np.array([]), np.array([]), np.array([])
        
        # Create windows
        windowed_data = np.zeros((n_windows, n_channels, samples_per_window))
        windowed_labels = np.zeros(n_windows, dtype=int)
        window_center_times = np.zeros(n_windows)
        
        for i in range(n_windows):
            start_idx = i * step_size
            end_idx = start_idx + samples_per_window
            center_idx = start_idx + samples_per_window // 2
            
            # Extract window
            windowed_data[i] = data[:, start_idx:end_idx]
            
            # Assign label based on center sample
            windowed_labels[i] = labels[center_idx]
            
            # Calculate window center time
            window_center_times[i] = center_idx / sampling_rate
        
        return windowed_data, windowed_labels, window_center_times