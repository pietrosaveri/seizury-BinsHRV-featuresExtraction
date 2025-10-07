#!/usr/bin/env python3
"""
HRV Feature Extraction and Statistical Testing Module

This module extends the SeizureHRVAnalyzer with feature extraction and statistical testing capabilities.
Implements windowed feature extraction and PERMANOVA testing for pre-seizure vs control comparison.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import euclidean_distances
from skbio import DistanceMatrix
from skbio.stats.distance import permanova
import warnings
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

class HRVFeatureExtractor_Enhanced:
    """
    Enhanced HRV feature extractor with windowed analysis and ln(LF/HF) ratio.
    """
    
    def __init__(self):
        self.feature_definitions = {
            # 3-minute window features
            'RRMean_3': 'Mean RR interval (ms)',
            'RRMin_3': 'Minimum RR interval (ms)',
            'RRMax_3': 'Maximum RR interval (ms)', 
            'RRVar_3': 'RR interval variance (ms²)',
            'RMSSD_3': 'Root mean square of successive differences (ms)',
            'SDNN_3': 'Standard deviation of RR intervals (ms)',
            'SDSD_3': 'Standard deviation of successive differences (ms)',
            'NN50_3': 'Number of successive RR intervals differing by >50ms',
            'pNN50_3': 'Percentage of NN50',
            'SampEn_3': 'Sample entropy',
            
            # 5-minute window features  
            'ApEn_5': 'Approximate entropy',
            'SD1_5': 'Poincaré plot SD1 (short-term variability, ms)',
            'SD2_5': 'Poincaré plot SD2 (long-term variability, ms)',
            'SD1toSD2_5': 'SD1/SD2 ratio',
            'LF_NORM_5': 'Normalized low frequency power (%)',
            'HF_NORM_5': 'Normalized high frequency power (%)',
            'LF_POWER_5': 'Low frequency power (0.04-0.15 Hz, ms²)',
            'HF_POWER_5': 'High frequency power (0.15-0.4 Hz, ms²)',
            'ln_LF_TO_HF_5': 'ln(LF/HF ratio)',
            
            # 25-minute window features
            'TOTAL_POWER_25': 'Total power in all frequency bands (ms²)',
            'VLF_POWER_25': 'Very low frequency power (0.003-0.04 Hz, ms²)',
            'VLF_NORM_25': 'Normalized VLF power (%)'
        }
    
    def extract_windowed_features(self, rr_intervals: np.ndarray, rr_times: np.ndarray, 
                                 window_center_time: float, analyzer) -> Dict[str, float]:
        """
        Extract HRV features for different window sizes centered on a given time point.
        
        Args:
            rr_intervals: Preprocessed RR intervals (seconds)
            rr_times: Corresponding times (seconds)
            window_center_time: Center time for feature extraction (seconds)
            analyzer: SeizureHRVAnalyzer instance for preprocessing methods
            
        Returns:
            Dictionary of extracted features
        """
        features = {}
        
        # Define window sizes in seconds
        windows = {
            3: 3 * 60,    # 3 minutes
            5: 5 * 60,    # 5 minutes  
            25: 25 * 60   # 25 minutes
        }
        
        for window_label, window_size in windows.items():
            # Define window bounds (going backwards from center time)
            window_end = window_center_time
            window_start = window_center_time - window_size
            
            # Extract RR intervals within window
            window_mask = (rr_times >= window_start) & (rr_times <= window_end)
            window_rr = rr_intervals[window_mask]
            window_times = rr_times[window_mask]
            
            if len(window_rr) < 5:  # Minimum requirement
                # Fill with NaN for insufficient data
                self._fill_window_features_nan(features, window_label)
                continue
            
            # Extract features for this window
            if window_label == 3:
                self._extract_3min_features(window_rr, window_times, features)
            elif window_label == 5:
                self._extract_5min_features(window_rr, window_times, features, analyzer)
            elif window_label == 25:
                self._extract_25min_features(window_rr, window_times, features, analyzer)
        
        return features
    
    def _extract_3min_features(self, rr_intervals: np.ndarray, rr_times: np.ndarray, 
                              features: Dict[str, float]):
        """Extract 3-minute window time-domain features."""
        # Convert to milliseconds
        rr_ms = rr_intervals * 1000
        
        # Basic statistics
        features['RRMean_3'] = np.mean(rr_ms)
        features['RRMin_3'] = np.min(rr_ms)
        features['RRMax_3'] = np.max(rr_ms)
        features['RRVar_3'] = np.var(rr_ms, ddof=1)
        features['SDNN_3'] = np.std(rr_ms, ddof=1)
        
        # Successive differences
        if len(rr_ms) > 1:
            rr_diff = np.diff(rr_ms)
            features['RMSSD_3'] = np.sqrt(np.mean(rr_diff**2))
            features['SDSD_3'] = np.std(rr_diff, ddof=1)
            features['NN50_3'] = np.sum(np.abs(rr_diff) > 50)
            features['pNN50_3'] = (features['NN50_3'] / len(rr_diff)) * 100
        else:
            features['RMSSD_3'] = 0
            features['SDSD_3'] = 0
            features['NN50_3'] = 0
            features['pNN50_3'] = 0
        
        # Sample entropy
        features['SampEn_3'] = self._calculate_sample_entropy(rr_intervals)
    
    def _extract_5min_features(self, rr_intervals: np.ndarray, rr_times: np.ndarray, 
                              features: Dict[str, float], analyzer):
        """Extract 5-minute window features including frequency domain and nonlinear."""
        # Approximate entropy
        features['ApEn_5'] = self._calculate_approximate_entropy(rr_intervals)
        
        # Poincaré plot features
        sd1, sd2, sd1_to_sd2 = self._calculate_poincare_features(rr_intervals)
        features['SD1_5'] = sd1 * 1000  # Convert to ms
        features['SD2_5'] = sd2 * 1000  # Convert to ms
        features['SD1toSD2_5'] = sd1_to_sd2
        
        # Frequency domain features using Akima interpolation
        try:
            # Use analyzer's preprocessing and interpolation
            interpolated_rr, interpolated_times = analyzer.interpolate_rr_intervals_akima(
                rr_intervals, rr_times, sampling_rate=4.0)
            
            if len(interpolated_rr) > 10:
                # Compute Welch PSD
                from scipy import signal
                freqs, psd = signal.welch(
                    interpolated_rr,
                    fs=4.0,
                    nperseg=min(256, len(interpolated_rr)//2),
                    noverlap=None,
                    window='hann',
                    detrend='constant',
                    scaling='density'
                )
                
                # Define frequency bands
                lf_band = [0.04, 0.15]
                hf_band = [0.15, 0.4]
                
                # Calculate band powers
                lf_power = self._compute_band_power(psd, freqs, lf_band) * 1e6  # Convert to ms²
                hf_power = self._compute_band_power(psd, freqs, hf_band) * 1e6
                total_power = np.trapz(psd, freqs) * 1e6
                
                # Normalized powers
                if total_power > 0:
                    features['LF_NORM_5'] = (lf_power / total_power) * 100
                    features['HF_NORM_5'] = (hf_power / total_power) * 100
                else:
                    features['LF_NORM_5'] = 0
                    features['HF_NORM_5'] = 0
                
                features['LF_POWER_5'] = lf_power
                features['HF_POWER_5'] = hf_power
                
                # ln(LF/HF) ratio - CHANGED FROM ORIGINAL
                if hf_power > 0 and lf_power > 0:
                    features['ln_LF_TO_HF_5'] = np.log(lf_power / hf_power)
                else:
                    features['ln_LF_TO_HF_5'] = 0
            else:
                # Fill with default values if interpolation failed
                features['LF_NORM_5'] = 0
                features['HF_NORM_5'] = 0
                features['LF_POWER_5'] = 0
                features['HF_POWER_5'] = 0
                features['ln_LF_TO_HF_5'] = 0
                
        except Exception as e:
            # Fill with default values on error
            features['LF_NORM_5'] = 0
            features['HF_NORM_5'] = 0
            features['LF_POWER_5'] = 0
            features['HF_POWER_5'] = 0
            features['ln_LF_TO_HF_5'] = 0
    
    def _extract_25min_features(self, rr_intervals: np.ndarray, rr_times: np.ndarray, 
                               features: Dict[str, float], analyzer):
        """Extract 25-minute window frequency domain features."""
        try:
            # Use analyzer's preprocessing and interpolation
            interpolated_rr, interpolated_times = analyzer.interpolate_rr_intervals_akima(
                rr_intervals, rr_times, sampling_rate=4.0)
            
            if len(interpolated_rr) > 10:
                # Compute Welch PSD with appropriate parameters for longer segments
                from scipy import signal
                freqs, psd = signal.welch(
                    interpolated_rr,
                    fs=4.0,
                    nperseg=min(2048, len(interpolated_rr)//4),  # Larger window for 25min
                    noverlap=None,
                    window='hann',
                    detrend='constant',
                    scaling='density'
                )
                
                # Define frequency bands
                vlf_band = [0.003, 0.04]
                
                # Calculate powers
                vlf_power = self._compute_band_power(psd, freqs, vlf_band) * 1e6  # Convert to ms²
                total_power = np.trapz(psd, freqs) * 1e6
                
                features['VLF_POWER_25'] = vlf_power
                features['TOTAL_POWER_25'] = total_power
                
                # Normalized VLF power
                if total_power > 0:
                    features['VLF_NORM_25'] = (vlf_power / total_power) * 100
                else:
                    features['VLF_NORM_25'] = 0
            else:
                features['VLF_POWER_25'] = 0
                features['TOTAL_POWER_25'] = 0
                features['VLF_NORM_25'] = 0
                
        except Exception as e:
            features['VLF_POWER_25'] = 0
            features['TOTAL_POWER_25'] = 0
            features['VLF_NORM_25'] = 0
    
    def _compute_band_power(self, psd: np.ndarray, freqs: np.ndarray, band: List[float]) -> float:
        """Compute power in a specific frequency band."""
        band_mask = (freqs >= band[0]) & (freqs <= band[1])
        if np.any(band_mask):
            return np.trapz(psd[band_mask], freqs[band_mask])
        return 0.0
    
    def _calculate_sample_entropy(self, rr_intervals: np.ndarray, m: int = 2, r: float = 0.2) -> float:
        """Calculate sample entropy."""
        try:
            if len(rr_intervals) < m + 1:
                return 0.0
            
            N = len(rr_intervals)
            patterns_m = []
            patterns_m1 = []
            
            # Create patterns of length m and m+1
            for i in range(N - m):
                patterns_m.append(rr_intervals[i:i+m])
                if i < N - m:
                    patterns_m1.append(rr_intervals[i:i+m+1])
            
            # Calculate distances and matches
            patterns_m = np.array(patterns_m)
            patterns_m1 = np.array(patterns_m1)
            
            def count_matches(patterns, r_thresh):
                n_patterns = len(patterns)
                matches = 0
                for i in range(n_patterns):
                    for j in range(i+1, n_patterns):
                        if np.max(np.abs(patterns[i] - patterns[j])) <= r_thresh:
                            matches += 1
                return matches
            
            r_thresh = r * np.std(rr_intervals)
            
            matches_m = count_matches(patterns_m, r_thresh)
            matches_m1 = count_matches(patterns_m1, r_thresh)
            
            if matches_m == 0 or matches_m1 == 0:
                return 0.0
            
            return -np.log(matches_m1 / matches_m)
            
        except Exception:
            return 0.0
    
    def _calculate_approximate_entropy(self, rr_intervals: np.ndarray, m: int = 2, r: float = 0.2) -> float:
        """Calculate approximate entropy."""
        try:
            if len(rr_intervals) < m + 1:
                return 0.0
            
            def _maxdist(xi, xj, m):
                return max([abs(ua - va) for ua, va in zip(xi, xj)])
            
            def _phi(m):
                patterns = np.array([rr_intervals[i:i+m] for i in range(len(rr_intervals) - m + 1)])
                C = np.zeros(len(patterns))
                
                r_thresh = r * np.std(rr_intervals)
                
                for i, pattern_i in enumerate(patterns):
                    template_match_count = 0
                    for pattern_j in patterns:
                        if _maxdist(pattern_i, pattern_j, m) <= r_thresh:
                            template_match_count += 1
                    C[i] = template_match_count / float(len(patterns))
                
                phi = (len(patterns) - m + 1.0)**(-1) * sum([np.log(c) for c in C])
                return phi
            
            return _phi(m) - _phi(m + 1)
            
        except Exception:
            return 0.0
    
    def _calculate_poincare_features(self, rr_intervals: np.ndarray) -> Tuple[float, float, float]:
        """Calculate Poincaré plot features (SD1, SD2, SD1/SD2)."""
        try:
            if len(rr_intervals) < 2:
                return 0.0, 0.0, 0.0
            
            # Get successive RR intervals
            rr1 = rr_intervals[:-1]
            rr2 = rr_intervals[1:]
            
            # Calculate SD1 and SD2
            sd1 = np.std(rr1 - rr2, ddof=1) / np.sqrt(2)
            sd2 = np.std(rr1 + rr2, ddof=1) / np.sqrt(2)
            
            sd1_to_sd2 = sd1 / sd2 if sd2 > 0 else 0.0
            
            return sd1, sd2, sd1_to_sd2
            
        except Exception:
            return 0.0, 0.0, 0.0
    
    def _fill_window_features_nan(self, features: Dict[str, float], window_label: int):
        """Fill features with NaN for insufficient data."""
        if window_label == 3:
            nan_features = ['RRMean_3', 'RRMin_3', 'RRMax_3', 'RRVar_3', 'RMSSD_3', 
                           'SDNN_3', 'SDSD_3', 'NN50_3', 'pNN50_3', 'SampEn_3']
        elif window_label == 5:
            nan_features = ['ApEn_5', 'SD1_5', 'SD2_5', 'SD1toSD2_5', 'LF_NORM_5', 
                           'HF_NORM_5', 'LF_POWER_5', 'HF_POWER_5', 'ln_LF_TO_HF_5']
        elif window_label == 25:
            nan_features = ['TOTAL_POWER_25', 'VLF_POWER_25', 'VLF_NORM_25']
        
        for feature in nan_features:
            features[feature] = np.nan


class StatisticalTester:
    """
    Statistical testing using PERMANOVA for HRV feature comparison.
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
    
    def prepare_feature_matrix(self, pre_seizure_features: List[Dict], 
                              control_features: List[Dict]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare feature matrices for statistical testing.
        
        Args:
            pre_seizure_features: List of feature dictionaries for pre-seizure windows
            control_features: List of feature dictionaries for control windows
            
        Returns:
            Tuple of (feature_matrix, group_labels, feature_names)
        """
        # Combine all features
        all_features = pre_seizure_features + control_features
        
        if not all_features:
            return np.array([]), np.array([]), []
        
        # Define metadata columns to exclude from the numerical matrix
        metadata_cols = {'run_id', 'window_type', 'minute_offset', 'center_time', 'window_index'}
        
        # Extract feature names from the first entry, excluding metadata
        feature_names = sorted([key for key in all_features[0].keys() if key not in metadata_cols])
        
        if not feature_names:
            print("  Error: No valid numeric feature columns found.")
            return np.array([]), np.array([]), []
            
        # Create a numeric feature matrix and convert to DataFrame for easier handling
        df = pd.DataFrame([[d.get(name, np.nan) for name in feature_names] for d in all_features], columns=feature_names)

        # Drop columns that are entirely NaN
        df.dropna(axis=1, how='all', inplace=True)
        if df.shape[1] < len(feature_names):
            print(f"  Dropped {len(feature_names) - df.shape[1]} columns that were all NaN.")
            feature_names = df.columns.tolist() # Update feature names

        # Impute any remaining NaNs with column means
        if df.isnull().values.any():
            print(f"  Found {df.isnull().values.sum()} missing values (NaNs). Imputing with column means.")
            df.fillna(df.mean(), inplace=True)

        # Convert DataFrame back to numpy array
        feature_matrix = df.to_numpy()

        # Remove columns with zero variance (all constant values)
        variances = np.var(feature_matrix, axis=0)
        non_zero_var_mask = variances > 1e-9  # Use a small threshold for float comparison
        
        if not np.all(non_zero_var_mask):
            original_feature_count = len(feature_names)
            feature_matrix = feature_matrix[:, non_zero_var_mask]
            # Update feature_names to reflect removed columns
            feature_names = [name for i, name in enumerate(feature_names) if non_zero_var_mask[i]]
            print(f"  Dropped {original_feature_count - len(feature_names)} columns with zero variance.")

        # Create group labels
        group_labels = np.array(['pre_seizure'] * len(pre_seizure_features) + ['control'] * len(control_features))
        
        # Standardize features
        feature_matrix_scaled = self.scaler.fit_transform(feature_matrix)
        
        return feature_matrix_scaled, group_labels, feature_names
    
    def run_permanova(self, feature_matrix: np.ndarray, group_labels: np.ndarray, 
                     n_permutations: int = 999) -> Dict:
        """
        Run PERMANOVA test on feature matrix.
        
        Args:
            feature_matrix: Standardized feature matrix
            group_labels: Group labels for each sample
            n_permutations: Number of permutations for testing
            
        Returns:
            Dictionary containing test results
        """
        try:
            # Calculate Euclidean distance matrix
            distances = euclidean_distances(feature_matrix)
            distance_matrix = DistanceMatrix(distances)
            
            # Run PERMANOVA
            results = permanova(distance_matrix, group_labels, permutations=n_permutations)
            
            return {
                'test_statistic': results['test statistic'],
                'p_value': results['p-value'],
                'n_permutations': results['number of permutations'],
                'n_samples': len(group_labels),
                'n_features': feature_matrix.shape[1]
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'test_statistic': np.nan,
                'p_value': np.nan,
                'n_permutations': n_permutations,
                'n_samples': len(group_labels) if len(group_labels) > 0 else 0,
                'n_features': feature_matrix.shape[1] if len(feature_matrix.shape) > 1 else 0
            }
    
    def analyze_feature_importance(self, feature_matrix: np.ndarray, group_labels: np.ndarray, 
                                  feature_names: List[str]) -> pd.DataFrame:
        """
        Analyze individual feature differences between groups.
        
        Args:
            feature_matrix: Standardized feature matrix
            group_labels: Group labels
            feature_names: Names of features
            
        Returns:
            DataFrame with feature importance analysis
        """
        results = []
        
        pre_seizure_mask = group_labels == 'pre_seizure'
        control_mask = group_labels == 'control'
        
        for i, feature_name in enumerate(feature_names):
            pre_seizure_vals = feature_matrix[pre_seizure_mask, i]
            control_vals = feature_matrix[control_mask, i]
            
            # Calculate statistics
            pre_mean = np.mean(pre_seizure_vals)
            control_mean = np.mean(control_vals)
            effect_size = (pre_mean - control_mean) / np.sqrt((np.var(pre_seizure_vals) + np.var(control_vals)) / 2)
            
            # Mann-Whitney U test
            try:
                statistic, p_value = stats.mannwhitneyu(pre_seizure_vals, control_vals, alternative='two-sided')
            except Exception:
                statistic, p_value = np.nan, np.nan
            
            results.append({
                'Feature': feature_name,
                'PreSeizure_Mean': pre_mean,
                'Control_Mean': control_mean,
                'Effect_Size': effect_size,
                'MWU_Statistic': statistic,
                'MWU_P_Value': p_value
            })
        
        return pd.DataFrame(results)


if __name__ == "__main__":
    print("HRV Feature Extraction and Statistical Testing Module")
    print("This module provides enhanced feature extraction and statistical testing capabilities.")
    print("Import this module in your main analysis script.")