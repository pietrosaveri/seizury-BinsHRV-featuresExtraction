#!/usr/bin/env python3
"""
HRV Feature Extraction Module

This module implements HRV feature extraction algorithms ported from MATLAB
to match the functionality described in the original research codebase.
Includes time-domain, frequency-domain, and nonlinear HRV features.
"""

import numpy as np
import pandas as pd
from scipy import signal, interpolate
from scipy.stats import entropy
import warnings
from typing import Dict, Tuple, Optional, List

try:
    from scipy.signal import lombscargle
    HAS_LOMBSCARGLE = True
except ImportError:
    HAS_LOMBSCARGLE = False


class HRVFeatureExtractor:
    """
    Extract HRV features from RR interval time series.
    
    Features implemented to match MATLAB codebase:
    - Time domain: NN50, pNN50, SDNN, RMSSD, SDSD, RRMean, RRMin, RRMax, RRVar
    - Frequency domain: TOTAL_POWER, VLF_POWER, LF_POWER, HF_POWER, LF_TO_HF
    - Nonlinear: SD1, SD2, SD1toSD2, SampEn, ApEn
    """
    
    def __init__(self, resampling_rate: float = 4.0):
        """
        Initialize HRV feature extractor.
        
        Args:
            resampling_rate: Rate for resampling tachogram for frequency analysis (Hz)
        """
        self.resampling_rate = resampling_rate
        
        # Frequency bands (Hz) - matching MATLAB definitions
        self.vlf_band = [0.003, 0.04]
        self.lf_band = [0.04, 0.15]  
        self.hf_band = [0.15, 0.4]
        
    def compute_time_features(self, rr_intervals: np.ndarray) -> Dict[str, float]:
        """
        Compute time-domain HRV features.
        
        Matches the output of time_features.m from MATLAB codebase.
        
        Args:
            rr_intervals: RR intervals in seconds
            
        Returns:
            Dictionary of time-domain features
        """
        if len(rr_intervals) < 2:
            return self._get_empty_time_features()
        
        # Convert to milliseconds for compatibility with MATLAB
        rr_ms = rr_intervals * 1000
        
        # Compute differences for short-term variability measures
        rr_diff = np.diff(rr_ms)
        
        # NN50: Number of successive RR intervals differing by more than 50ms
        nn50 = np.sum(np.abs(rr_diff) > 50)
        
        # pNN50: Percentage of NN50
        pnn50 = (nn50 / len(rr_diff)) * 100 if len(rr_diff) > 0 else 0.0
        
        # RMSSD: Root mean square of successive differences
        rmssd = np.sqrt(np.mean(rr_diff**2)) if len(rr_diff) > 0 else 0.0
        
        # SDNN: Standard deviation of RR intervals
        sdnn = np.std(rr_ms, ddof=1)  # Use sample std (N-1)
        
        # SDSD: Standard deviation of successive differences
        sdsd = np.std(rr_diff, ddof=1) if len(rr_diff) > 0 else 0.0
        
        # Basic statistics
        rr_mean = np.mean(rr_ms)
        rr_min = np.min(rr_ms)
        rr_max = np.max(rr_ms)
        rr_var = np.var(rr_ms, ddof=1)  # Sample variance
        
        return {
            'NN50': nn50,
            'pNN50': pnn50,
            'RMSSD': rmssd,
            'SDNN': sdnn,
            'SDSD': sdsd,
            'RRMean': rr_mean,
            'RRMin': rr_min,
            'RRMax': rr_max,
            'RRVar': rr_var
        }
    
    def interpolate_rr_intervals(self, rr_intervals: np.ndarray, 
                               rr_times: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Interpolate RR intervals to uniform sampling using cubic spline.
        
        This is essential for accurate frequency-domain analysis as recommended
        by the Task Force of the European Society of Cardiology.
        
        Args:
            rr_intervals: RR intervals in seconds
            rr_times: Times corresponding to RR intervals in seconds
            
        Returns:
            Tuple of (interpolated_rr, uniform_times)
        """
        # Handle empty or insufficient input
        if len(rr_intervals) == 0 or len(rr_times) == 0:
            return np.array([]), np.array([])
            
        if len(rr_intervals) < 2 or len(rr_times) < 2:
            # Need at least 2 points for interpolation
            return np.array([]), np.array([])
        
        # Create uniform time vector at resampling rate
        start_time = rr_times[0]
        end_time = rr_times[-1]
        time_span = end_time - start_time
        
        if time_span <= 0:
            return np.array([]), np.array([])
            
        uniform_times = np.arange(start_time, end_time, 1/self.resampling_rate)
        
        if len(uniform_times) < 2:
            # Not enough points for meaningful analysis
            return np.array([]), np.array([])
        
        # Cubic spline interpolation
        try:
            interp_func = interpolate.interp1d(rr_times, rr_intervals, 
                                             kind='cubic', 
                                             bounds_error=False,
                                             fill_value='extrapolate')
            interpolated_rr = interp_func(uniform_times)
            
            # Remove any invalid values
            valid_mask = np.isfinite(interpolated_rr)
            if not np.any(valid_mask):
                return np.array([]), np.array([])
                
            return interpolated_rr[valid_mask], uniform_times[valid_mask]
            
        except Exception as e:
            warnings.warn(f"RR interval interpolation failed: {e}")
            return np.array([]), np.array([])

    def compute_frequency_features(self, rr_intervals: np.ndarray, 
                                 rr_times: np.ndarray) -> Dict[str, float]:
        """
        Compute frequency-domain HRV features using interpolation-based PSD.
        
        Uses cubic spline interpolation for uniform sampling followed by
        Welch's method for PSD estimation, as recommended for HRV analysis.
        
        Args:
            rr_intervals: RR intervals in seconds
            rr_times: Times corresponding to RR intervals in seconds
            
        Returns:
            Dictionary of frequency-domain features
        """
        if len(rr_intervals) < 9:  # Minimum from MATLAB code
            return self._get_empty_frequency_features()
        
        try:
            # Use interpolation-based approach as primary method
            interpolated_rr, uniform_times = self.interpolate_rr_intervals(rr_intervals, rr_times)
            
            if len(interpolated_rr) < 10:
                # Fallback to Lomb-Scargle for very short segments
                return self._compute_lombscargle_features(rr_intervals, rr_times)
            
            # QUI DOBBIAMO SPLITTARE I CASI: DIVERSI MINUTI
            # Compute Welch PSD on interpolated data  
            freqs, psd = signal.welch(interpolated_rr, 
                                    fs=self.resampling_rate,
                                    nperseg=min(256, len(interpolated_rr)//2),
                                    noverlap=None,
                                    window='hann',
                                    detrend='constant')
            
            # Compute power in frequency bands
            total_power = self._compute_band_power(psd, freqs, [freqs[0], freqs[-1]])
            vlf_power = self._compute_band_power(psd, freqs, self.vlf_band)
            lf_power = self._compute_band_power(psd, freqs, self.lf_band)
            hf_power = self._compute_band_power(psd, freqs, self.hf_band)
            
            # Convert to ms^2 units as in MATLAB
            scale_factor = 1e6
            total_power *= scale_factor
            vlf_power *= scale_factor
            lf_power *= scale_factor
            hf_power *= scale_factor
            
            # Normalized power
            vlf_norm = (vlf_power / total_power) * 100 if total_power > 0 else 0.0
            lf_norm = (lf_power / total_power) * 100 if total_power > 0 else 0.0
            hf_norm = (hf_power / total_power) * 100 if total_power > 0 else 0.0
            
            # LF/HF ratio
            lf_to_hf = lf_power / hf_power if hf_power > 0 else 0.0
            
            return {
                'TOTAL_POWER': total_power,
                'VLF_POWER': vlf_power,
                'LF_POWER': lf_power,
                'HF_POWER': hf_power,
                'VLF_NORM': vlf_norm,
                'LF_NORM': lf_norm,
                'HF_NORM': hf_norm,
                'LF_TO_HF': lf_to_hf
            }
            
        except Exception as e:
            warnings.warn(f"Frequency analysis failed: {e}")
            return self._get_empty_frequency_features()
    
    def _compute_lombscargle_features(self, rr_intervals: np.ndarray, 
                                    rr_times: np.ndarray) -> Dict[str, float]:
        """
        Fallback method using Lomb-Scargle periodogram for very short segments.
        """
        try:
            # Use beat numbers as in original MATLAB code
            beat_numbers = np.arange(1, len(rr_intervals) + 1)
            
            # Frequency range for analysis
            f_min = 0.001  # Minimum frequency
            f_max = 0.5    # Maximum frequency
            freqs = np.linspace(f_min, f_max, 1000)
            
            # Compute Lomb-Scargle periodogram
            if HAS_LOMBSCARGLE:
                # Use angular frequencies for scipy.signal.lombscargle  
                omega = 2 * np.pi * freqs
                psd = lombscargle(beat_numbers, rr_intervals, omega, normalize=True)
            else:
                # Final fallback: simple interpolation + Welch
                psd, freqs = self._compute_welch_psd(rr_intervals, rr_times)
            
            # Compute power in frequency bands (same as main method)
            total_power = self._compute_band_power(psd, freqs, [freqs[0], freqs[-1]])
            vlf_power = self._compute_band_power(psd, freqs, self.vlf_band)
            lf_power = self._compute_band_power(psd, freqs, self.lf_band)
            hf_power = self._compute_band_power(psd, freqs, self.hf_band)
            
            # Convert to ms^2 units
            scale_factor = 1e6
            total_power *= scale_factor
            vlf_power *= scale_factor
            lf_power *= scale_factor
            hf_power *= scale_factor
            
            # Normalized power
            vlf_norm = (vlf_power / total_power) * 100 if total_power > 0 else 0.0
            lf_norm = (lf_power / total_power) * 100 if total_power > 0 else 0.0
            hf_norm = (hf_power / total_power) * 100 if total_power > 0 else 0.0
            
            # LF/HF ratio
            lf_to_hf = lf_power / hf_power if hf_power > 0 else 0.0
            
            return {
                'TOTAL_POWER': total_power,
                'VLF_POWER': vlf_power,
                'LF_POWER': lf_power,
                'HF_POWER': hf_power,
                'VLF_NORM': vlf_norm,
                'LF_NORM': lf_norm,
                'HF_NORM': hf_norm,
                'LF_TO_HF': lf_to_hf
            }
            
        except Exception as e:
            warnings.warn(f"Lomb-Scargle analysis failed: {e}")
            return self._get_empty_frequency_features()
    
    def _compute_welch_psd(self, rr_intervals: np.ndarray, 
                          rr_times: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Final fallback method using simple interpolation + Welch PSD.
        """
        try:
            # Use the dedicated interpolation method
            interpolated_rr, uniform_times = self.interpolate_rr_intervals(rr_intervals, rr_times)
            
            if len(interpolated_rr) < 10:
                # Return empty arrays if interpolation failed
                freqs = np.linspace(0.001, 0.5, 100)
                return np.zeros_like(freqs), freqs
            
            # Compute Welch PSD
            freqs, psd = signal.welch(interpolated_rr, 
                                    fs=self.resampling_rate, 
                                    nperseg=min(256, len(interpolated_rr)//2))
            
            return psd, freqs
            
        except Exception:
            # Ultimate fallback
            freqs = np.linspace(0.001, 0.5, 100)
            return np.zeros_like(freqs), freqs
    
    def _compute_band_power(self, psd: np.ndarray, freqs: np.ndarray, 
                           band: List[float]) -> float:
        """
        Compute power in a specific frequency band.
        """
        mask = (freqs >= band[0]) & (freqs <= band[1])
        if not np.any(mask):
            return 0.0
        
        # Integrate using trapezoidal rule
        band_psd = psd[mask]
        band_freqs = freqs[mask]
        
        if len(band_freqs) < 2:
            return 0.0
        
        power = np.trapz(band_psd, band_freqs)
        return max(0.0, power)  # Ensure non-negative
    
    def compute_poincare_features(self, rr_intervals: np.ndarray) -> Dict[str, float]:
        """
        Compute Poincaré plot features.
        
        Matches the output of poincare_plot.m from MATLAB codebase.
        
        Args:
            rr_intervals: RR intervals in seconds
            
        Returns:
            Dictionary with SD1, SD2, SD1toSD2
        """
        if len(rr_intervals) < 2:
            return {'SD1': 0.0, 'SD2': 0.0, 'SD1toSD2': 0.0}
        
        # Convert to milliseconds
        rr_ms = rr_intervals * 1000
        
        # Extract consecutive RR intervals
        rr_i = rr_ms[:-1]    # RR(i)
        rr_ii = rr_ms[1:]    # RR(i+1)
        
        # Compute SD1 and SD2 using the geometrical approach
        # SD1: standard deviation along the line perpendicular to identity line
        sd1 = np.std(rr_i - rr_ii, ddof=1) / np.sqrt(2)
        
        # SD2: standard deviation along the line of identity
        sd2 = np.std(rr_i + rr_ii, ddof=1) / np.sqrt(2)
        
        # SD1/SD2 ratio
        sd1_to_sd2 = sd1 / sd2 if sd2 > 0 else 0.0
        
        return {
            'SD1': sd1,
            'SD2': sd2,
            'SD1toSD2': sd1_to_sd2
        }
    
    def compute_sample_entropy(self, rr_intervals: np.ndarray, 
                             m: int = 2, r_tolerance: float = 0.2) -> float:
        """
        Compute Sample Entropy.
        
        Matches the output of sample_entropy.m from MATLAB codebase.
        
        Args:
            rr_intervals: RR intervals in seconds
            m: Pattern length (default: 2)
            r_tolerance: Tolerance for matching (default: 0.2)
            
        Returns:
            Sample entropy value
        """
        if len(rr_intervals) < m + 1:
            return 0.0
        
        # Normalize signal
        sig_norm = (rr_intervals - np.mean(rr_intervals)) / np.std(rr_intervals)
        n = len(sig_norm)
        
        # Tolerance based on standard deviation
        r = r_tolerance * np.std(rr_intervals)
        
        # Initialize counters
        a = 0  # Template matches of length m+1
        b = 0  # Template matches of length m
        
        # Compare all template pairs
        for i in range(n - m):
            for j in range(i + 1, n - m + 1):
                # Check match for length m
                d_m = max(abs(sig_norm[i + k] - sig_norm[j + k]) for k in range(m))
                
                if d_m <= r:
                    b += 1
                    
                    # Check match for length m+1 (if possible)
                    if j <= n - m - 1:
                        d_m1 = max(abs(sig_norm[i + k] - sig_norm[j + k]) for k in range(m + 1))
                        if d_m1 <= r:
                            a += 1
        
        # Calculate Sample Entropy
        if a == 0 or b == 0:
            return 0.0
        
        sample_entropy = -np.log((a / b) * ((n - m + 2) / (n - m)))
        return sample_entropy
    
    def compute_approximate_entropy(self, rr_intervals: np.ndarray,
                                  m: int = 2, r_tolerance: float = 0.2) -> float:
        """
        Compute Approximate Entropy.
        
        Args:
            rr_intervals: RR intervals in seconds
            m: Pattern length (default: 2)
            r_tolerance: Tolerance for matching (default: 0.2)
            
        Returns:
            Approximate entropy value
        """
        if len(rr_intervals) < m + 1:
            return 0.0
        
        # Normalize signal
        sig_norm = (rr_intervals - np.mean(rr_intervals)) / np.std(rr_intervals)
        n = len(sig_norm)
        
        # Tolerance
        r = r_tolerance * np.std(rr_intervals)
        
        def _maxdist(xi, xj, m):
            return max([abs(ua - va) for ua, va in zip(xi[0:m], xj[0:m])])
        
        def _phi(m):
            phi = 0.0
            for i in range(n - m + 1):
                template_i = sig_norm[i:i + m]
                matches = 0
                for j in range(n - m + 1):
                    template_j = sig_norm[j:j + m]
                    if _maxdist(template_i, template_j, m) <= r:
                        matches += 1
                
                if matches > 0:
                    phi += np.log(matches / (n - m + 1.0))
            
            return phi / (n - m + 1.0)
        
        return _phi(m) - _phi(m + 1)
    
    def compute_all_features(self, rr_intervals: np.ndarray, 
                           rr_times: np.ndarray) -> Dict[str, float]:
        """
        Compute all HRV features.
        
        Args:
            rr_intervals: RR intervals in seconds
            rr_times: Times corresponding to RR intervals
            
        Returns:
            Dictionary with all HRV features
        """
        features = {}
        
        # Time-domain features
        features.update(self.compute_time_features(rr_intervals))
        
        # Frequency-domain features
        features.update(self.compute_frequency_features(rr_intervals, rr_times))
        
        # Poincaré features
        features.update(self.compute_poincare_features(rr_intervals))
        
        # Entropy features
        if len(rr_intervals) >= 10:  # Minimum for entropy calculation
            features['SampEn'] = self.compute_sample_entropy(rr_intervals)
            features['ApEn'] = self.compute_approximate_entropy(rr_intervals)
        else:
            features['SampEn'] = 0.0
            features['ApEn'] = 0.0
        
        return features
    
    def _get_empty_time_features(self) -> Dict[str, float]:
        """Return empty time-domain features."""
        return {
            'NN50': 0.0, 'pNN50': 0.0, 'RMSSD': 0.0, 'SDNN': 0.0, 'SDSD': 0.0,
            'RRMean': 0.0, 'RRMin': 0.0, 'RRMax': 0.0, 'RRVar': 0.0
        }
    
    def _get_empty_frequency_features(self) -> Dict[str, float]:
        """Return empty frequency-domain features."""
        return {
            'TOTAL_POWER': 0.0, 'VLF_POWER': 0.0, 'LF_POWER': 0.0, 'HF_POWER': 0.0,
            'VLF_NORM': 0.0, 'LF_NORM': 0.0, 'HF_NORM': 0.0, 'LF_TO_HF': 0.0
        }
    
    @staticmethod
    def get_feature_names() -> List[str]:
        """Get list of all feature names in order."""
        return [
            # Time-domain
            'NN50', 'pNN50', 'RMSSD', 'SDNN', 'SDSD', 
            'RRMean', 'RRMin', 'RRMax', 'RRVar',
            # Frequency-domain
            'TOTAL_POWER', 'VLF_POWER', 'LF_POWER', 'HF_POWER',
            'VLF_NORM', 'LF_NORM', 'HF_NORM', 'LF_TO_HF',
            # Nonlinear
            'SD1', 'SD2', 'SD1toSD2', 'SampEn', 'ApEn'
        ]
    
    @staticmethod
    def get_feature_descriptions() -> Dict[str, str]:
        """Get descriptions of all features."""
        return {
            'NN50': 'Number of successive RR intervals differing by >50ms',
            'pNN50': 'Percentage of NN50',
            'RMSSD': 'Root mean square of successive differences (ms)',
            'SDNN': 'Standard deviation of RR intervals (ms)',
            'SDSD': 'Standard deviation of successive differences (ms)',
            'RRMean': 'Mean RR interval (ms)',
            'RRMin': 'Minimum RR interval (ms)',
            'RRMax': 'Maximum RR interval (ms)',
            'RRVar': 'Variance of RR intervals (ms²)',
            'TOTAL_POWER': 'Total power in all frequency bands (ms²)',
            'VLF_POWER': 'Very low frequency power (ms²)',
            'LF_POWER': 'Low frequency power (ms²)',
            'HF_POWER': 'High frequency power (ms²)',
            'VLF_NORM': 'Normalized VLF power (%)',
            'LF_NORM': 'Normalized LF power (%)',
            'HF_NORM': 'Normalized HF power (%)',
            'LF_TO_HF': 'LF/HF ratio',
            'SD1': 'Poincaré plot SD1 (short-term variability, ms)',
            'SD2': 'Poincaré plot SD2 (long-term variability, ms)',
            'SD1toSD2': 'SD1/SD2 ratio',
            'SampEn': 'Sample entropy',
            'ApEn': 'Approximate entropy'
        }


def test_hrv_features():
    """Test HRV feature extraction with synthetic data."""
    
    # Generate synthetic RR intervals (0.8s ± variation)
    np.random.seed(42)
    n_beats = 100
    base_rr = 0.8  # seconds (75 bpm)
    rr_intervals = base_rr + 0.1 * np.random.randn(n_beats)
    rr_intervals = np.clip(rr_intervals, 0.4, 1.5)  # Physiological range
    
    # Generate corresponding times
    rr_times = np.cumsum(rr_intervals)
    
    # Extract features
    extractor = HRVFeatureExtractor()
    features = extractor.compute_all_features(rr_intervals, rr_times)
    
    print("HRV Feature Extraction Test Results:")
    print("-" * 50)
    
    feature_names = extractor.get_feature_names()
    descriptions = extractor.get_feature_descriptions()
    
    for name in feature_names:
        if name in features:
            print(f"{name:15}: {features[name]:8.3f} - {descriptions[name]}")
    
    return features


if __name__ == "__main__":
    test_hrv_features()