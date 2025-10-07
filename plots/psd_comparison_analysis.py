#!/usr/bin/env python3
"""
PSD Method Comparison Analysis Across Multiple Runs

This script selects 50 non-seizure runs across all patients and computes the mean
of detailed quantitative comparison metrics between Lomb-Scargle and Welch methods.

Uses the exact methods and parameters validated in psd_method_comparison.ipynb
"""

import numpy as np
import pandas as pd
import sys
import os
from pathlib import Path
import warnings
from scipy import signal, interpolate
from scipy.signal import lombscargle
from scipy.interpolate import PchipInterpolator, Akima1DInterpolator
import mne
from tqdm import tqdm
import random

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Add parent directory to path to import our modules
parent_dir = Path(__file__).absolute().parent.parent
sys.path.append(str(parent_dir))

# Import our custom modules
from data_processing_pipeline import DataDiscovery, AnnotationProcessor
from ecg_processing import ECGProcessor
from hrv_features import HRVFeatureExtractor

# FIXED PARAMETERS - DO NOT CHANGE (validated from notebook)
TIME_SEGMENT = [0, 1500]  # First 25 minutes as specified
INTERPOLATION_METHOD = 'akima'  # Validated method
N_RUNS = 50  # Number of runs to analyze
DATA_ROOT = "/Volumes/Seizury/ds005873"  # Data path

# Analysis parameters (from notebook)
vlf_band = [0.003, 0.04]
lf_band = [0.04, 0.15]
hf_band = [0.15, 0.4]
f_min = 0.003
f_max = 0.5
n_freq_points = 1000
scale_factor = 1e6  # Convert to ms²

print("="*80)
print("PSD METHOD COMPARISON ANALYSIS - MULTI-RUN STATISTICAL ANALYSIS")
print("="*80)
print(f"Configuration:")
print(f"  Time segment: {TIME_SEGMENT[0]}-{TIME_SEGMENT[1]} seconds ({TIME_SEGMENT[1]-TIME_SEGMENT[0]} seconds)")
print(f"  Interpolation method: {INTERPOLATION_METHOD.upper()}")
print(f"  Number of runs to analyze: {N_RUNS}")
print(f"  Frequency bands: VLF={vlf_band}, LF={lf_band}, HF={hf_band}")


def remove_outliers_iqr(rr_intervals, rr_times):
    """
    Remove outliers from RR intervals using the Interquartile Range (IQR) method.
    Replaces outliers with appropriate quartile values instead of removing them.
    
    IMPORTED FROM VALIDATED NOTEBOOK - DO NOT MODIFY
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
    cleaned_rr[outlier_mask_low] = Q1
    cleaned_rr[outlier_mask_high] = Q3
    
    return cleaned_rr, rr_times, ~outlier_mask


def preprocess_rr_intervals(rr_intervals, rr_times):
    """
    Comprehensive preprocessing of RR intervals for spectral analysis.
    
    IMPORTED FROM VALIDATED NOTEBOOK - DO NOT MODIFY
    """
    # Step 1: Handle outliers by replacing with quartiles
    clean_rr, clean_times, non_outlier_mask = remove_outliers_iqr(rr_intervals, rr_times)
    
    if len(clean_rr) < 4:
        return np.array([]), np.array([]), {}
    
    # Step 2: Remove mean (centering)
    original_mean = np.mean(clean_rr)
    centered_rr = clean_rr - original_mean
    
    # Step 3: High-pass filter detrending
    from scipy import signal as scipy_signal
    
    time_span = clean_times[-1] - clean_times[0]
    n_samples = len(clean_times)
    original_fs = n_samples / time_span
    target_fs = max(4.0, original_fs * 2)
    
    uniform_times_filter = np.linspace(clean_times[0], clean_times[-1], 
                                     int(time_span * target_fs))
    interp_for_filter = np.interp(uniform_times_filter, clean_times, centered_rr)
    
    cutoff_freq = 0.003  # Hz - FIXED as validated
    nyquist_freq = target_fs / 2
    normalized_cutoff = cutoff_freq / nyquist_freq
    
    if normalized_cutoff >= 1.0:
        detrended_rr = scipy_signal.detrend(centered_rr, type='linear')
        trend_method = "linear (fallback)"
    else:
        try:
            #print("normalized_cutoff < 1.0")
            sos = scipy_signal.butter(4, normalized_cutoff, btype='high', output='sos')
            filtered_uniform = scipy_signal.sosfiltfilt(sos, interp_for_filter)
            detrended_rr = np.interp(clean_times, uniform_times_filter, filtered_uniform)
            trend_method = f"high-pass filter (cutoff: {cutoff_freq} Hz)"
        except Exception:
            print("fallback")
            detrended_rr = scipy_signal.detrend(centered_rr, type='linear')
            trend_method = "linear (fallback)"
    
    trend_removed = np.std(centered_rr) - np.std(detrended_rr)
    
    preprocessing_info = {
        'n_outliers_replaced': np.sum(~non_outlier_mask),
        'outlier_percentage': np.sum(~non_outlier_mask) / len(rr_intervals) * 100,
        'original_mean': original_mean,
        'detrend_method': trend_method,
        'cutoff_frequency': cutoff_freq,
        'final_mean': np.mean(detrended_rr),
        'final_std': np.std(detrended_rr),
        'trend_removed_variance': trend_removed
    }
    
    return detrended_rr, clean_times, preprocessing_info


def compute_lombscargle_psd(rr_intervals, rr_times, f_min=0.003, f_max=0.5, n_points=1000):
    """
    Compute Lomb-Scargle periodogram with proper normalization.
    
    IMPORTED FROM VALIDATED NOTEBOOK - DO NOT MODIFY
    """
    processed_rr, processed_times, preproc_info = preprocess_rr_intervals(rr_intervals, rr_times)
    
    if len(processed_rr) < 4:
        return np.array([]), np.array([])
    
    freqs = np.linspace(f_min, f_max, n_points)
    omega = 2 * np.pi * freqs
    N = len(processed_times)
    
    psd_ls = lombscargle(processed_times, processed_rr, omega, normalize=False)
    df = freqs[1] - freqs[0]
    psd_ls = psd_ls * 2.0 / (N * df)
    
    signal_variance = np.var(processed_rr)
    total_periodogram_power = np.sum(psd_ls) * df
    
    if total_periodogram_power > 0:
        normalization_factor = signal_variance / total_periodogram_power
        psd_ls = psd_ls * normalization_factor
    
    return freqs, psd_ls


def interpolate_rr_intervals_configurable(rr_intervals, rr_times, method='akima', sampling_rate=4.0):
    """
    Interpolate RR intervals using specified method with preprocessing.
    
    IMPORTED FROM VALIDATED NOTEBOOK - DO NOT MODIFY
    """
    processed_rr, processed_times, preproc_info = preprocess_rr_intervals(rr_intervals, rr_times)
    
    if len(processed_rr) == 0 or len(processed_times) == 0:
        return np.array([]), np.array([])
        
    if len(processed_rr) < 2 or len(processed_times) < 2:
        return np.array([]), np.array([])
    
    start_time = processed_times[0]
    end_time = processed_times[-1]
    time_span = end_time - start_time
    
    if time_span <= 0:
        return np.array([]), np.array([])
        
    uniform_times = np.arange(start_time, end_time, 1/sampling_rate)
    
    if len(uniform_times) < 2:
        return np.array([]), np.array([])
    
    try:
        if method.lower() == 'pchip':
            interp_func = PchipInterpolator(processed_times, processed_rr, extrapolate=True)
            interpolated_rr = interp_func(uniform_times)
        elif method.lower() == 'akima':
            interp_func = Akima1DInterpolator(processed_times, processed_rr)
            interpolated_rr = interp_func(uniform_times, extrapolate=True)
        elif method.lower() == 'cubic':
            interp_func = interpolate.interp1d(processed_times, processed_rr, 
                                             kind='cubic', 
                                             bounds_error=False,
                                             fill_value='extrapolate')
            interpolated_rr = interp_func(uniform_times)
        else:
            raise ValueError(f"Unknown interpolation method: {method}")
        
        valid_mask = np.isfinite(interpolated_rr)
        if not np.any(valid_mask):
            return np.array([]), np.array([])
            
        return interpolated_rr[valid_mask], uniform_times[valid_mask]
        
    except Exception:
        return np.array([]), np.array([])


def compute_welch_psd_interpolated(rr_intervals, rr_times, method='akima', sampling_rate=4.0):
    """
    Compute Welch PSD after preprocessing and interpolation.
    
    IMPORTED FROM VALIDATED NOTEBOOK - DO NOT MODIFY
    """
    interpolated_rr, interpolated_times = interpolate_rr_intervals_configurable(
        rr_intervals, rr_times, method, sampling_rate)
    
    if len(interpolated_rr) < 10:
        return np.array([]), np.array([]), np.array([]), np.array([])
    
    detrend_method = None
    
    freqs_welch, psd_welch = signal.welch(
        interpolated_rr,
        fs=sampling_rate,
        nperseg=2048,
        noverlap=1536,
        window='hann',
        detrend=detrend_method,
        scaling='density',
        return_onesided=True
    )
    
    return freqs_welch, psd_welch, interpolated_rr, interpolated_times


def compute_band_power(psd, freqs, band):
    """Compute power in a specific frequency band using trapezoidal integration."""
    mask = (freqs >= band[0]) & (freqs <= band[1])
    if not np.any(mask):
        return 0.0
    
    band_psd = psd[mask]
    band_freqs = freqs[mask]
    
    if len(band_freqs) < 2:
        return 0.0
    
    power = np.trapz(band_psd, band_freqs)
    return max(0.0, power)


def find_spectral_peaks(psd, freqs, height_threshold=0.1):
    """Find peaks in power spectral density."""
    from scipy.signal import find_peaks
    
    psd_norm = psd / np.max(psd)
    peaks, properties = find_peaks(psd_norm, height=height_threshold, distance=10)
    
    peak_freqs = freqs[peaks]
    peak_powers = psd[peaks]
    
    return peak_freqs, peak_powers, peaks


def analyze_single_run(run_info, ecg_processor, hrv_extractor, annotation_processor):
    """
    Analyze a single run and return comparison metrics.
    Returns None if analysis fails.
    """
    try:
        # Load seizure annotations to verify no seizures in segment
        seizure_events = pd.DataFrame()
        if run_info['annotation_file']:
            seizure_events = annotation_processor.load_annotations(run_info['annotation_file'])
            
            # Check for seizures in our time segment
            start_time, end_time = TIME_SEGMENT
            seizures_in_segment = []
            if not seizure_events.empty:
                for _, seizure in seizure_events.iterrows():
                    if start_time <= seizure['onset'] <= end_time:
                        seizures_in_segment.append(seizure['onset'])
                        
            if seizures_in_segment:
                return None  # Skip runs with seizures in our segment
        
        # Load ECG data
        raw_ecg = mne.io.read_raw_edf(run_info['ecg_file'], preload=True, verbose=False)
        if raw_ecg.info['sfreq'] != ecg_processor.sampling_rate:
            raw_ecg.resample(ecg_processor.sampling_rate, verbose=False)
        ecg_data = raw_ecg.get_data()[0]
        del raw_ecg
        
        # Extract tachogram
        tachogram_result = ecg_processor.process_ecg_to_tachogram(ecg_data)
        
        # Extract time segment
        rr_intervals = tachogram_result['filtered_rr']
        rr_times = tachogram_result['filtered_times']
        
        start_time, end_time = TIME_SEGMENT
        segment_mask = (rr_times >= start_time) & (rr_times <= end_time)
        segment_rr = rr_intervals[segment_mask]
        segment_times = rr_times[segment_mask]
        
        if len(segment_rr) < 20:
            return None  # Skip segments with too few RR intervals
        
        # Compute Lomb-Scargle PSD
        freqs_ls, psd_ls = compute_lombscargle_psd(segment_rr, segment_times, f_min, f_max, n_freq_points)
        if len(psd_ls) == 0:
            return None
        
        # Compute Welch PSD
        freqs_welch, psd_welch, interpolated_rr, interpolated_times = compute_welch_psd_interpolated(
            segment_rr, segment_times, INTERPOLATION_METHOD, hrv_extractor.resampling_rate
        )
        if len(psd_welch) == 0:
            return None
        
        # Compute power in frequency bands - Lomb-Scargle
        vlf_power_ls = compute_band_power(psd_ls, freqs_ls, vlf_band) * scale_factor
        lf_power_ls = compute_band_power(psd_ls, freqs_ls, lf_band) * scale_factor
        hf_power_ls = compute_band_power(psd_ls, freqs_ls, hf_band) * scale_factor
        total_power_ls = vlf_power_ls + lf_power_ls + hf_power_ls
        
        # Compute power in frequency bands - Welch
        vlf_power_welch = compute_band_power(psd_welch, freqs_welch, vlf_band) * scale_factor
        lf_power_welch = compute_band_power(psd_welch, freqs_welch, lf_band) * scale_factor
        hf_power_welch = compute_band_power(psd_welch, freqs_welch, hf_band) * scale_factor
        total_power_welch = vlf_power_welch + lf_power_welch + hf_power_welch
        
        # Calculate metrics
        lf_hf_ls = lf_power_ls / hf_power_ls if hf_power_ls > 0 else 0
        lf_hf_welch = lf_power_welch / hf_power_welch if hf_power_welch > 0 else 0
        hf_percent_ls = hf_power_ls / total_power_ls * 100 if total_power_ls > 0 else 0
        hf_percent_welch = hf_power_welch / total_power_welch * 100 if total_power_welch > 0 else 0
        
        # Find spectral peaks
        peak_freqs_ls, _, _ = find_spectral_peaks(psd_ls, freqs_ls)
        peak_freqs_welch, _, _ = find_spectral_peaks(psd_welch, freqs_welch)
        
        # Calculate differences
        total_diff = abs(total_power_welch - total_power_ls) / total_power_ls * 100 if total_power_ls > 0 else 0
        lf_hf_diff = abs(lf_hf_welch - lf_hf_ls) / lf_hf_ls * 100 if lf_hf_ls > 0 else 0
        hf_diff = abs(hf_percent_welch - hf_percent_ls) / hf_percent_ls * 100 if hf_percent_ls > 0 else 0
        
        # Calculate ln(LF/HF) difference
        ln_lf_hf_ls = np.log(lf_hf_ls) if lf_hf_ls > 0 else None
        ln_lf_hf_welch = np.log(lf_hf_welch) if lf_hf_welch > 0 else None
        ln_lf_hf_diff = abs(ln_lf_hf_welch - ln_lf_hf_ls) if (ln_lf_hf_ls is not None and ln_lf_hf_welch is not None) else None
        
        peak_count_diff = len(peak_freqs_welch) - len(peak_freqs_ls)
        
        return {
            'run_id': f"{run_info['subject']}/{run_info['session']}/run-{run_info['run']}",
            'total_diff': total_diff,
            'lf_hf_diff': lf_hf_diff,
            'ln_lf_hf_diff': ln_lf_hf_diff,
            'hf_diff': hf_diff,
            'peak_count_diff': peak_count_diff,
            'n_rr_intervals': len(segment_rr),
            'total_power_ls': total_power_ls,
            'total_power_welch': total_power_welch,
            'lf_hf_ls': lf_hf_ls,
            'lf_hf_welch': lf_hf_welch
        }
        
    except Exception as e:
        print(f"Error analyzing run {run_info['subject']}/{run_info['session']}/run-{run_info['run']}: {e}")
        return None


def main():
    """Main analysis function"""
    
    print("\nInitializing processing modules...")
    
    # Initialize processors
    discovery = DataDiscovery(DATA_ROOT)
    discovery.scan_dataset()
    matched_runs = discovery.match_runs()
    
    ecg_processor = ECGProcessor(sampling_rate=256)
    hrv_extractor = HRVFeatureExtractor(resampling_rate=4.0)
    annotation_processor = AnnotationProcessor()
    
    print(f"Found {len(matched_runs)} total runs in dataset")
    
    # Randomly select runs to analyze
    random.seed(42)  # For reproducibility
    random.shuffle(matched_runs)
    
    print(f"\nAnalyzing runs to find {N_RUNS} non-seizure segments...")
    
    results = []
    analyzed_count = 0
    
    progress_bar = tqdm(matched_runs, desc="Processing runs")
    
    for run in progress_bar:
        if len(results) >= N_RUNS:
            break
            
        progress_bar.set_description(f"Processing {run['subject']}/{run['session']}/run-{run['run']}")
        
        result = analyze_single_run(run, ecg_processor, hrv_extractor, annotation_processor)
        
        if result is not None:
            results.append(result)
            progress_bar.set_description(f"Found {len(results)}/{N_RUNS} valid runs")
        
        analyzed_count += 1
    
    progress_bar.close()
    
    if len(results) < N_RUNS:
        print(f"\nWarning: Only found {len(results)} valid runs out of {N_RUNS} requested")
        print(f"Analyzed {analyzed_count} total runs")
    
    if len(results) == 0:
        print("No valid runs found. Exiting.")
        return
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(results)
    
    print(f"\n" + "="*80)
    print("MULTI-RUN STATISTICAL ANALYSIS RESULTS")
    print("="*80)
    
    print(f"\nDataset Summary:")
    print(f"  Valid runs analyzed: {len(results)}")
    print(f"  Total runs processed: {analyzed_count}")
    print(f"  Success rate: {len(results)/analyzed_count*100:.1f}%")
    
    # Remove runs with None values for ln_lf_hf_diff
    valid_ln_mask = df['ln_lf_hf_diff'].notna()
    n_valid_ln = valid_ln_mask.sum()
    
    print(f"  Runs with valid ln(LF/HF): {n_valid_ln}/{len(results)} ({n_valid_ln/len(results)*100:.1f}%)")
    
    print(f"\nRun Characteristics:")
    print(f"  Mean RR intervals per run: {df['n_rr_intervals'].mean():.0f} ± {df['n_rr_intervals'].std():.0f}")
    print(f"  RR interval range: {df['n_rr_intervals'].min():.0f} - {df['n_rr_intervals'].max():.0f}")
    
    print(f"\nMean Method Differences Across {len(results)} Runs:")
    print(f"  Total Power difference: {df['total_diff'].mean():.2f} ± {df['total_diff'].std():.2f}%")
    print(f"  LF/HF ratio difference: {df['lf_hf_diff'].mean():.2f} ± {df['lf_hf_diff'].std():.2f}%")
    
    if n_valid_ln > 0:
        ln_mean = df.loc[valid_ln_mask, 'ln_lf_hf_diff'].mean()
        ln_std = df.loc[valid_ln_mask, 'ln_lf_hf_diff'].std()
        print(f"  ln(LF/HF) absolute difference: {ln_mean:.3f} ± {ln_std:.3f} (n={n_valid_ln})")
    else:
        print(f"  ln(LF/HF) absolute difference: undefined (no valid values)")
    
    print(f"  HF percentage difference: {df['hf_diff'].mean():.2f} ± {df['hf_diff'].std():.2f}%")
    print(f"  Peak count difference: {df['peak_count_diff'].mean():.2f} ± {df['peak_count_diff'].std():.2f}")
    
    print(f"\nDistribution Analysis:")
    print(f"  Total Power difference - Median: {df['total_diff'].median():.1f}%, IQR: [{df['total_diff'].quantile(0.25):.1f}%, {df['total_diff'].quantile(0.75):.1f}%]")
    print(f"  LF/HF ratio difference - Median: {df['lf_hf_diff'].median():.1f}%, IQR: [{df['lf_hf_diff'].quantile(0.25):.1f}%, {df['lf_hf_diff'].quantile(0.75):.1f}%]")
    print(f"  HF percentage difference - Median: {df['hf_diff'].median():.1f}%, IQR: [{df['hf_diff'].quantile(0.25):.1f}%, {df['hf_diff'].quantile(0.75):.1f}%]")
    
    print(f"\nMethod Agreement Assessment:")
    
    # Total power agreement
    total_good = (df['total_diff'] < 10).sum()
    total_moderate = ((df['total_diff'] >= 10) & (df['total_diff'] < 25)).sum()
    total_poor = (df['total_diff'] >= 25).sum()
    
    print(f"  Total Power Agreement:")
    print(f"    GOOD (<10% diff): {total_good}/{len(results)} ({total_good/len(results)*100:.1f}%)")
    print(f"    MODERATE (10-25% diff): {total_moderate}/{len(results)} ({total_moderate/len(results)*100:.1f}%)")
    print(f"    POOR (>25% diff): {total_poor}/{len(results)} ({total_poor/len(results)*100:.1f}%)")
    
    # LF/HF agreement
    lf_hf_good = (df['lf_hf_diff'] < 15).sum()
    lf_hf_moderate = ((df['lf_hf_diff'] >= 15) & (df['lf_hf_diff'] < 30)).sum()
    lf_hf_poor = (df['lf_hf_diff'] >= 30).sum()
    
    print(f"  LF/HF Ratio Agreement:")
    print(f"    GOOD (<15% diff): {lf_hf_good}/{len(results)} ({lf_hf_good/len(results)*100:.1f}%)")
    print(f"    MODERATE (15-30% diff): {lf_hf_moderate}/{len(results)} ({lf_hf_moderate/len(results)*100:.1f}%)")
    print(f"    POOR (>30% diff): {lf_hf_poor}/{len(results)} ({lf_hf_poor/len(results)*100:.1f}%)")
    
    # ln(LF/HF) agreement
    if n_valid_ln > 0:
        ln_df = df.loc[valid_ln_mask, 'ln_lf_hf_diff']
        ln_good = (ln_df < 0.2).sum()
        ln_moderate = ((ln_df >= 0.2) & (ln_df < 0.5)).sum()
        ln_poor = (ln_df >= 0.5).sum()
        
        print(f"  ln(LF/HF) Agreement:")
        print(f"    GOOD (<0.2 abs diff): {ln_good}/{n_valid_ln} ({ln_good/n_valid_ln*100:.1f}%)")
        print(f"    MODERATE (0.2-0.5 abs diff): {ln_moderate}/{n_valid_ln} ({ln_moderate/n_valid_ln*100:.1f}%)")
        print(f"    POOR (>0.5 abs diff): {ln_poor}/{n_valid_ln} ({ln_poor/n_valid_ln*100:.1f}%)")
    
    print(f"\nOverall Conclusions:")
    overall_good = total_good + lf_hf_good
    overall_total = len(results) * 2
    print(f"  Overall GOOD agreement rate: {overall_good}/{overall_total} ({overall_good/overall_total*100:.1f}%)")
    
    print(f"  Method consistency: {'HIGH' if df['total_diff'].std() < 10 else 'MODERATE' if df['total_diff'].std() < 20 else 'LOW'}")
    print(f"  Recommended for seizure prediction: {'YES' if df['total_diff'].mean() < 15 and df['lf_hf_diff'].mean() < 20 else 'WITH CAUTION'}")
    
    # Save detailed results
    output_file = Path(__file__).parent / f"psd_comparison_results_{len(results)}runs.csv"
    df.to_csv(output_file, index=False)
    print(f"\nDetailed results saved to: {output_file}")
    
    print(f"\n" + "="*80)
    print("ANALYSIS COMPLETED")
    print("="*80)


if __name__ == "__main__":
    main()