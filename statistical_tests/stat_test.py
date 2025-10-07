#!/usr/bin/env python3

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Add parent directory to path to import our modules
parent_dir = Path(__file__).absolute().parent.parent
sys.path.append(str(parent_dir))

# Import our custom modules
from data_processing_pipeline import DataDiscovery, AnnotationProcessor
from ecg_processing import ECGProcessor
from hrv_features import HRVFeatureExtractor

# Import the statistical testing modules
from seizure_hrv_statistical_test import SeizureHRVAnalyzer
from hrv_feature_extraction import HRVFeatureExtractor_Enhanced, StatisticalTester

class CompleteSeizureHRVAnalysis:
    """
    Complete seizure HRV analysis with feature extraction and statistical testing.
    """
    
    def __init__(self, data_root: str, pre_seizure_window_min: int = 25, 
                 control_window_offset_min: int = 60, max_runs_to_process: int = 2):
        """
        Initialize the complete analysis.
        
        Args:
            data_root: Path to dataset root directory
            pre_seizure_window_min: Pre-seizure window duration in minutes
            control_window_offset_min: Offset for control window from recording start (minutes)
            max_runs_to_process: Maximum number of runs to process (None = process all)
        """
        self.data_root = data_root
        self.pre_seizure_window_min = pre_seizure_window_min
        self.control_window_offset_min = control_window_offset_min
        self.max_runs_to_process = max_runs_to_process
        
        # Initialize analyzer components
        self.analyzer = SeizureHRVAnalyzer(data_root, pre_seizure_window_min)
        self.feature_extractor = HRVFeatureExtractor_Enhanced()
        self.statistical_tester = StatisticalTester()
        
        # Results storage
        self.results = {}
        self.all_features = {'pre_seizure': [], 'control': []}
        
    def run_complete_analysis(self) -> Dict:
        """
        Run the complete seizure HRV statistical analysis.
        
        Returns:
            Dictionary containing all analysis results
        """
        print("="*80)
        print("COMPLETE SEIZURE HRV STATISTICAL ANALYSIS")
        print("="*80)
        print(f"Analysis parameters:")
        print(f"  • Pre-seizure window: {self.pre_seizure_window_min} minutes")
        print(f"  • Control window offset: {self.control_window_offset_min} minutes")
        print(f"  • Max runs to process: {self.max_runs_to_process if self.max_runs_to_process else 'All'}")
        print(f"  • Feature extraction: 3, 5, 25-minute windows")
        print(f"  • Statistical test: PERMANOVA")
        
        # Step 1: Find valid seizure runs
        print(f"\\nStep 1: Finding valid seizure runs...")
        valid_runs = self.analyzer.find_valid_seizure_runs()
        
        if len(valid_runs) == 0:
            print("No valid seizure runs found!")
            return {}
        
        # Store total valid runs before limiting
        total_valid_runs = len(valid_runs)
        self.results['n_valid_runs_total'] = total_valid_runs

        # Apply run limiting if requested
        if self.max_runs_to_process is not None and self.max_runs_to_process > 0:
            if self.max_runs_to_process < total_valid_runs:
                print(f"Limiting processing to first {self.max_runs_to_process} of {total_valid_runs} valid runs")
                valid_runs = valid_runs[:self.max_runs_to_process]
            else:
                print(f"max_runs_to_process ({self.max_runs_to_process}) >= total valid runs ({total_valid_runs}); processing all runs")
        else:
            print("No run limit applied (processing all valid runs)")

        # Store the runs actually used
        self.results['n_valid_runs_used'] = len(valid_runs)
        self.results['valid_runs'] = valid_runs
        
        # Step 2: Process each run for feature extraction
        print(f"\\nStep 2: Processing {len(valid_runs)} runs for feature extraction...")
        
        processed_runs = 0
        failed_runs = 0
        
        for i, run_info in enumerate(valid_runs):
            print(f"\\n  Processing run {i+1}/{len(valid_runs)}: "
                  f"{run_info['subject']}/ses-{run_info['session']}/run-{run_info['run']}")
            
            try:
                success = self._process_single_run(run_info)
                if success:
                    processed_runs += 1
                    print(f"    ✓ Successfully processed")
                else:
                    failed_runs += 1
                    print(f"    ✗ Processing failed")
            except Exception as e:
                failed_runs += 1
                print(f"    ✗ Error: {e}")
        
        print(f"\\nRun processing summary:")
        print(f"  • Successfully processed: {processed_runs}")
        print(f"  • Failed: {failed_runs}")
        
        self.results['processed_runs'] = processed_runs
        self.results['failed_runs'] = failed_runs
        
        if processed_runs == 0:
            print("No runs processed successfully!")
            return self.results
        
        # Step 3: Statistical analysis
        print(f"\\nStep 3: Running statistical analysis...")
        self._run_statistical_analysis()
        
        # Step 4: Generate report
        print(f"\\nStep 4: Generating analysis report...")
        self._generate_analysis_report()
        
        return self.results
    
    def _process_single_run(self, run_info: Dict) -> bool:
        """
        Process a single run for feature extraction.
        
        Args:
            run_info: Dictionary containing run information
            
        Returns:
            True if processing successful, False otherwise
        """
        try:
            # Extract RR intervals up to seizure onset
            seizure_onset = run_info['seizure_onset']
            rr_intervals, rr_times = self.analyzer.extract_rr_intervals(run_info, seizure_onset)
            
            if len(rr_intervals) < 100:  # Minimum requirement
                print(f"      Insufficient RR intervals: {len(rr_intervals)}")
                return False
            
            print(f"      Extracted {len(rr_intervals)} RR intervals")
            
            # Apply preprocessing
            processed_rr, processed_times, preproc_info = self.analyzer.preprocess_rr_intervals(
                rr_intervals, rr_times)
            
            if len(processed_rr) < 50:
                print(f"      Insufficient data after preprocessing: {len(processed_rr)}")
                return False
            
            print(f"      Preprocessing: {len(rr_intervals)} → {len(processed_rr)} intervals")
            
            # Define analysis windows
            pre_seizure_end = seizure_onset
            pre_seizure_start = seizure_onset - (self.pre_seizure_window_min * 60)
            
            # Control window: from 25 minutes after recording start to pre-seizure window start
            control_start = max(processed_times[0], 25 * 60)  # Start from 25 minutes or data start, whichever is later
            control_end = pre_seizure_start  # End where pre-seizure window starts
            
            # Ensure control window is large enough for meaningful analysis
            control_duration = control_end - control_start
            min_control_duration = self.pre_seizure_window_min * 60  # At least as long as pre-seizure window
            
            if control_duration < min_control_duration:
                print(f"      Insufficient control window duration: {control_duration/60:.1f} min "
                      f"(need at least {min_control_duration/60:.1f} min)")
                return False
            
            # Ensure we have enough data for both windows
            data_start = processed_times[0]
            data_end = processed_times[-1]
            data_duration = data_end - data_start
            
            print(f"      Temporal coverage analysis:")
            print(f"        Data spans: {data_start:.1f} - {data_end:.1f} sec ({data_duration/60:.1f} min)")
            print(f"        Control window needs: {control_start:.1f} - {control_end:.1f} sec ({control_duration/60:.1f} min)")
            print(f"        Pre-seizure window needs: {pre_seizure_start:.1f} - {pre_seizure_end:.1f} sec")
            print(f"        Seizure onset: {seizure_onset:.1f} sec ({seizure_onset/60:.1f} min)")
            
            # Since control_start is now set to processed_times[0], this check is always true
            # But keep a sanity check for data availability
            
            if processed_times[-1] < pre_seizure_end:
                print(f"      ✗ Data ends too early: {data_end:.1f} < {pre_seizure_end:.1f}")
                return False
            
            print(f"      ✓ Temporal coverage sufficient")
            
            print(f"      Window definition:")
            print(f"        Control: {control_start/60:.1f} - {control_end/60:.1f} min "
                  f"(duration: {control_duration/60:.1f} min)")
            print(f"        Pre-seizure: {pre_seizure_start/60:.1f} - {pre_seizure_end/60:.1f} min "
                  f"(duration: {self.pre_seizure_window_min} min)")
            print(f"        Gap between windows: NO OVERLAP (control ends where pre-seizure starts)")
            
            # Extract features for pre-seizure window (every minute going backwards)
            pre_seizure_features = []
            for minute_offset in range(self.pre_seizure_window_min):
                center_time = pre_seizure_end - (minute_offset * 60)
                
                # Skip if not enough data for longest window (25 minutes)
                if center_time - (25 * 60) < processed_times[0]:
                    continue
                
                features = self.feature_extractor.extract_windowed_features(
                    processed_rr, processed_times, center_time, self.analyzer)
                
                if any(np.isfinite(list(features.values()))):  # At least some valid features
                    features['run_id'] = f"{run_info['subject']}_ses{run_info['session']}_run{run_info['run']}"
                    features['window_type'] = 'pre_seizure'
                    features['minute_offset'] = minute_offset
                    features['center_time'] = center_time
                    pre_seizure_features.append(features)
            
            # Extract features for control window (sample evenly across available control period)
            control_features = []
            
            # Calculate how many control windows we can extract (every minute with 25-min lookback)
            control_available_start = control_start + (25 * 60)  # First valid center time (25 min from start)
            control_available_end = control_end  # Last valid center time
            
            if control_available_start < control_available_end:
                # Sample control windows evenly across the available control period
                # Use same number as pre-seizure windows for balanced comparison
                n_control_windows = min(self.pre_seizure_window_min, 
                                       int((control_available_end - control_available_start) / 60))
                
                if n_control_windows > 0:
                    control_interval = (control_available_end - control_available_start) / n_control_windows
                    
                    for i in range(n_control_windows):
                        center_time = control_available_start + (i * control_interval)
                        
                        # Skip if not enough data for longest window (25 minutes)
                        if center_time - (25 * 60) < processed_times[0]:
                            continue
                        if center_time > processed_times[-1]:
                            break
                
                        features = self.feature_extractor.extract_windowed_features(
                            processed_rr, processed_times, center_time, self.analyzer)
                        
                        if any(np.isfinite(list(features.values()))):  # At least some valid features
                            features['run_id'] = f"{run_info['subject']}_ses{run_info['session']}_run{run_info['run']}"
                            features['window_type'] = 'control'
                            features['window_index'] = i  # Use window index instead of minute offset
                            features['center_time'] = center_time
                            control_features.append(features)
            
            print(f"      Extracted features: {len(pre_seizure_features)} pre-seizure, {len(control_features)} control windows")
            
            # Store features
            self.all_features['pre_seizure'].extend(pre_seizure_features)
            self.all_features['control'].extend(control_features)
            
            return len(pre_seizure_features) > 0 and len(control_features) > 0
            
        except Exception as e:
            print(f"      Processing error: {e}")
            return False
    
    def _run_statistical_analysis(self):
        """Run statistical analysis on extracted features."""
        print(f"  Feature summary:")
        print(f"    • Pre-seizure windows: {len(self.all_features['pre_seizure'])}")
        print(f"    • Control windows: {len(self.all_features['control'])}")
        
        if len(self.all_features['pre_seizure']) == 0 or len(self.all_features['control']) == 0:
            print("  Insufficient data for statistical analysis!")
            return
        
        # Prepare feature matrix
        feature_matrix, group_labels, feature_names = self.statistical_tester.prepare_feature_matrix(
            self.all_features['pre_seizure'], self.all_features['control'])
        
        if len(feature_matrix) == 0:
            print("  Failed to prepare feature matrix!")
            return
        
        print(f"  Prepared feature matrix: {feature_matrix.shape[0]} samples × {feature_matrix.shape[1]} features")
        
        # Run PERMANOVA test
        print(f"  Running PERMANOVA test...")
        permanova_results = self.statistical_tester.run_permanova(feature_matrix, group_labels)
        
        # Individual feature analysis
        print(f"  Analyzing individual features...")
        feature_analysis = self.statistical_tester.analyze_feature_importance(
            feature_matrix, group_labels, feature_names)
        
        # Store results
        self.results['permanova'] = permanova_results
        self.results['feature_analysis'] = feature_analysis
        self.results['feature_names'] = feature_names
        self.results['n_pre_seizure_windows'] = len(self.all_features['pre_seizure'])
        self.results['n_control_windows'] = len(self.all_features['control'])
        
        # Print main results
        print(f"\\n  PERMANOVA Results:")
        if 'error' not in permanova_results:
            print(f"    • Test statistic: {permanova_results['test_statistic']:.6f}")
            print(f"    • P-value: {permanova_results['p_value']:.6f}")
            print(f"    • Significant: {'YES' if permanova_results['p_value'] < 0.05 else 'NO'}")
        else:
            print(f"    • Error: {permanova_results['error']}")
    
    def _generate_analysis_report(self):
        """Generate comprehensive analysis report."""
        output_dir = Path(__file__).parent / "results"
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results to JSON
        results_file = output_dir / f"seizure_hrv_analysis_{timestamp}.json"
        
        # Convert numpy types for JSON serialization
        json_results = {}
        for key, value in self.results.items():
            if key == 'feature_analysis':
                json_results[key] = value.to_dict() if hasattr(value, 'to_dict') else str(value)
            elif isinstance(value, np.ndarray):
                json_results[key] = value.tolist()
            elif isinstance(value, (np.integer, np.floating)):
                json_results[key] = float(value)
            else:
                json_results[key] = value
        
        with open(results_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"  Results saved to: {results_file}")
        
        # Save feature analysis CSV
        if 'feature_analysis' in self.results and hasattr(self.results['feature_analysis'], 'to_csv'):
            csv_file = output_dir / f"feature_analysis_{timestamp}.csv"
            self.results['feature_analysis'].to_csv(csv_file, index=False)
            print(f"  Feature analysis saved to: {csv_file}")
        
        # Generate summary report
        self._generate_summary_report(output_dir / f"analysis_summary_{timestamp}.txt")
    
    def _generate_summary_report(self, output_file: Path):
        """Generate a human-readable summary report."""
        with open(output_file, 'w') as f:
            f.write("SEIZURE HRV STATISTICAL ANALYSIS - SUMMARY REPORT\\n")
            f.write("="*60 + "\\n\\n")
            
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n")
            f.write(f"Dataset: {self.data_root}\\n")
            f.write(f"Pre-seizure window: {self.pre_seizure_window_min} minutes\\n")
            f.write(f"Control window offset: {self.control_window_offset_min} minutes\\n\\n")
            
            # Data summary
            f.write("DATA SUMMARY\\n")
            f.write("-" * 20 + "\\n")
            f.write(f"Valid seizure runs found: {self.results.get('n_valid_runs', 0)}\\n")
            f.write(f"Successfully processed: {self.results.get('processed_runs', 0)}\\n")
            f.write(f"Failed processing: {self.results.get('failed_runs', 0)}\\n")
            f.write(f"Pre-seizure windows: {self.results.get('n_pre_seizure_windows', 0)}\\n")
            f.write(f"Control windows: {self.results.get('n_control_windows', 0)}\\n\\n")
            
            # Statistical results
            if 'permanova' in self.results:
                f.write("STATISTICAL RESULTS\\n")
                f.write("-" * 20 + "\\n")
                permanova = self.results['permanova']
                
                if 'error' not in permanova:
                    f.write(f"PERMANOVA Test Statistic: {permanova['test_statistic']:.6f}\\n")
                    f.write(f"P-value: {permanova['p_value']:.6f}\\n")
                    f.write(f"Number of permutations: {permanova['n_permutations']}\\n")
                    f.write(f"Statistical significance: {'YES' if permanova['p_value'] < 0.05 else 'NO'}\\n")
                    
                    if permanova['p_value'] < 0.05:
                        f.write("\\nCONCLUSION: HRV features show statistically significant differences\\n")
                        f.write("between pre-seizure and control periods.\\n")
                    else:
                        f.write("\\nCONCLUSION: No statistically significant differences found\\n")
                        f.write("between pre-seizure and control periods.\\n")
                else:
                    f.write(f"Statistical analysis failed: {permanova['error']}\\n")
            
            # Top features
            if 'feature_analysis' in self.results and hasattr(self.results['feature_analysis'], 'nlargest'):
                f.write("\\nTOP DISCRIMINATIVE FEATURES (by effect size)\\n")
                f.write("-" * 45 + "\\n")
                
                top_features = self.results['feature_analysis'].nlargest(10, 'Effect_Size', keep='all')
                for _, row in top_features.iterrows():
                    f.write(f"{row['Feature']}: Effect size = {row['Effect_Size']:.3f}, p = {row['MWU_P_Value']:.6f}\\n")
        
        print(f"  Summary report saved to: {output_file}")


def main():
    """Main function to run the complete seizure HRV analysis."""
    
    # Configuration
    DATA_ROOT = "/Volumes/Seizury/ds005873/"
    PRE_SEIZURE_WINDOW_MIN = 25  # Can be easily changed as requested
    CONTROL_WINDOW_OFFSET_MIN = 25  # Not used - control window spans from recording start to pre-seizure start
    
    # Check if data root exists
    if not os.path.exists(DATA_ROOT):
        print(f"Error: Data root directory not found: {DATA_ROOT}")
        print("Please update the DATA_ROOT variable in this script.")
        return
    
    try:
        # Initialize and run analysis
        analysis = CompleteSeizureHRVAnalysis(
            DATA_ROOT, 
            PRE_SEIZURE_WINDOW_MIN,
            CONTROL_WINDOW_OFFSET_MIN,
            max_runs_to_process = 2
        )
        
        results = analysis.run_complete_analysis()
        
        if results:
            print(f"\\n" + "="*80)
            print("ANALYSIS COMPLETED SUCCESSFULLY")
            print("="*80)
            
            if 'permanova' in results and 'error' not in results['permanova']:
                p_value = results['permanova']['p_value']
                print(f"\\n🎯 MAIN RESULT:")
                print(f"   P-value: {p_value:.6f}")
                if p_value < 0.05:
                    print(f" SIGNIFICANT: HRV features differ significantly between")
                    print(f"      pre-seizure and control periods (p < 0.05)")
                else:
                    print(f" NOT SIGNIFICANT: No significant differences found")
                    print(f"      between pre-seizure and control periods")
            
            print(f"\\n Data processed:")
            print(f"   • {results.get('processed_runs', 0)} seizure runs")
            print(f"   • {results.get('n_pre_seizure_windows', 0)} pre-seizure windows")
            print(f"   • {results.get('n_control_windows', 0)} control windows")
            
        else:
            print("\\nAnalysis failed - no results generated")
            
    except Exception as e:
        print(f"\\nAnalysis failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
