#!/usr/bin/env python3
"""
Seizure Event Type Counter

This script connects to the seizure database and counts unique seizure event types
from the annotations table. It provides a comprehensive breakdown of all event types
found in the dataset.

Usage:
    python count_seizure_events.py
"""

import sys
import os
from pathlib import Path
import pandas as pd
from collections import Counter
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Add parent directory to path to import our modules
parent_dir = Path(__file__).absolute().parent.parent
sys.path.append(str(parent_dir))

# Import our custom modules
from data_processing_pipeline import DataDiscovery, AnnotationProcessor

def count_seizure_event_types(data_root="/Volumes/Seizury/ds005873"):
    """
    Count ALL unique event types from the annotations database.
    This includes ALL event types: bckg (background), impd (impaired), sz (seizure), 
    and any other event types present in the eventType column.
    
    Args:
        data_root: Path to the dataset root directory
        
    Returns:
        Dictionary with event type counts and statistics
    """
    print("="*80)
    print("COMPLETE EVENT TYPE ANALYSIS (ALL TYPES)")
    print("="*80)
    
    print(f"Dataset root: {data_root}")
    print("Initializing data discovery and annotation processing...")
    
    # Initialize processors
    discovery = DataDiscovery(data_root)
    discovery.scan_dataset()
    matched_runs = discovery.match_runs()
    
    annotation_processor = AnnotationProcessor()
    
    print(f"Found {len(matched_runs)} total runs in dataset")
    
    # Collect all event types from all annotation files
    all_event_types = []
    all_events = []
    files_processed = 0
    files_with_annotations = 0
    total_events = 0
    
    print("\nProcessing annotation files...")
    
    for i, run in enumerate(matched_runs):
        if run['annotation_file']:
            try:
                # Debug: Try reading the file directly first
                direct_read = None
                try:
                    direct_read = pd.read_csv(run['annotation_file'], sep='\t')
                except Exception as e:
                    print(f"    Direct read failed: {e}")
                
                # Load annotations for this run using AnnotationProcessor
                annotations = annotation_processor.load_annotations(run['annotation_file'])
                files_processed += 1
                
                
                # If AnnotationProcessor returns empty but direct read shows data, use direct read
                if annotations.empty and direct_read is not None and not direct_read.empty:
                    annotations = direct_read
                
                if not annotations.empty and 'eventType' in annotations.columns:
                    files_with_annotations += 1
                    
                    # Debug: Print unique event types in this file
                    unique_events_in_file = annotations['eventType'].unique()
                    
                    # Extract ALL event types (including bckg, impd, sz, etc.)
                    event_types = annotations['eventType'].tolist()
                    all_event_types.extend(event_types)
                    
                    # Store detailed event information for ALL event types
                    for _, event in annotations.iterrows():
                        event_info = {
                            'subject': run['subject'],
                            'session': run['session'],
                            'run': run['run'],
                            'eventType': event.get('eventType', 'Unknown'),
                            'onset': event.get('onset', None),
                            'duration': event.get('duration', None)
                        }
                        all_events.append(event_info)
                    
                    total_events += len(annotations)
                
                # Progress update every 10 files (more frequent for debugging)
                if files_processed % 10 == 0:
                    print(f"  Progress: Processed {files_processed} files, found {len(all_event_types)} total events so far")
                        
            except Exception as e:
                print(f"  ERROR: Failed to process {run['annotation_file']}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    print(f"\nProcessing completed:")
    print(f"  Total runs: {len(matched_runs)}")
    print(f"  Annotation files processed: {files_processed}")
    print(f"  Files with eventType data: {files_with_annotations}")
    print(f"  Total events found: {total_events}")
    print(f"  Total event types collected: {len(all_event_types)}")
    
    # Debug: Show first few event types found
    if all_event_types:
        print(f"  First 20 event types found: {all_event_types[:20]}")
        print(f"  Unique event types preview: {list(set(all_event_types))[:10]}")
    
    if not all_event_types:
        print("\nNo event types found in the dataset!")
        print("This suggests either:")
        print("  1. No annotation files were found")
        print("  2. Annotation files exist but don't contain 'eventType' column")
        print("  3. All annotation files are empty")
        print("  4. There's an issue with the AnnotationProcessor.load_annotations() method")
        return {}
    
    # Count unique event types
    event_type_counts = Counter(all_event_types)
    
    print(f"\n" + "="*80)
    print("ALL UNIQUE EVENT TYPES (bckg, impd, sz, etc.)")
    print("="*80)
    
    print(f"\nTotal unique event types: {len(event_type_counts)}")
    print(f"Total events: {sum(event_type_counts.values())}")
    
    print(f"\nEvent Type Breakdown:")
    print("-" * 50)
    
    # Sort by count (descending) for better readability
    sorted_events = sorted(event_type_counts.items(), key=lambda x: x[1], reverse=True)
    
    for event_type, count in sorted_events:
        percentage = (count / total_events) * 100 if total_events > 0 else 0
        print(f"  {event_type:<30}: {count:>6} events ({percentage:>5.1f}%)")
    
    # Additional statistics
    print(f"\n" + "="*80)
    print("STATISTICAL SUMMARY")
    print("="*80)
    
    # Convert to DataFrame for additional analysis
    events_df = pd.DataFrame(all_events)
    
    if not events_df.empty:
        print(f"\nDataset Coverage:")
        unique_subjects = events_df['subject'].nunique()
        unique_sessions = events_df.groupby('subject')['session'].nunique().sum()
        unique_runs = len(events_df.groupby(['subject', 'session', 'run']).size())
        
        print(f"  Unique subjects with events: {unique_subjects}")
        print(f"  Unique sessions with events: {unique_sessions}")
        print(f"  Unique runs with events: {unique_runs}")
        
        # Event distribution by subject
        events_per_subject = events_df.groupby('subject').size()
        print(f"\nEvents per subject:")
        print(f"  Mean: {events_per_subject.mean():.1f}")
        print(f"  Median: {events_per_subject.median():.1f}")
        print(f"  Min: {events_per_subject.min()}")
        print(f"  Max: {events_per_subject.max()}")
        
        # Most common event types (top 10) - including all types (bckg, impd, sz, etc.)
        print(f"\nTop 10 Most Common Event Types (ALL types including bckg, impd, sz):")
        top_10_events = sorted_events[:10]
        for i, (event_type, count) in enumerate(top_10_events, 1):
            percentage = (count / total_events) * 100 if total_events > 0 else 0
            print(f"  {i:>2}. {event_type:<25}: {count:>6} ({percentage:>5.1f}%)")
        
        
    
    # Return results dictionary
    results = {
        'total_events': total_events,
        'unique_event_types': len(event_type_counts),
        'event_type_counts': dict(event_type_counts),
        'files_processed': files_processed,
        'files_with_annotations': files_with_annotations,
        'sorted_events': sorted_events
    }
    
    return results

def analyze_sz_foc_ia_nm_events(data_root="/Volumes/Seizury/ds005873"):
    """
    Analyze "sz_foc_ia_nm" events with specific filters:
    1) Must occur at least 50 minutes (3000 seconds) after recording start
    2) Select only the first seizure in each recording
    
    Args:
        data_root: Path to the dataset root directory
        
    Returns:
        Dictionary with filtered sz_foc_ia_nm analysis results
    """
    print("\n" + "="*80)
    print("SPECIFIC ANALYSIS: sz_foc_ia_nm EVENTS")
    print("="*80)
    print("Filters applied:")
    print("  1) Events must occur ≥50 minutes (3000 seconds) after recording start")
    print("  2) Only first seizure per recording is counted")
    print("-" * 80)
    
    # Initialize processors
    discovery = DataDiscovery(data_root)
    discovery.scan_dataset()
    matched_runs = discovery.match_runs()
    
    annotation_processor = AnnotationProcessor()
    
    # Store filtered events
    filtered_events = []
    recordings_with_sz_foc_ia_nm = 0
    total_sz_foc_ia_nm_found = 0
    total_sz_foc_ia_nm_after_50min = 0
    files_processed = 0
    
    print(f"\nProcessing {len(matched_runs)} recordings for sz_foc_ia_nm events...")
    
    for run in matched_runs:
        if run['annotation_file']:
            try:
                # Load annotations
                annotations = annotation_processor.load_annotations(run['annotation_file'])
                
                # If AnnotationProcessor returns empty, try direct CSV reading
                if annotations.empty:
                    try:
                        annotations = pd.read_csv(run['annotation_file'], sep='\t')
                    except Exception:
                        continue
                
                files_processed += 1
                
                if not annotations.empty and 'eventType' in annotations.columns:
                    # Filter for sz_foc_ia_nm events
                    sz_events = annotations[annotations['eventType'] == 'sz_foc_ia_nm'].copy()
                    
                    if not sz_events.empty:
                        total_sz_foc_ia_nm_found += len(sz_events)
                        
                        # Apply filter 1: Events must be ≥50 minutes (3000 seconds) after start
                        if 'onset' in sz_events.columns:
                            sz_events_after_50min = sz_events[sz_events['onset'] >= 3000.0].copy()
                            total_sz_foc_ia_nm_after_50min += len(sz_events_after_50min)
                            
                            if not sz_events_after_50min.empty:
                                # Apply filter 2: Select only the first seizure (earliest onset)
                                first_seizure = sz_events_after_50min.loc[sz_events_after_50min['onset'].idxmin()]
                                
                                # Store the filtered event
                                event_info = {
                                    'subject': run['subject'],
                                    'session': run['session'],
                                    'run': run['run'],
                                    'eventType': first_seizure['eventType'],
                                    'onset': first_seizure['onset'],
                                    'duration': first_seizure.get('duration', None),
                                    'onset_minutes': first_seizure['onset'] / 60.0  # Convert to minutes
                                }
                                filtered_events.append(event_info)
                                recordings_with_sz_foc_ia_nm += 1
                                
                                print(f"  Found qualifying sz_foc_ia_nm in {run['subject']}/ses-{run['session']}/run-{run['run']}: "
                                      f"onset at {event_info['onset_minutes']:.1f} minutes")
                        
            except Exception as e:
                print(f"  Error processing {run['annotation_file']}: {e}")
                continue
    
    # Results summary
    print(f"\n" + "="*80)
    print("sz_foc_ia_nm ANALYSIS RESULTS")
    print("="*80)
    
    print(f"Files processed: {files_processed}")
    print(f"Total sz_foc_ia_nm events found (all): {total_sz_foc_ia_nm_found}")
    print(f"sz_foc_ia_nm events after 50+ minutes: {total_sz_foc_ia_nm_after_50min}")
    print(f"Recordings with qualifying first seizures: {recordings_with_sz_foc_ia_nm}")
    print(f"Final count (after both filters): {len(filtered_events)}")
    
    if filtered_events:
        print(f"\nDetailed Results:")
        print("-" * 50)
        
        # Convert to DataFrame for analysis
        events_df = pd.DataFrame(filtered_events)
        
        # Statistics on onset times
        onset_times_minutes = events_df['onset_minutes']
        print(f"Seizure onset time statistics (minutes after recording start):")
        print(f"  Mean: {onset_times_minutes.mean():.1f} minutes")
        print(f"  Median: {onset_times_minutes.median():.1f} minutes")
        print(f"  Min: {onset_times_minutes.min():.1f} minutes")
        print(f"  Max: {onset_times_minutes.max():.1f} minutes")
        print(f"  Standard deviation: {onset_times_minutes.std():.1f} minutes")
        
        # Subject distribution
        subjects_with_seizures = events_df['subject'].nunique()
        print(f"\nSubject distribution:")
        print(f"  Unique subjects with qualifying seizures: {subjects_with_seizures}")
        
        # Show first few results
        print(f"\nFirst 10 qualifying seizures:")
        print("-" * 70)
        for i, event in enumerate(events_df.head(10).itertuples(), 1):
            print(f"  {i:>2}. Subject {event.subject}, Session {event.session}, Run {event.run}: "
                  f"{event.onset_minutes:.1f} min ({event.duration:.1f}s duration)")
        
        # Save results to CSV
        output_file = Path(__file__).parent / "sz_foc_ia_nm_filtered_events.csv"
        events_df.to_csv(output_file, index=False)
        print(f"\nFiltered sz_foc_ia_nm events saved to: {output_file}")
    
    else:
        print("\nNo sz_foc_ia_nm events met both filtering criteria.")
    
    # Return results
    results = {
        'total_sz_foc_ia_nm_found': total_sz_foc_ia_nm_found,
        'sz_foc_ia_nm_after_50min': total_sz_foc_ia_nm_after_50min,
        'recordings_with_qualifying_seizures': recordings_with_sz_foc_ia_nm,
        'final_filtered_count': len(filtered_events),
        'filtered_events': filtered_events,
        'files_processed': files_processed
    }
    
    return results

def main():
    """Main function to run the seizure event type analysis."""
    
    # You can modify this path if needed
    DATA_ROOT = "/Volumes/Seizury/ds005873"
    
    # Check if data root exists
    if not os.path.exists(DATA_ROOT):
        print(f"Error: Data root directory not found: {DATA_ROOT}")
        print("Please update the DATA_ROOT variable in this script to point to your dataset.")
        return
    
    try:
        # Run the main event type analysis
        results = count_seizure_event_types(DATA_ROOT)
        
        if results:
            print(f"\n" + "="*80)
            print("GENERAL ANALYSIS COMPLETED SUCCESSFULLY")
            print("="*80)
            print(f"Summary:")
            print(f"  • Total events analyzed: {results['total_events']}")
            print(f"  • Unique event types found: {results['unique_event_types']}")
            print(f"  • Annotation files processed: {results['files_processed']}")
            print(f"  • Files with event data: {results['files_with_annotations']}")
            
            # Show top 5 event types as final summary
            print(f"\nTop 5 Event Types (including bckg, impd, sz, etc.):")
            for i, (event_type, count) in enumerate(results['sorted_events'][:5], 1):
                percentage = (count / results['total_events']) * 100 if results['total_events'] > 0 else 0
                print(f"  {i}. {event_type}: {count} events ({percentage:.1f}%)")
        else:
            print("\nNo event type data found in the dataset.")
        
        # Run the specific sz_foc_ia_nm analysis
        sz_results = analyze_sz_foc_ia_nm_events(DATA_ROOT)
        
        if sz_results:
            print(f"\n" + "="*80)
            print("SPECIFIC sz_foc_ia_nm ANALYSIS COMPLETED")
            print("="*80)
            print(f"sz_foc_ia_nm Summary:")
            print(f"  • Total sz_foc_ia_nm events found: {sz_results['total_sz_foc_ia_nm_found']}")
            print(f"  • Events after 50+ minutes: {sz_results['sz_foc_ia_nm_after_50min']}")
            print(f"  • Recordings with qualifying seizures: {sz_results['recordings_with_qualifying_seizures']}")
            print(f"  • Final filtered count: {sz_results['final_filtered_count']}")
            
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()