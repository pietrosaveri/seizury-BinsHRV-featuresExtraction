#!/usr/bin/env python3
"""
HRV Feature Processing Pipeline for Seizure Proximity Probability Prediction

This module processes the entire OpenNeuro dataset with nested folder structure:
data/sub-XXX/ses-XX/eeg|ecg/ containing EEG/ECG files and annotations.

Creates minute-level feature packages with multi-temporal HRV features:
- Features calculated over 3, 5, and 10 minute historical windows
- Each minute gets a bin label based on distance to closest seizure
- 60-minute overlapping windows (30-minute overlap) for training
- Dense supervision: probability distribution over seizure proximity bins
- Ictal and post-ictal periods are masked from training but preserved for inference

Bin definitions (only for non-ictal, non-postictal minutes):
- Bin 1: 0-5 minutes to seizure
- Bin 2: 5-30 minutes to seizure  
- Bin 3: 30-60 minutes to seizure
- Bin 4: >60 minutes to seizure (no seizure in sight)

Seizure masking:
- Ictal minutes: During seizure (onset to onset+duration), excluded from training
- Post-ictal minutes: 30 minutes after seizure onset, excluded from training
- Recent seizure flag: Indicates post-seizure period for model inference
- Training mask: 0 for excluded minutes, 1 for trainable minutes
"""

import os
import glob
import pandas as pd
import numpy as np
import mne
from pathlib import Path
import json
import time
from typing import Dict, List, Tuple, Optional
import warnings
import tempfile
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from urllib.parse import urlparse
import io
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed    



def process_single_run_worker(run_data: Dict, config: Dict) -> Optional[Dict]:
    """Worker function for parallel processing of a single run."""
    try:
        # Initialize processors in worker process (reuse S3 handler)
        s3_handler = S3FileHandler()  # Uses class-level connection
        
        annotation_processor = AnnotationProcessor()
        annotation_processor.s3_handler = s3_handler  # Share S3 handler
        
        hrv_processor = HRVFeatureProcessor(
            sampling_rate=config['sampling_rate'],
            window_size_minutes=config['window_size_minutes'],
            overlap_minutes=config['overlap_minutes']
        )
        hrv_processor.s3_handler = s3_handler  # Share S3 handler
        
        # Check if we have required files
        if not run_data['eeg_file'] or not run_data['ecg_file']:
            return None
        
        # Create output filename
        output_filename = f"{run_data['subject']}_{run_data['session']}_run-{run_data['run']}_features.csv"
        
        if s3_handler.is_s3_path(config['output_dir']):
            output_file = f"{config['output_dir']}/{output_filename}"
        else:
            output_file = Path(config['output_dir']) / output_filename
        
        # Load seizure annotations
        seizure_events = pd.DataFrame()
        if run_data['annotation_file']:
            seizure_events = annotation_processor.load_annotations(run_data['annotation_file'])
        
        # Process recording to extract HRV features
        features_df = hrv_processor.process_recording(
            run_data['eeg_file'], run_data['ecg_file'], seizure_events
        )
        
        if features_df.empty:
            return None
        
        # Save CSV file (S3 or local)
        if s3_handler.is_s3_path(config['output_dir']):
            s3_handler.upload_dataframe_to_s3(features_df, output_file)
        else:
            features_df.to_csv(output_file, index=False)
        
        # Calculate statistics
        bin_counts = {}
        for bin_col in ['bin_1', 'bin_2', 'bin_3', 'bin_4']:
            if bin_col in features_df.columns:
                bin_counts[bin_col] = int(features_df[bin_col].sum())
        
        total_minutes = len(features_df)
        total_windows = len(features_df.groupby(['window_start_time', 'window_end_time']))
        
        # Check for padded data
        padded_minutes = 0
        if 'is_padded' in features_df.columns:
            padded_minutes = int(features_df['is_padded'].sum())
        
        # Calculate seizure status counts
        ictal_minutes = int(features_df['is_ictal'].sum()) if 'is_ictal' in features_df.columns else 0
        postictal_minutes = int(features_df['is_postictal'].sum()) if 'is_postictal' in features_df.columns else 0
        training_minutes = int(features_df['training_mask'].sum()) if 'training_mask' in features_df.columns else total_minutes
        
        return {
            'subject': run_data['subject'],
            'session': run_data['session'],
            'run': run_data['run'],
            'n_minutes': total_minutes,
            'n_windows': total_windows,
            'n_padded_minutes': padded_minutes,
            'n_ictal_minutes': ictal_minutes,
            'n_postictal_minutes': postictal_minutes,
            'n_training_minutes': training_minutes,
            'bin_counts': bin_counts,
            'seizure_events': len(seizure_events),
            'output_file': str(output_file)
        }
        
    except Exception as e:
        import traceback
        error_msg = f"Worker error for {run_data['subject']}/{run_data['session']}/run-{run_data['run']}: {e}"
        print(error_msg)
        print(f"Full traceback: {traceback.format_exc()}")
        return None

def main():
    warnings.filterwarnings('ignore', category=RuntimeWarning)

# Import our specialized modules
from ecg_processing import ECGProcessor
from hrv_features import HRVFeatureExtractor

class S3FileHandler:
    """Handler for S3 file operations with local file fallback."""
    
    _s3_client = None  # Class-level client for reuse
    _s3_available = None  # Class-level availability flag
    
    def __init__(self):
        """Initialize S3 client with connection reuse."""
        if S3FileHandler._s3_client is None:
            try:
                S3FileHandler._s3_client = boto3.client('s3')
                S3FileHandler._s3_available = True
                print("S3 connection established successfully")
            except (NoCredentialsError, ClientError) as e:
                print(f"Warning: S3 not available ({e}). Will use local files only.")
                S3FileHandler._s3_client = None
                S3FileHandler._s3_available = False
        
        self.s3_client = S3FileHandler._s3_client
        self.s3_available = S3FileHandler._s3_available
    
    @classmethod
    def initialize_s3_connection(cls):
        """Pre-initialize S3 connection to avoid repeated initialization."""
        if cls._s3_client is None:
            try:
                cls._s3_client = boto3.client('s3')
                cls._s3_available = True
                print("S3 connection established successfully")
            except (NoCredentialsError, ClientError) as e:
                print(f"Warning: S3 not available ({e}). Will use local files only.")
                cls._s3_client = None
                cls._s3_available = False
    
    def is_s3_path(self, path: str) -> bool:
        """Check if path is an S3 URL."""
        return str(path).startswith('s3://')
    
    def parse_s3_path(self, s3_path: str) -> Tuple[str, str]:
        """Parse S3 path into bucket and key."""
        parsed = urlparse(s3_path)
        bucket = parsed.netloc
        key = parsed.path.lstrip('/')
        return bucket, key
    
    def list_s3_objects(self, s3_path: str, suffix: str = "") -> List[str]:
        """List objects in S3 bucket with optional suffix filter."""
        if not self.s3_available:
            raise RuntimeError("S3 not available")
        
        bucket, prefix = self.parse_s3_path(s3_path)
        
        try:
            paginator = self.s3_client.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=bucket, Prefix=prefix)
            
            objects = []
            for page in pages:
                if 'Contents' in page:
                    for obj in page['Contents']:
                        key = obj['Key']
                        if suffix and not key.endswith(suffix):
                            continue
                        objects.append(f"s3://{bucket}/{key}")
            
            return sorted(objects)
            
        except ClientError as e:
            print(f"Error listing S3 objects: {e}")
            return []
    
    def download_s3_file(self, s3_path: str) -> str:
        """Download S3 file to temporary local file and return local path."""
        if not self.s3_available:
            raise RuntimeError("S3 not available")
        
        bucket, key = self.parse_s3_path(s3_path)
        
        # Create temporary file with same extension
        suffix = Path(key).suffix
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        temp_path = temp_file.name
        temp_file.close()
        
        try:
            self.s3_client.download_file(bucket, key, temp_path)
            return temp_path
        except ClientError as e:
            print(f"Error downloading {s3_path}: {e}")
            os.unlink(temp_path)  # Clean up temp file
            raise
    
    def batch_download_s3_files(self, s3_paths: List[str]) -> Dict[str, str]:
        """Download multiple S3 files concurrently."""
        if not self.s3_available:
            raise RuntimeError("S3 not available")
        
        local_files = {}
        max_workers = min(len(s3_paths), 15)  # Limit concurrent downloads
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_s3path = {
                executor.submit(self.download_s3_file, s3_path): s3_path 
                for s3_path in s3_paths
            }
            
            for future in as_completed(future_to_s3path):
                s3_path = future_to_s3path[future]
                try:
                    local_files[s3_path] = future.result()
                except Exception as e:
                    print(f"Failed to download {s3_path}: {e}")
        
        return local_files
    
    def upload_s3_file(self, local_path: str, s3_path: str):
        """Upload local file to S3."""
        if not self.s3_available:
            raise RuntimeError("S3 not available")
        
        bucket, key = self.parse_s3_path(s3_path)
        
        try:
            self.s3_client.upload_file(local_path, bucket, key)
            print(f"Uploaded {local_path} to {s3_path}")
        except ClientError as e:
            print(f"Error uploading to {s3_path}: {e}")
            raise
    
    def upload_dataframe_to_s3(self, df: pd.DataFrame, s3_path: str):
        """Upload DataFrame directly to S3 as CSV."""
        if not self.s3_available:
            raise RuntimeError("S3 not available")
        
        bucket, key = self.parse_s3_path(s3_path)
        
        try:
            # Convert DataFrame to CSV string
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)
            csv_string = csv_buffer.getvalue()
            
            # Upload to S3
            self.s3_client.put_object(
                Bucket=bucket,
                Key=key,
                Body=csv_string.encode('utf-8'),
                ContentType='text/csv'
            )
            print(f"Uploaded DataFrame to {s3_path}")
        except ClientError as e:
            print(f"Error uploading DataFrame to {s3_path}: {e}")
            raise
    
    def upload_json_to_s3(self, data: dict, s3_path: str):
        """Upload dictionary as JSON to S3."""
        if not self.s3_available:
            raise RuntimeError("S3 not available")
        
        bucket, key = self.parse_s3_path(s3_path)
        
        try:
            json_string = json.dumps(data, indent=2)
            self.s3_client.put_object(
                Bucket=bucket,
                Key=key,
                Body=json_string.encode('utf-8'),
                ContentType='application/json'
            )
            print(f"Uploaded JSON to {s3_path}")
        except ClientError as e:
            print(f"Error uploading JSON to {s3_path}: {e}")
            raise

class DataDiscovery:
    """Module for discovering and organizing dataset files."""
    
    def __init__(self, data_root: str):
        self.data_root = data_root
        self.subjects_data = {}
        self.s3_handler = S3FileHandler()
        
    def scan_dataset(self) -> Dict:
        """Scan the dataset and organize files by subject/session/run."""
        print("Scanning dataset structure...")
        
        if self.s3_handler.is_s3_path(self.data_root):
            return self._scan_s3_dataset()
        else:
            return self._scan_local_dataset()
    
    def _scan_local_dataset(self) -> Dict:
        """Scan local filesystem dataset."""
        # Find all subject directories
        subject_dirs = sorted(glob.glob(str(Path(self.data_root) / "sub-*")))
        
        for subject_dir in subject_dirs:
            subject_id = Path(subject_dir).name
            self.subjects_data[subject_id] = {}
            
            # Find all session directories for this subject
            session_dirs = sorted(glob.glob(os.path.join(subject_dir, "ses-*")))
            
            for session_dir in session_dirs:
                session_id = Path(session_dir).name
                self.subjects_data[subject_id][session_id] = {
                    'eeg_files': [],
                    'ecg_files': [],
                    'annotation_files': []
                }
                
                # Scan EEG directory
                eeg_dir = os.path.join(session_dir, "eeg")
                if os.path.exists(eeg_dir):
                    eeg_files = glob.glob(os.path.join(eeg_dir, "*_eeg.edf"))
                    tsv_files = glob.glob(os.path.join(eeg_dir, "*_events.tsv"))
                    
                    self.subjects_data[subject_id][session_id]['eeg_files'] = sorted(eeg_files)
                    self.subjects_data[subject_id][session_id]['annotation_files'] = sorted(tsv_files)
                
                # Scan ECG directory
                ecg_dir = os.path.join(session_dir, "ecg")
                if os.path.exists(ecg_dir):
                    ecg_files = glob.glob(os.path.join(ecg_dir, "*_ecg.edf"))
                    self.subjects_data[subject_id][session_id]['ecg_files'] = sorted(ecg_files)
        
        return self.subjects_data
    
    def _scan_s3_dataset(self) -> Dict:
        """Scan S3 dataset."""
        print("Scanning S3 dataset...")
        
        # List all objects in the S3 bucket
        all_objects = self.s3_handler.list_s3_objects(self.data_root)
        
        # Organize by subject/session
        for s3_path in all_objects:
            # Extract path components
            # Example: s3://seizury-data/ds005873/sub-001/ses-01/eeg/sub-001_ses-01_task-szMonitoring_run-01_eeg.edf
            path_parts = s3_path.replace(self.data_root + "/", "").split("/")
            
            if len(path_parts) < 4:  # Need at least sub-XXX/ses-XX/modality/file
                continue
                
            subject_id = path_parts[0]  # sub-001
            session_id = path_parts[1]  # ses-01
            modality = path_parts[2]    # eeg or ecg
            filename = path_parts[3]    # actual filename
            
            # Initialize subject if not exists
            if subject_id not in self.subjects_data:
                self.subjects_data[subject_id] = {}
            
            # Initialize session if not exists
            if session_id not in self.subjects_data[subject_id]:
                self.subjects_data[subject_id][session_id] = {
                    'eeg_files': [],
                    'ecg_files': [],
                    'annotation_files': []
                }
            
            # Categorize files by type
            if modality == "eeg":
                if filename.endswith("_eeg.edf"):
                    self.subjects_data[subject_id][session_id]['eeg_files'].append(s3_path)
                elif filename.endswith("_events.tsv"):
                    self.subjects_data[subject_id][session_id]['annotation_files'].append(s3_path)
            elif modality == "ecg":
                if filename.endswith("_ecg.edf"):
                    self.subjects_data[subject_id][session_id]['ecg_files'].append(s3_path)
        
        # Sort file lists
        for subject_id in self.subjects_data:
            for session_id in self.subjects_data[subject_id]:
                self.subjects_data[subject_id][session_id]['eeg_files'].sort()
                self.subjects_data[subject_id][session_id]['ecg_files'].sort()
                self.subjects_data[subject_id][session_id]['annotation_files'].sort()
        
        return self.subjects_data
    
    def match_runs(self) -> List[Dict]:
        """Match EEG and ECG files by run number."""
        matched_runs = []
        
        for subject_id, sessions in self.subjects_data.items():
            for session_id, files in sessions.items():
                # Extract run numbers from EEG files
                eeg_runs = {}
                for eeg_file in files['eeg_files']:
                    filename = Path(eeg_file).name
                    if '_run-' in filename:
                        run_num = filename.split('_run-')[1].split('_')[0]
                        eeg_runs[run_num] = eeg_file
                
                # Extract run numbers from ECG files
                ecg_runs = {}
                for ecg_file in files['ecg_files']:
                    filename = Path(ecg_file).name
                    if '_run-' in filename:
                        run_num = filename.split('_run-')[1].split('_')[0]
                        ecg_runs[run_num] = ecg_file
                
                # Find matching annotation files
                annotation_runs = {}
                for ann_file in files['annotation_files']:
                    filename = Path(ann_file).name
                    if '_run-' in filename:
                        run_num = filename.split('_run-')[1].split('_')[0]
                        annotation_runs[run_num] = ann_file
                
                # Match runs across modalities
                all_runs = set(eeg_runs.keys()) | set(ecg_runs.keys()) | set(annotation_runs.keys())
                
                for run_num in all_runs:
                    run_data = {
                        'subject': subject_id,
                        'session': session_id,
                        'run': run_num,
                        'eeg_file': eeg_runs.get(run_num),
                        'ecg_file': ecg_runs.get(run_num),
                        'annotation_file': annotation_runs.get(run_num)
                    }
                    matched_runs.append(run_data)
        
        return matched_runs
    
    def print_summary(self):
        """Print a summary of discovered data."""
        total_subjects = len(self.subjects_data)
        total_sessions = sum(len(sessions) for sessions in self.subjects_data.values())
        
        print(f"\nDataset Summary:")
        print(f"Total subjects: {total_subjects}")
        print(f"Total sessions: {total_sessions}")
        
        for subject_id, sessions in self.subjects_data.items():
            print(f"\n{subject_id}:")
            for session_id, files in sessions.items():
                eeg_count = len(files['eeg_files'])
                ecg_count = len(files['ecg_files'])
                ann_count = len(files['annotation_files'])
                print(f"  {session_id}: {eeg_count} EEG, {ecg_count} ECG, {ann_count} annotations")

class AnnotationProcessor:
    """Module for processing seizure annotations."""
    
    def __init__(self, s3_handler=None):
        self.event_definitions = self._load_event_definitions()
        self.s3_handler = s3_handler or S3FileHandler()
    
    def _load_event_definitions(self) -> Dict:
        """Load ILAE 2017 seizure event definitions."""
        event_definitions = {
            "1.1": "Focal Aware Motor - Automatisms",
            "1.2": "Focal Aware Motor - Atonic", 
            "1.3": "Focal Aware Motor - Clonic",
            "1.4": "Focal Aware Motor - Epileptic spasms",
            "1.5": "Focal Aware Motor - Hyperkinetic",
            "1.6": "Focal Aware Motor - Myoclonic",
            "1.7": "Focal Aware Motor - Tonic",
            "2.1": "Focal Aware Non-motor - Autonomic",
            "2.2": "Focal Aware Non-motor - Behavioral arrest",
            "2.3": "Focal Aware Non-motor - Cognitive",
            "2.4": "Focal Aware Non-motor - Emotional",
            "2.5": "Focal Aware Non-motor - Sensory",
            "3.1": "Focal Impaired Awareness Motor - Automatisms",
            "3.2": "Focal Impaired Awareness Motor - Atonic",
            "3.3": "Focal Impaired Awareness Motor - Clonic", 
            "3.4": "Focal Impaired Awareness Motor - Epileptic spasms",
            "3.5": "Focal Impaired Awareness Motor - Hyperkinetic",
            "3.6": "Focal Impaired Awareness Motor - Myoclonic",
            "3.7": "Focal Impaired Awareness Motor - Tonic",
            "4.1": "Focal Impaired Awareness Non-motor - Behavioral arrest",
            "4.2": "Focal Impaired Awareness Non-motor - Cognitive",
            "4.3": "Focal Impaired Awareness Non-motor - Emotional",
            "4.4": "Focal Impaired Awareness Non-motor - Sensory",
            "5.1": "Focal to bilateral tonic-clonic - Aware at onset",
            "5.2": "Focal to bilateral tonic-clonic - Impaired awareness at onset",
            "5.3": "Focal to bilateral tonic-clonic - Awareness unknown at onset",
            "6.1": "Generalized Motor - Tonic-clonic",
            "6.2": "Generalized Motor - Clonic", 
            "6.3": "Generalized Motor - Tonic",
            "6.4": "Generalized Motor - Myoclonic",
            "6.5": "Generalized Motor - Myoclonic-tonic-clonic",
            "6.6": "Generalized Motor - Myoclonic-atonic",
            "6.7": "Generalized Motor - Atonic",
            "6.8": "Generalized Motor - Epileptic spasms",
            "7.1": "Generalized Non-motor (absence) - Typical",
            "7.2": "Generalized Non-motor (absence) - Atypical",
            "7.3": "Generalized Non-motor (absence) - Myoclonic",
            "7.4": "Generalized Non-motor (absence) - Eyelid myoclonia"
        }
        
        return event_definitions
    
    def is_seizure_event(self, event_type: str) -> bool:
        """Check if an event type represents a seizure."""
        if pd.isna(event_type) or event_type == '':
            return False
        
        # Convert to string and check if it matches seizure patterns
        event_str = str(event_type).strip().lower()
        
        # Check if it's a numbered seizure type (1.1, 1.2, etc.)
        if event_str in self.event_definitions:
            return True
        
        # Check for seizure-specific patterns in the dataset
        # Based on observed patterns: sz_foc_*, sz_gen_*, etc.
        if event_str.startswith('sz_'):
            return True
        
        if event_type == 'sz':
            return True
            
        # Check for common seizure-related terms
        seizure_terms = [
            'seizure', 'sz', 'focal', 'generalized', 'tonic', 'clonic',
            'myoclonic', 'absence', 'atonic', 'spasm', 'automatism'
        ]
        
        return any(term in event_str for term in seizure_terms)
    
    def load_annotations(self, annotation_file: str) -> pd.DataFrame:
        """Load and process annotation file."""
        local_file = None
        
        try:
            # Handle S3 files
            if self.s3_handler.is_s3_path(annotation_file):
                if not self.s3_handler.s3_available:
                    print(f"Warning: S3 not available, cannot load {annotation_file}")
                    return pd.DataFrame()
                local_file = self.s3_handler.download_s3_file(annotation_file)
                file_to_read = local_file
            else:
                # Local file
                if not os.path.exists(annotation_file):
                    return pd.DataFrame()
                file_to_read = annotation_file
            
            annotations = pd.read_csv(file_to_read, sep='\t')
            
            # Check if we have the expected eventType column
            if 'eventType' not in annotations.columns:
                print(f"Warning: 'eventType' column not found in {annotation_file}")
                return pd.DataFrame()
            
            # Filter for seizure events
            seizure_events = annotations[
                annotations['eventType'].apply(self.is_seizure_event)
            ].copy()
            
            return seizure_events
            
        except Exception as e:
            print(f"Error loading annotations from {annotation_file}: {e}")
            return pd.DataFrame()
        finally:
            # Clean up temporary file if downloaded from S3
            if local_file and os.path.exists(local_file):
                os.unlink(local_file)

class HRVFeatureProcessor:
    """Module for processing ECG to HRV features with seizure proximity bin labeling."""
    
    def __init__(self, sampling_rate: int = 256, window_size_minutes: int = 60, 
                 overlap_minutes: int = 30, s3_handler=None):
        self.sampling_rate = sampling_rate
        self.window_size_minutes = window_size_minutes
        self.overlap_minutes = overlap_minutes
        self.window_size_seconds = window_size_minutes * 60
        self.overlap_seconds = overlap_minutes * 60
        
        # Bin definitions (minutes to seizure)
        self.bin_ranges = [
            (0, 5),     # Bin 1: 0-5 minutes
            (5, 30),    # Bin 2: 5-30 minutes  
            (30, 60),   # Bin 3: 30-60 minutes
            (60, float('inf'))  # Bin 4: >60 minutes
        ]
        
        # Feature time windows (minutes)
        self.feature_windows = {
            '3min': ['RRMean', 'RRMin', 'RRMax', 'RRVar', 'RMSSD', 'SDNN', 'SDSD', 'NN50', 'pNN50', 'SampEn'],
            '5min': ['ApEn', 'SD1', 'SD2', 'SD1toSD2'],
            '10min': ['TOTAL_POWER', 'LF_NORM', 'HF_NORM', 'LF_POWER', 'HF_POWER', 'LF_TO_HF', 'VLF_POWER', 'VLF_NORM']
        }
        
        # Initialize processing modules
        self.ecg_processor = ECGProcessor(sampling_rate=sampling_rate)
        self.hrv_extractor = HRVFeatureExtractor()
        self.s3_handler = s3_handler or S3FileHandler()
        
    def process_recording(self, eeg_file: str, ecg_file: str, 
                         seizure_events: pd.DataFrame) -> pd.DataFrame:
        """Process a recording to extract minute-level HRV feature packages with seizure proximity bins."""
        
        eeg_local_file = None
        ecg_local_file = None
        
        try:
            # Handle S3 files - batch download if both are S3
            s3_files_to_download = []
            if self.s3_handler.is_s3_path(eeg_file):
                s3_files_to_download.append(eeg_file)
            if self.s3_handler.is_s3_path(ecg_file):
                s3_files_to_download.append(ecg_file)
            
            if s3_files_to_download:
                # Batch download for efficiency
                downloaded_files = self.s3_handler.batch_download_s3_files(s3_files_to_download)
                eeg_file_to_read = downloaded_files.get(eeg_file, eeg_file)
                ecg_file_to_read = downloaded_files.get(ecg_file, ecg_file)
                
                # Track temp files for cleanup
                if eeg_file in downloaded_files:
                    eeg_local_file = downloaded_files[eeg_file]
                if ecg_file in downloaded_files:
                    ecg_local_file = downloaded_files[ecg_file]
            else:
                # Local files
                eeg_file_to_read = eeg_file
                ecg_file_to_read = ecg_file
            
            # Load EEG to get recording duration
            raw_eeg = mne.io.read_raw_edf(eeg_file_to_read, preload=True, verbose=False)
            if raw_eeg.info['sfreq'] != self.sampling_rate:
                raw_eeg.resample(self.sampling_rate, verbose=False)
            
            total_duration = raw_eeg.times[-1]  # Total recording duration in seconds
            
            # Load ECG with minimal processing
            raw_ecg = mne.io.read_raw_edf(ecg_file_to_read, preload=True, verbose=False)
            if raw_ecg.info['sfreq'] != self.sampling_rate:
                raw_ecg.resample(self.sampling_rate, verbose=False)
            ecg_data = raw_ecg.get_data()[0]  # Assume single channel
            
            # Clear raw objects to free memory
            del raw_eeg, raw_ecg
            
            # Extract tachogram from ECG
            tachogram_result = self.ecg_processor.process_ecg_to_tachogram(ecg_data)
            
            if len(tachogram_result['filtered_rr']) == 0:
                return pd.DataFrame()
            
            # Extract seizure information (timestamps and durations)
            seizure_timestamps = []
            has_seizures = False
            if not seizure_events.empty and 'onset' in seizure_events.columns:
                seizure_timestamps = seizure_events['onset'].tolist()
                has_seizures = len(seizure_timestamps) > 0
            
            # Check if recording is too short and should be skipped
            total_duration_minutes = total_duration / 60.0
            
            if total_duration_minutes < 60 and not has_seizures:
                # Skip short recordings without seizures - not useful for training
                print(f"Skipping short recording ({total_duration_minutes:.1f} min) with no seizures")
                return pd.DataFrame()
            
            # Create minute-level feature packages
            minute_packages = self._create_minute_packages(
                tachogram_result, seizure_events, total_duration
            )
            
            if not minute_packages:
                return pd.DataFrame()
            
            # Create 60-minute overlapping windows from minute packages
            # For short recordings with seizures, this will handle gracefully
            features_df = self._create_overlapping_windows(minute_packages)
            
            if features_df.empty:
                return pd.DataFrame()
            
            # Add metadata columns
            features_df['subject_id'] = self._extract_subject_id(ecg_file)
            features_df['recording_id'] = Path(ecg_file).stem
            
            # Reorder columns - check if all expected columns exist first
            metadata_cols = ['subject_id', 'recording_id', 'window_start_time', 
                            'window_end_time', 'minute_time', 'minute_in_window']
            bin_cols = ['bin_1', 'bin_2', 'bin_3', 'bin_4']
            seizure_status_cols = ['is_ictal', 'is_postictal', 'recent_seizure_flag']
            special_cols = ['training_mask', 'mask', 'is_padded']
            
            # Only include columns that actually exist
            existing_metadata_cols = [col for col in metadata_cols if col in features_df.columns]
            existing_bin_cols = [col for col in bin_cols if col in features_df.columns] 
            existing_seizure_cols = [col for col in seizure_status_cols if col in features_df.columns]
            existing_special_cols = [col for col in special_cols if col in features_df.columns]
            
            all_system_cols = metadata_cols + bin_cols + seizure_status_cols + special_cols
            feature_cols = [col for col in features_df.columns if col not in all_system_cols]
            
            ordered_cols = (existing_metadata_cols + feature_cols + existing_bin_cols + 
                          existing_seizure_cols + existing_special_cols)
            features_df = features_df[ordered_cols]
            
            return features_df
            
        finally:
            # Clean up temporary files
            if eeg_local_file and os.path.exists(eeg_local_file):
                os.unlink(eeg_local_file)
            if ecg_local_file and os.path.exists(ecg_local_file):
                os.unlink(ecg_local_file)
    
    def _create_minute_packages(self, tachogram_result: Dict, 
                              seizure_events: pd.DataFrame, 
                              total_duration: float) -> List[Dict]:
        """Create minute-level feature packages with multi-temporal HRV features."""
        rr_intervals = tachogram_result['filtered_rr']
        rr_times = tachogram_result['filtered_times']
        
        if len(rr_intervals) == 0:
            return []
        
        # Create minute-by-minute packages
        minute_packages = []
        total_minutes = int(total_duration // 60)
        
        for minute in range(total_minutes):
            minute_start_time = minute * 60.0
            minute_end_time = (minute + 1) * 60.0
            
            # Skip if beyond recording
            if minute_end_time > total_duration:
                break
            
            # Create feature package for this minute
            package = self._create_single_minute_package(
                minute_start_time, rr_intervals, rr_times, seizure_events
            )
            
            if package is not None:
                minute_packages.append(package)
        
        return minute_packages
    
    def _create_single_minute_package(self, minute_time: float, 
                                    rr_intervals: np.ndarray, 
                                    rr_times: np.ndarray,
                                    seizure_events: pd.DataFrame) -> Dict:
        """Create a single minute feature package with multi-temporal features."""
        
        try:
            # Initialize feature package
            package = {
                'minute_time': minute_time,
                'mask': {}  # Track which features are available
            }
            
            # Calculate features for different time windows
            self._add_3min_features(package, minute_time, rr_intervals, rr_times)
            self._add_5min_features(package, minute_time, rr_intervals, rr_times)
            self._add_10min_features(package, minute_time, rr_intervals, rr_times)
            
            # Calculate seizure proximity bin and ictal/post-ictal status
            seizure_info = self._calculate_seizure_status(minute_time, seizure_events)
            package.update(seizure_info)
            
            # Ensure all required fields are present
            required_fields = ['bin_1', 'bin_2', 'bin_3', 'bin_4', 'is_ictal', 'is_postictal', 
                             'recent_seizure_flag', 'training_mask']
            for field in required_fields:
                if field not in package:
                    if field == 'training_mask':
                        package[field] = 1  # Default to usable for training
                    else:
                        package[field] = 0  # Default to 0 if missing
            
            return package
            
        except Exception as e:
            # Return a minimal valid package if there's an error
            return {
                'minute_time': minute_time,
                'mask': {},
                'bin_1': 0, 'bin_2': 0, 'bin_3': 0, 'bin_4': 1,  # Default to bin 4 (no seizure)
                'is_ictal': 0, 'is_postictal': 0, 'recent_seizure_flag': 0, 'training_mask': 1
            }
    
    def _add_3min_features(self, package: Dict, minute_time: float, 
                          rr_intervals: np.ndarray, rr_times: np.ndarray):
        """Add 3-minute historical features to package."""
        features_3min = ['RRMean', 'RRMin', 'RRMax', 'RRVar', 'RMSSD', 'SDNN', 'SDSD', 'NN50', 'pNN50', 'SampEn']
        
        # Get RR data from 3 minutes before current minute
        start_time = minute_time - 180.0  # 3 minutes before
        end_time = minute_time
        
        if start_time >= 0:
            # Extract RR intervals in this window
            window_rr, window_rr_times = self.ecg_processor.extract_tachogram_window(
                rr_intervals, rr_times, start_time, end_time
            )
            
            if len(window_rr) >= 5:
                # Calculate all HRV features
                all_features = self.hrv_extractor.compute_all_features(window_rr, window_rr_times)
                
                # Extract only the 3-minute features
                for feature in features_3min:
                    package[f"{feature}_3"] = all_features.get(feature, 0.0)
                    package['mask'][f"{feature}_3"] = 1
            else:
                # Not enough data - set to 0
                for feature in features_3min:
                    package[f"{feature}_3"] = 0.0
                    package['mask'][f"{feature}_3"] = 0
        else:
            # Not enough history - set to 0
            for feature in features_3min:
                package[f"{feature}_3"] = 0.0
                package['mask'][f"{feature}_3"] = 0
    
    def _add_5min_features(self, package: Dict, minute_time: float, 
                          rr_intervals: np.ndarray, rr_times: np.ndarray):
        """Add 5-minute historical features to package."""
        features_5min = ['ApEn', 'SD1', 'SD2', 'SD1toSD2']
        
        # Get RR data from 5 minutes before current minute
        start_time = minute_time - 300.0  # 5 minutes before
        end_time = minute_time
        
        if start_time >= 0:
            # Extract RR intervals in this window
            window_rr, window_rr_times = self.ecg_processor.extract_tachogram_window(
                rr_intervals, rr_times, start_time, end_time
            )
            
            if len(window_rr) >= 10:  # Need more data for 5-minute features
                # Calculate all HRV features
                all_features = self.hrv_extractor.compute_all_features(window_rr, window_rr_times)
                
                # Extract only the 5-minute features
                for feature in features_5min:
                    package[f"{feature}_5"] = all_features.get(feature, 0.0)
                    package['mask'][f"{feature}_5"] = 1
            else:
                # Not enough data - set to 0
                for feature in features_5min:
                    package[f"{feature}_5"] = 0.0
                    package['mask'][f"{feature}_5"] = 0
        else:
            # Not enough history - set to 0
            for feature in features_5min:
                package[f"{feature}_5"] = 0.0
                package['mask'][f"{feature}_5"] = 0
    
    def _add_10min_features(self, package: Dict, minute_time: float, 
                           rr_intervals: np.ndarray, rr_times: np.ndarray):
        """Add 10-minute historical features to package."""
        features_10min = ['TOTAL_POWER', 'LF_NORM', 'HF_NORM', 'LF_POWER', 'HF_POWER', 'LF_TO_HF', 'VLF_POWER', 'VLF_NORM']
        
        # Get RR data from 10 minutes before current minute
        start_time = minute_time - 600.0  # 10 minutes before
        end_time = minute_time
        
        if start_time >= 0:
            # Extract RR intervals in this window
            window_rr, window_rr_times = self.ecg_processor.extract_tachogram_window(
                rr_intervals, rr_times, start_time, end_time
            )
            
            if len(window_rr) >= 20:  # Need more data for frequency domain features
                # Calculate all HRV features
                all_features = self.hrv_extractor.compute_all_features(window_rr, window_rr_times)
                
                # Extract only the 10-minute features
                for feature in features_10min:
                    package[f"{feature}_10"] = all_features.get(feature, 0.0)
                    package['mask'][f"{feature}_10"] = 1
            else:
                # Not enough data - set to 0
                for feature in features_10min:
                    package[f"{feature}_10"] = 0.0
                    package['mask'][f"{feature}_10"] = 0
        else:
            # Not enough history - set to 0
            for feature in features_10min:
                package[f"{feature}_10"] = 0.0
                package['mask'][f"{feature}_10"] = 0
    
    def _calculate_seizure_status(self, minute_time: float, 
                                 seizure_events: pd.DataFrame) -> Dict:
        """
        Calculate comprehensive seizure status for a minute package.
        
        Args:
            minute_time: Time of the minute in seconds
            seizure_events: DataFrame with 'onset' and optionally 'duration' columns
            
        Returns:
            Dictionary with seizure proximity bins, ictal/postictal status, and training mask
        """
        if seizure_events.empty:
            # No seizures - Bin 4 (>60 minutes)
            return {
                'bin_1': 0, 'bin_2': 0, 'bin_3': 0, 'bin_4': 1,
                'is_ictal': 0, 'is_postictal': 0, 'recent_seizure_flag': 0,
                'training_mask': 1  # Can use for training
            }
        
        # Extract seizure information
        is_ictal = 0
        is_postictal = 0
        recent_seizure_flag = 0
        training_mask = 1  # Default: can use for training
        
        # Check each seizure event
        for _, seizure in seizure_events.iterrows():
            seizure_start = seizure['onset']
            seizure_duration = seizure.get('duration', 60)  # Default 1 minute if no duration
            seizure_end = seizure_start + seizure_duration
            
            # Check if minute is during seizure (ictal)
            if seizure_start <= minute_time < seizure_end:
                is_ictal = 1
                training_mask = 0  # Don't use ictal minutes for training
                break
            
            # Check if minute is in post-ictal refractory period (30 minutes after seizure start)
            time_since_seizure_start = (minute_time - seizure_start) / 60.0  # Convert to minutes
            if 0 <= time_since_seizure_start <= 30:  # Up to 30 minutes after seizure start
                is_postictal = 1
                recent_seizure_flag = 1  # Model feature: knows it's post-seizure
                training_mask = 0  # Don't use post-ictal minutes for training
        
        # If not ictal or post-ictal, calculate proximity bins
        if not is_ictal and not is_postictal:  
            # Find closest seizure for proximity binning
            seizure_timestamps = seizure_events['onset'].tolist()
            min_distance = float('inf')
            for ts in seizure_timestamps:
                distance = abs(ts - minute_time) / 60.0  # Convert to minutes
                if distance < min_distance:
                    min_distance = distance
            
            # Determine bin based on distance (only for non-ictal, non-postictal minutes)
            if 0 <= min_distance < 5:
                bin_labels = {'bin_1': 1, 'bin_2': 0, 'bin_3': 0, 'bin_4': 0}  # Bin 1: 0-5 min
            elif 5 <= min_distance < 30:
                bin_labels = {'bin_1': 0, 'bin_2': 1, 'bin_3': 0, 'bin_4': 0}  # Bin 2: 5-30 min
            elif 30 <= min_distance < 60:
                bin_labels = {'bin_1': 0, 'bin_2': 0, 'bin_3': 1, 'bin_4': 0}  # Bin 3: 30-60 min
            else:
                bin_labels = {'bin_1': 0, 'bin_2': 0, 'bin_3': 0, 'bin_4': 1}  # Bin 4: >=60 min
        else:
            # For ictal/post-ictal minutes, don't assign proximity bins (all zeros)
            bin_labels = {'bin_1': 0, 'bin_2': 0, 'bin_3': 0, 'bin_4': 0}
        
        return {
            **bin_labels,
            'is_ictal': is_ictal,
            'is_postictal': is_postictal, 
            'recent_seizure_flag': recent_seizure_flag,
            'training_mask': training_mask
        }
    
    def _create_overlapping_windows(self, minute_packages: List[Dict]) -> pd.DataFrame:
        """Create 60-minute overlapping windows from minute packages."""
        results = []
        step_size = 30  # 30-minute overlap
        
        if len(minute_packages) < 60:
            # For short recordings, check if they contain seizures
            has_seizures = any(pkg.get('bin_1', 0) == 1 or pkg.get('bin_2', 0) == 1 
                             for pkg in minute_packages)
            
            if not has_seizures:
                return pd.DataFrame()  # Skip short recordings without seizures
            
            # For short recordings with seizures, create a single window with available data
            # Pad with the last available package to reach 60 minutes if needed
            if minute_packages:
                padded_packages = minute_packages[:]
                last_package = minute_packages[-1].copy()
                
                # Pad to 60 minutes by repeating the last package
                while len(padded_packages) < 60:
                    padded_packages.append(last_package.copy())
                
                # Create single window from padded data
                window_start_time = padded_packages[0]['minute_time']
                window_end_time = padded_packages[59]['minute_time'] + 60
                
                for j, package in enumerate(padded_packages):
                    # Check if package has all required fields
                    required_keys = ['minute_time', 'bin_1', 'bin_2', 'bin_3', 'bin_4', 
                                   'is_ictal', 'is_postictal', 'recent_seizure_flag', 'training_mask']
                    if not all(key in package for key in required_keys):
                        continue  # Skip malformed packages
                        
                    # Flatten the package into a row
                    row = {
                        'window_start_time': window_start_time,
                        'window_end_time': window_end_time,
                        'minute_time': package['minute_time'],
                        'minute_in_window': j,
                        'bin_1': package['bin_1'],
                        'bin_2': package['bin_2'], 
                        'bin_3': package['bin_3'],
                        'bin_4': package['bin_4'],
                        'is_ictal': package['is_ictal'],
                        'is_postictal': package['is_postictal'],
                        'recent_seizure_flag': package['recent_seizure_flag'],
                        'training_mask': package['training_mask'],
                        'is_padded': j >= len(minute_packages)  # Mark padded data
                    }
                    
                    # Add all features (excluding metadata and seizure fields)
                    excluded_keys = ['minute_time', 'bin_1', 'bin_2', 'bin_3', 'bin_4', 'mask',
                                   'is_ictal', 'is_postictal', 'recent_seizure_flag', 'training_mask']
                    for key, value in package.items():
                        if key not in excluded_keys:
                            row[key] = value
                    
                    # Add mask as binary features
                    mask_dict = package.get('mask', {})
                    mask_features = []
                    for feature_group in ['3', '5', '10']:
                        for base_feature in self.feature_windows[f'{feature_group}min']:
                            feature_name = f"{base_feature}_{feature_group}"
                            mask_value = mask_dict.get(feature_name, 0)
                            mask_features.append(mask_value)
                    
                    # Convert mask to single value (can be expanded later)
                    if mask_features:
                        row['mask'] = int(np.mean(mask_features))  # Average availability
                    else:
                        row['mask'] = 0  # No features available
                    
                    results.append(row)
                
                return pd.DataFrame(results)
            
            return pd.DataFrame()
        
        # Create overlapping 60-minute windows for normal-length recordings
        for i in range(0, len(minute_packages) - 59, step_size):
            window_packages = minute_packages[i:i+60]
            
            if len(window_packages) != 60:
                continue
            
            window_start_time = window_packages[0]['minute_time']
            window_end_time = window_packages[-1]['minute_time'] + 60
            
            # Create rows for each minute in the window
            for j, package in enumerate(window_packages):
                # Check if package has all required fields
                required_keys = ['minute_time', 'bin_1', 'bin_2', 'bin_3', 'bin_4', 
                               'is_ictal', 'is_postictal', 'recent_seizure_flag', 'training_mask']
                if not all(key in package for key in required_keys):
                    continue  # Skip malformed packages
                    
                # Flatten the package into a row
                row = {
                    'window_start_time': window_start_time,
                    'window_end_time': window_end_time,
                    'minute_time': package['minute_time'],
                    'minute_in_window': j,
                    'bin_1': package['bin_1'],
                    'bin_2': package['bin_2'], 
                    'bin_3': package['bin_3'],
                    'bin_4': package['bin_4'],
                    'is_ictal': package['is_ictal'],
                    'is_postictal': package['is_postictal'],
                    'recent_seizure_flag': package['recent_seizure_flag'],
                    'training_mask': package['training_mask'],
                    'is_padded': False  # Normal recordings are not padded
                }
                
                # Add all features (excluding metadata and seizure fields)
                excluded_keys = ['minute_time', 'bin_1', 'bin_2', 'bin_3', 'bin_4', 'mask',
                               'is_ictal', 'is_postictal', 'recent_seizure_flag', 'training_mask']
                for key, value in package.items():
                    if key not in excluded_keys:
                        row[key] = value
                
                # Add mask as binary features
                mask_dict = package.get('mask', {})
                mask_features = []
                for feature_group in ['3', '5', '10']:
                    for base_feature in self.feature_windows[f'{feature_group}min']:
                        feature_name = f"{base_feature}_{feature_group}"
                        mask_value = mask_dict.get(feature_name, 0)
                        mask_features.append(mask_value)
                
                # Convert mask to single value (can be expanded later)
                if mask_features:
                    row['mask'] = int(np.mean(mask_features))  # Average availability
                else:
                    row['mask'] = 0  # No features available
                
                results.append(row)
        
        return pd.DataFrame(results)
    
    def _extract_subject_id(self, filepath: str) -> str:
        """Extract subject ID from filepath."""
        filename = Path(filepath).name
        if 'sub-' in filename:
            # Return full subject ID like "sub-001" instead of just "001"
            subject_part = filename.split('sub-')[1].split('_')[0]
            return f"sub-{subject_part}"
        return 'unknown'


class DataProcessingPipeline:
    """Main pipeline for HRV feature extraction with seizure proximity bin labeling."""
    
    def __init__(self, data_root: str, output_dir: str = "hrv_features", 
                 n_workers: int = None, use_parallel: bool = True, 
                 top_n_patients: int = None):
        self.data_root = data_root
        self.output_dir = output_dir
        self.s3_handler = S3FileHandler()
        self.use_parallel = use_parallel
        self.top_n_patients = top_n_patients
        
        # Set workers based on system capabilities
        if n_workers is None:
            cpu_count = multiprocessing.cpu_count()
            # For m7i-flex.large (2 vCPUs), use both cores but leave headroom
            if cpu_count <= 2:
                self.n_workers = 2
            else:
                self.n_workers = min(cpu_count - 1, 4)  # Max 4 workers for memory constraints
        else:
            self.n_workers = n_workers
            
        print(f"Configured for {self.n_workers} parallel workers")
        
        # Create output directory only if it's local
        if not self.s3_handler.is_s3_path(output_dir):
            Path(output_dir).mkdir(exist_ok=True)
        
        # Initialize modules
        self.discovery = DataDiscovery(data_root)
        self.annotation_processor = AnnotationProcessor()
        self.hrv_processor = HRVFeatureProcessor()
        
        # Results storage
        self.processing_results = []
        
    def _validate_seizures(self, seizure_events: pd.DataFrame, recording_duration: float) -> pd.DataFrame:
        """
        Validate seizures based on timing criteria.
        
        Args:
            seizure_events: DataFrame with 'onset' and optionally 'duration' columns
            recording_duration: Total duration of recording in seconds
            
        Returns:
            DataFrame with only valid seizures
        """
        if seizure_events.empty:
            return seizure_events
        
        # Make a copy to avoid modifying original
        seizures = seizure_events.copy()
        seizures = seizures.sort_values('onset').reset_index(drop=True)
        
        valid_seizures = []
        
        for idx, seizure in seizures.iterrows():
            onset_time = seizure['onset']
            is_valid = True
            
            # Criterion 1: Must be at least 20 minutes from start of recording
            if onset_time < 20 * 60:  # 20 minutes = 1200 seconds
                is_valid = False
                continue
                
            # Criterion 2: Must not be within 30-minute post-ictal phase of another VALID seizure
            # Only check against seizures we've already validated as valid
            for valid_seizure in valid_seizures:
                valid_onset = valid_seizure['onset']
                time_since_valid = (onset_time - valid_onset) / 60.0  # Convert to minutes
                
                # If current seizure is within 30 minutes of a valid previous seizure, it's invalid
                if 0 < time_since_valid <= 30:
                    is_valid = False
                    break
            
            if is_valid:
                valid_seizures.append(seizure)
        
        if valid_seizures:
            return pd.DataFrame(valid_seizures).reset_index(drop=True)
        else:
            return pd.DataFrame()

    def _count_seizures_per_patient(self, matched_runs):
        """Count total VALID seizures per patient across all runs."""
        patient_seizure_counts = {}
        patient_valid_counts = {}
        patient_total_counts = {}
        
        print("Analyzing seizure distribution across patients...")
        print("Only counting seizures that are:")
        print("  - At least 20 minutes from recording start")
        print("  - Not within 30-minute post-ictal phase of another seizure")
        
        for run_data in matched_runs:
            subject_id = run_data['subject']
            
            if subject_id not in patient_seizure_counts:
                patient_seizure_counts[subject_id] = 0
                patient_valid_counts[subject_id] = 0
                patient_total_counts[subject_id] = 0
            
            # Count seizures in this run
            if run_data['annotation_file']:
                try:
                    # Load all seizures
                    all_seizures = self.annotation_processor.load_annotations(run_data['annotation_file'])
                    patient_total_counts[subject_id] += len(all_seizures)
                    
                    if len(all_seizures) > 0:
                        # Get recording duration (approximate from ECG file if available)
                        recording_duration = 24 * 3600  # Default to 24 hours
                        if run_data['ecg_file']:
                            # Try to get actual duration - for now use default
                            pass
                        
                        # Validate seizures
                        valid_seizures = self._validate_seizures(all_seizures, recording_duration)
                        patient_seizure_counts[subject_id] += len(valid_seizures)
                        patient_valid_counts[subject_id] += len(valid_seizures)
                        
                        # Debug output for recordings with seizures
                        if len(all_seizures) > len(valid_seizures):
                            excluded = len(all_seizures) - len(valid_seizures)
                            print(f"  {subject_id}: {len(all_seizures)} total seizures, {len(valid_seizures)} valid ({excluded} excluded)")
                            
                except Exception as e:
                    print(f"  Warning: Could not load annotations for {subject_id}: {e}")
        
        # Print validation summary
        total_patients = len([p for p in patient_total_counts.values() if p > 0])
        total_seizures = sum(patient_total_counts.values())
        valid_seizures = sum(patient_valid_counts.values())
        
        if total_seizures > 0:
            print(f"\nSeizure Validation Summary:")
            print(f"  Patients with seizures: {total_patients}")
            print(f"  Total seizures found: {total_seizures}")
            print(f"  Valid seizures: {valid_seizures} ({valid_seizures/total_seizures*100:.1f}%)")
            print(f"  Excluded seizures: {total_seizures - valid_seizures}")
                    
        return patient_seizure_counts
    
    def _select_top_patients(self, matched_runs, patient_seizure_counts, top_n):
        """Select only runs from patients with the most VALID seizures."""
        # Sort patients by valid seizure count (descending)
        sorted_patients = sorted(patient_seizure_counts.items(), 
                               key=lambda x: x[1], reverse=True)
        
        print(f"\nPatient valid seizure distribution:")
        for i, (patient_id, count) in enumerate(sorted_patients[:10]):  # Show top 10
            print(f"  {i+1}. {patient_id}: {count} valid seizures")
        
        if len(sorted_patients) > 10:
            print(f"  ... and {len(sorted_patients) - 10} more patients")
            
        # Select top N patients based on valid seizure count
        top_patients = [patient_id for patient_id, count in sorted_patients[:top_n]]
        
        print(f"\nSelected top {top_n} patients with most valid seizures:")
        for patient_id in top_patients:
            count = patient_seizure_counts[patient_id]
            print(f"  {patient_id}: {count} valid seizures")
        
        # Filter runs to only include selected patients
        filtered_runs = [run for run in matched_runs if run['subject'] in top_patients]
        
        print(f"\nFiltered from {len(matched_runs)} to {len(filtered_runs)} runs")
        return filtered_runs
    
    def process_dataset(self):
        """Process the entire dataset to extract HRV features."""
        print("Starting HRV feature extraction pipeline...")
        print("Configuration:")
        print(f"  Window size: {self.hrv_processor.window_size_minutes} minutes")
        print(f"  Overlap: {self.hrv_processor.overlap_minutes} minutes")
        print(f"  Approach: Minute-level packages with seizure proximity bins")
        print(f"  Feature windows: 3min, 5min, 10min historical features")
        print(f"  Seizure proximity bins: 0-5min, 5-30min, 30-60min, >60min")
        print(f"  Parallel workers: {self.n_workers}")
        if self.top_n_patients:
            print(f"  Patient selection: Top {self.top_n_patients} patients with most valid seizures")
        
        # Step 1: Discover data
        start_time = time.time()
        self.discovery.scan_dataset()
        self.discovery.print_summary()
        
        matched_runs = self.discovery.match_runs()
        print(f"\nFound {len(matched_runs)} matched runs to process")
        discovery_time = time.time() - start_time
        print(f"Discovery completed in {discovery_time:.2f}s")
        
        # Step 1.5: Optional patient selection based on valid seizure count
        if self.top_n_patients:
            patient_seizure_counts = self._count_seizures_per_patient(matched_runs)
            matched_runs = self._select_top_patients(matched_runs, patient_seizure_counts, self.top_n_patients)
        
        # Step 2: Process runs (parallel or sequential)
        if self.use_parallel and self.n_workers > 1:
            self._process_parallel(matched_runs)
        else:
            self._process_sequential(matched_runs)
        
        # Step 3: Save comprehensive results
        self._save_results()
    
    def _process_parallel(self, matched_runs: List[Dict]):
        """Process runs using parallel workers."""
        print(f"\nStarting parallel processing with {self.n_workers} workers...")
        
        # Create serializable config for workers
        worker_config = {
            'output_dir': self.output_dir,
            'sampling_rate': self.hrv_processor.sampling_rate,
            'window_size_minutes': self.hrv_processor.window_size_minutes,
            'overlap_minutes': self.hrv_processor.overlap_minutes
        }
        
        start_time = time.time()
        completed = 0
        
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            # Submit all jobs
            future_to_run = {
                executor.submit(process_single_run_worker, run_data, worker_config): (i, run_data)
                for i, run_data in enumerate(matched_runs)
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_run):
                i, run_data = future_to_run[future]
                completed += 1
                
                try:
                    result = future.result(timeout=600)  # 10 min timeout per file
                    if result:
                        self.processing_results.append(result)
                        elapsed = time.time() - start_time
                        rate = completed / elapsed if elapsed > 0 else 0
                        eta = (len(matched_runs) - completed) / rate if rate > 0 else 0
                        print(f"Completed {completed}/{len(matched_runs)} "
                              f"({completed/len(matched_runs)*100:.1f}%) "
                              f"- Rate: {rate:.2f}/min - ETA: {eta/60:.1f}min")
                    else:
                        print(f"No features extracted for {run_data['subject']}/{run_data['session']}/run-{run_data['run']}")
                        
                except Exception as e:
                    print(f"Failed {run_data['subject']}/{run_data['session']}/run-{run_data['run']}: {e}")
    
    def _process_sequential(self, matched_runs: List[Dict]):
        """Process runs sequentially (fallback method)."""
        print("\nProcessing sequentially...")
        
        start_time = time.time()
        
        for i, run_data in enumerate(matched_runs):
            print(f"\nProcessing run {i+1}/{len(matched_runs)}: {run_data['subject']}/{run_data['session']}/run-{run_data['run']}")
            
            result = self._process_single_run(run_data)
            if result:
                self.processing_results.append(result)
                
            # Progress reporting
            if (i + 1) % 10 == 0 or i == len(matched_runs) - 1:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                eta = (len(matched_runs) - i - 1) / rate if rate > 0 else 0
                print(f"Progress: {i+1}/{len(matched_runs)} ({(i+1)/len(matched_runs)*100:.1f}%) "
                      f"- Rate: {rate*60:.1f}/min - ETA: {eta/60:.1f}min")
        
    def _process_single_run(self, run_data: Dict) -> Optional[Dict]:
        """Process a single run to extract HRV features."""
        try:
            # Check if we have required files
            if not run_data['eeg_file'] or not run_data['ecg_file']:
                print(f"  Skipping - missing EEG or ECG file")
                return None
            
            # Create output filename
            output_filename = f"{run_data['subject']}_{run_data['session']}_run-{run_data['run']}_features.csv"
            
            if self.s3_handler.is_s3_path(self.output_dir):
                output_file = f"{self.output_dir}/{output_filename}"
            else:
                output_file = Path(self.output_dir) / output_filename
            
            # Load seizure annotations
            seizure_events = pd.DataFrame()
            if run_data['annotation_file']:
                seizure_events = self.annotation_processor.load_annotations(run_data['annotation_file'])
                print(f"  Found {len(seizure_events)} seizure events")
            
            # Process recording to extract HRV features
            print(f"  Extracting HRV features...")
            features_df = self.hrv_processor.process_recording(
                run_data['eeg_file'], run_data['ecg_file'], seizure_events
            )
            
            if features_df.empty:
                print(f"  No features extracted")
                return None
            
            # Save CSV file (S3 or local)
            if self.s3_handler.is_s3_path(self.output_dir):
                self.s3_handler.upload_dataframe_to_s3(features_df, output_file)
            else:
                features_df.to_csv(output_file, index=False)
            
            # Calculate statistics
            label_counts = features_df['label'].value_counts().sort_index()
            total_windows = len(features_df)
            
            print(f"  Created {total_windows} windows")
            print(f"  Label distribution: {dict(label_counts)}")
            print(f"  Saved to: {output_filename}")
            
            return {
                'subject': run_data['subject'],
                'session': run_data['session'],
                'run': run_data['run'],
                'n_windows': total_windows,
                'label_counts': label_counts.to_dict(),
                'seizure_events': len(seizure_events),
                'output_file': str(output_file)
            }
            
        except Exception as e:
            print(f"  Error processing run: {e}")
            return None
    
    def _save_results(self):
        """Save processing summary."""
        if not self.processing_results:
            print("No successful processing results to save")
            return
        
        # Create summary DataFrame
        summary_df = pd.DataFrame(self.processing_results)
        
        # Calculate totals
        total_minutes = summary_df['n_minutes'].sum()
        total_windows = summary_df['n_windows'].sum()
        total_seizure_events = summary_df['seizure_events'].sum()
        
        # Calculate seizure status totals 
        total_ictal = summary_df['n_ictal_minutes'].sum() if 'n_ictal_minutes' in summary_df.columns else 0
        total_postictal = summary_df['n_postictal_minutes'].sum() if 'n_postictal_minutes' in summary_df.columns else 0
        total_training = summary_df['n_training_minutes'].sum() if 'n_training_minutes' in summary_df.columns else total_minutes
        total_excluded = total_minutes - total_training
        
        # Calculate overall bin distribution (only for trainable minutes)
        total_bin_1 = sum(result['bin_counts'].get('bin_1', 0) for result in self.processing_results)
        total_bin_2 = sum(result['bin_counts'].get('bin_2', 0) for result in self.processing_results)
        total_bin_3 = sum(result['bin_counts'].get('bin_3', 0) for result in self.processing_results)
        total_bin_4 = sum(result['bin_counts'].get('bin_4', 0) for result in self.processing_results)
        
        print(f"\n{'='*60}")
        print("PROCESSING COMPLETE")
        print(f"{'='*60}")
        print(f"Total runs processed: {len(self.processing_results)}")
        print(f"Total minute packages: {total_minutes:,}")
        print(f"Total 60-min windows: {total_windows:,}")
        print(f"Total seizure events: {total_seizure_events}")
        
        print(f"\nSeizure Status Distribution:")
        print(f"  Ictal minutes (excluded): {total_ictal:,} ({total_ictal/total_minutes*100:.1f}%)")
        print(f"  Post-ictal minutes (excluded): {total_postictal:,} ({total_postictal/total_minutes*100:.1f}%)")
        print(f"  Training minutes (usable): {total_training:,} ({total_training/total_minutes*100:.1f}%)")
        print(f"  Excluded from training: {total_excluded:,} ({total_excluded/total_minutes*100:.1f}%)")
        
        print(f"\nSeizure Proximity Bin Distribution (trainable minutes only):")
        print(f"  Bin 1 (0-5 min): {total_bin_1:,} ({total_bin_1/total_training*100:.1f}% of trainable)")
        print(f"  Bin 2 (5-30 min): {total_bin_2:,} ({total_bin_2/total_training*100:.1f}% of trainable)")
        print(f"  Bin 3 (30-60 min): {total_bin_3:,} ({total_bin_3/total_training*100:.1f}% of trainable)")
        print(f"  Bin 4 (>60 min): {total_bin_4:,} ({total_bin_4/total_training*100:.1f}% of trainable)")
        
        # Save summary
        if self.s3_handler.is_s3_path(self.output_dir):
            summary_file = f"{self.output_dir}/processing_summary.csv"
            self.s3_handler.upload_dataframe_to_s3(summary_df, summary_file)
        else:
            summary_file = Path(self.output_dir) / "processing_summary.csv"
            summary_df.to_csv(summary_file, index=False)
        print(f"\nProcessing summary saved to: {summary_file}")
        
        # Save consolidated dataset info
        dataset_info = {
            'total_runs_processed': len(self.processing_results),
            'total_minute_packages': int(total_minutes),
            'total_windows': int(total_windows),
            'total_seizure_events': int(total_seizure_events),
            'window_size_minutes': self.hrv_processor.window_size_minutes,
            'overlap_minutes': self.hrv_processor.overlap_minutes,
            'window_size_seconds': self.hrv_processor.window_size_seconds,
            'overlap_seconds': self.hrv_processor.overlap_seconds,
            'sampling_rate': self.hrv_processor.sampling_rate,
            'strategy': f'Minute-level packages with seizure proximity bins and {self.hrv_processor.window_size_minutes}min overlapping windows',
            'feature_windows': self.hrv_processor.feature_windows,
            'bin_ranges': self.hrv_processor.bin_ranges,
            'bin_distribution': {
                'bin_1_0_5min': int(total_bin_1),
                'bin_2_5_30min': int(total_bin_2),
                'bin_3_30_60min': int(total_bin_3),
                'bin_4_over_60min': int(total_bin_4)
            }
        }

        if self.s3_handler.is_s3_path(self.output_dir):
            info_file = f"{self.output_dir}/dataset_info.json"
            self.s3_handler.upload_json_to_s3(dataset_info, info_file)
        else:
            info_file = Path(self.output_dir) / "dataset_info.json"
            with open(info_file, 'w') as f:
                json.dump(dataset_info, f, indent=2)
            
        print(f"Dataset info saved to: {info_file}")
        print(f"CSV feature files saved in: {self.output_dir}")
        
        # Create combined CSV
        print(f"\nCreating combined features CSV...")
        if self.s3_handler.is_s3_path(self.output_dir):
            # For S3, we need to list and download feature files to combine them
            print("Note: Combined CSV creation for S3 output not implemented in this version.")
            print("Individual feature files are available in S3.")
        else:
            # Local file combining (existing logic)
            all_csvs = list(Path(self.output_dir).glob("*_features.csv"))
            if all_csvs:
                combined_dfs = []
                for csv_file in all_csvs:
                    df = pd.read_csv(csv_file)
                    combined_dfs.append(df)
                
                combined_df = pd.concat(combined_dfs, ignore_index=True)
                combined_file = Path(self.output_dir) / "combined_features.csv"
                combined_df.to_csv(combined_file, index=False)
                print(f"Combined features saved to: {combined_file}")
                print(f"Total combined windows: {len(combined_df):,}")

def main():
    """Main function to run the HRV feature extraction pipeline."""
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='HRV Feature Extraction Pipeline')
    parser.add_argument('--data-root', type=str, 
                       help='Data root directory (default: /Volumes/Seizury/ds005873)')
    parser.add_argument('--output-dir', type=str,
                       help='Output directory (default: /Volumes/Seizury/HRV/hrv_features)')
    parser.add_argument('--top-n-patients', type=int, default=None,
                       help='Select only top N patients with most valid seizures (≥20min from start, not in 30min post-ictal)')
    parser.add_argument('--n-workers', type=int, default=2,
                       help='Number of parallel workers (default: 2)')
    parser.add_argument('--no-parallel', action='store_true',
                       help='Disable parallel processing')
    
    args = parser.parse_args()
    
    # Configuration - LOCAL FILES (comment out if using S3)
    #data_root = args.data_root or "/Volumes/Seizury/ds005873"
    #output_dir = args.output_dir or "/Volumes/Seizury/HRV/60min_overlapping_hrv_features"

    # Configuration - AWS S3 (comment out if using local files)
    data_root = args.data_root or "s3://seizury-data/ds005873"
    output_dir = args.output_dir or "s3://seizury-data/60min_overlapping_hrv_features"

    # Performance settings for m7i-flex.large (2 vCPUs, 8GB RAM)
    n_workers = args.n_workers  # Use specified number of workers
    use_parallel = not args.no_parallel  # Enable parallel processing unless disabled
    top_n_patients = args.top_n_patients  # Patient selection
    
    print(f"Data source: {data_root}")
    print(f"Output destination: {output_dir}")
    print(f"Parallel processing: {use_parallel} ({n_workers} workers)")
    if top_n_patients:
        print(f"Patient selection: Top {top_n_patients} patients with most valid seizures")
    
    # Pre-initialize S3 connection to avoid repeated messages
    S3FileHandler.initialize_s3_connection()
    
    # Create and run pipeline
    pipeline = DataProcessingPipeline(
        data_root=data_root, 
        output_dir=output_dir,
        n_workers=n_workers,
        use_parallel=use_parallel,
        top_n_patients=top_n_patients
    )
    
    start_time = time.time()
    pipeline.process_dataset()
    total_time = time.time() - start_time
    
    print(f"\nTotal processing time: {total_time/60:.2f} minutes")
    if pipeline.processing_results:
        rate = len(pipeline.processing_results) / (total_time / 60)
        print(f"Processing rate: {rate:.2f} files/minute")

if __name__ == "__main__":
    # Required for multiprocessing on Windows and some Unix systems
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Method already set
    main()