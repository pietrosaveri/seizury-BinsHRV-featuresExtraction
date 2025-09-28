#!/usr/bin/env python3
"""
LSTM Sequence Construction Module

This module builds sequences from HRV feature CSVs for LSTM training,
implementing proper temporal sequences with configurable history length.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import h5py
from typing import Tuple, Dict, List, Optional
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight


class LSTMSequenceBuilder:
    """
    Build LSTM training sequences from HRV feature CSVs.
    
    Creates sequences of format (N, seq_len, n_features) where:
    - N: number of sequences
    - seq_len: sequence length (history / stride)
    - n_features: number of HRV features
    """
    
    def __init__(self, 
                 seq_len: int = 36,
                 stride: int = 1,
                 history_seconds: float = 180.0,
                 window_stride_seconds: float = 5.0,
                 normalize_features: bool = True):
        """
        Initialize sequence builder.
        
        Args:
            seq_len: Sequence length (default: 36 for 180s history with 5s stride)
            stride: Stride for sequence generation (default: 1)
            history_seconds: History length in seconds (default: 180s)
            window_stride_seconds: Stride between feature windows (default: 5s)
            normalize_features: Whether to normalize features
        """
        self.seq_len = seq_len
        self.stride = stride
        self.history_seconds = history_seconds
        self.window_stride_seconds = window_stride_seconds
        self.normalize_features = normalize_features
        
        # Verify consistency
        calculated_seq_len = int(history_seconds / window_stride_seconds)
        if calculated_seq_len != seq_len:
            logging.warning(f"seq_len ({seq_len}) doesn't match calculated value ({calculated_seq_len})")
        
        self.feature_names = None
        self.scaler = StandardScaler() if normalize_features else None
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
    
    def load_features_from_csv(self, csv_path: str) -> pd.DataFrame:
        """
        Load features from CSV file.
        
        Args:
            csv_path: Path to features CSV
            
        Returns:
            Features DataFrame
        """
        df = pd.read_csv(csv_path)
        
        # Verify required columns exist
        required_cols = ['minute_time', 'bin_1', 'bin_2', 'bin_3', 'bin_4']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Sort by time to ensure temporal order
        df = df.sort_values('minute_time').reset_index(drop=True)
        
        return df
    
    def extract_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Extract feature column names from DataFrame.
        
        Args:
            df: Features DataFrame
            
        Returns:
            List of feature column names
        """
        # Exclude metadata and label columns
        metadata_cols = [
            'subject_id', 'recording_id', 'window_start_time', 
            'window_end_time', 'minute_time', 'minute_in_window',
            'bin_1', 'bin_2', 'bin_3', 'bin_4', 'mask'
        ]
        
        feature_cols = [col for col in df.columns if col not in metadata_cols]
        return feature_cols
    
    def create_sequences_from_recording(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create sequences from a single recording.
        
        Args:
            df: Features DataFrame for one recording
            
        Returns:
            Tuple of (sequences, labels, timestamps) where:
            - sequences: (N, seq_len, n_features)
            - labels: (N,) labels for each sequence
            - timestamps: (N,) center timestamps for each sequence
        """
        if len(df) < self.seq_len:
            return np.array([]), np.array([]), np.array([])
        
        # Get feature columns
        if self.feature_names is None:
            self.feature_names = self.extract_feature_columns(df)
        
        # Extract features and labels (now multi-class bins)
        features = df[self.feature_names].values  # (n_minutes, n_features)
        bin_labels = df[['bin_1', 'bin_2', 'bin_3', 'bin_4']].values  # (n_minutes, 4)
        timestamps = df['minute_time'].values
        
        # Handle NaN values
        nan_mask = np.isnan(features).any(axis=1)
        if np.any(nan_mask):
            self.logger.warning(f"Found {np.sum(nan_mask)} windows with NaN features")
            # Option 1: Remove NaN windows (can break temporal continuity)
            # Option 2: Fill with zeros or interpolate
            features[nan_mask] = 0.0  # Simple: replace with zeros
        
        # Calculate number of sequences
        n_windows = len(features)
        n_sequences = (n_windows - self.seq_len) // self.stride + 1
        
        if n_sequences <= 0:
            return np.array([]), np.array([]), np.array([])
        
        # Create sequences
        sequences = np.zeros((n_sequences, self.seq_len, len(self.feature_names)))
        seq_labels = np.zeros(n_sequences, dtype=int)
        seq_timestamps = np.zeros(n_sequences)
        
        for i in range(n_sequences):
            start_idx = i * self.stride
            end_idx = start_idx + self.seq_len
            
            # Extract sequence (history)
            sequences[i] = features[start_idx:end_idx]
            
            # Label is from the last window in the sequence
            seq_labels[i] = labels[end_idx - 1]
            seq_timestamps[i] = timestamps[end_idx - 1]
        
        return sequences, seq_labels, seq_timestamps
    
    def create_sequences_from_csv(self, csv_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create sequences from features CSV file.
        
        Args:
            csv_path: Path to features CSV
            
        Returns:
            Tuple of (sequences, labels, timestamps)
        """
        df = self.load_features_from_csv(csv_path)
        return self.create_sequences_from_recording(df)
    
    def build_dataset_sequences(self, csv_files: List[str], 
                              output_dir: str = "sequences") -> Dict[str, str]:
        """
        Build sequences from multiple CSV files and save to HDF5.
        
        Args:
            csv_files: List of paths to feature CSV files
            output_dir: Output directory for sequence files
            
        Returns:
            Dictionary with paths to saved files
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        all_sequences = []
        all_labels = []
        all_timestamps = []
        all_subjects = []
        all_recordings = []
        
        for csv_file in csv_files:
            self.logger.info(f"Processing {Path(csv_file).name}...")
            
            df = self.load_features_from_csv(csv_file)
            
            if df.empty:
                self.logger.warning(f"Empty DataFrame in {csv_file}")
                continue
            
            # Debug: Check what columns we have
            self.logger.debug(f"Columns in {Path(csv_file).name}: {list(df.columns)}")
            if 'subject_id' in df.columns:
                unique_subjects = df['subject_id'].unique()
                self.logger.debug(f"Subject IDs found: {unique_subjects} (types: {[type(s) for s in unique_subjects]})")
            
            # Group by recording to maintain temporal continuity
            if 'recording_id' in df.columns:
                recordings = df['recording_id'].unique()
            else:
                recordings = ['recording_1']  # Single recording
                df['recording_id'] = 'recording_1'
            
            for recording_id in recordings:
                recording_df = df[df['recording_id'] == recording_id].copy()
                
                if len(recording_df) < self.seq_len:
                    self.logger.warning(f"Recording {recording_id} too short for sequences")
                    continue
                
                sequences, labels, timestamps = self.create_sequences_from_recording(recording_df)
                
                if len(sequences) > 0:
                    all_sequences.append(sequences)
                    all_labels.append(labels)
                    all_timestamps.append(timestamps)
                    
                    # Track metadata - ensure subject_id is string
                    if 'subject_id' in recording_df.columns:
                        subject_id = str(recording_df['subject_id'].iloc[0])
                        # If it's just a number, try to reconstruct from filename
                        if subject_id.isdigit():
                            csv_filename = Path(csv_file).name
                            if 'sub-' in csv_filename:
                                subject_part = csv_filename.split('sub-')[1].split('_')[0]
                                subject_id = f"sub-{subject_part}"
                            else:
                                subject_id = f"sub-{subject_id.zfill(3)}"  # Pad with zeros
                    else:
                        # Extract from filename if no subject_id column
                        csv_filename = Path(csv_file).name
                        if 'sub-' in csv_filename:
                            subject_part = csv_filename.split('sub-')[1].split('_')[0]
                            subject_id = f"sub-{subject_part}"
                        else:
                            subject_id = 'unknown'
                    
                    self.logger.debug(f"Using subject_id: {subject_id} for {len(sequences)} sequences")
                    all_subjects.extend([subject_id] * len(sequences))
                    all_recordings.extend([str(recording_id)] * len(sequences))
        
        if not all_sequences:
            raise ValueError("No sequences could be created from the provided CSV files")
        
        # Combine all sequences
        X = np.vstack(all_sequences)
        y = np.concatenate(all_labels)
        timestamps = np.concatenate(all_timestamps)
        
        self.logger.info(f"Created {len(X)} sequences with shape {X.shape}")
        
        # Normalize features if requested
        if self.normalize_features:
            self.logger.info("Normalizing features...")
            original_shape = X.shape
            X_reshaped = X.reshape(-1, X.shape[-1])  # (N*seq_len, n_features)
            X_normalized = self.scaler.fit_transform(X_reshaped)
            X = X_normalized.reshape(original_shape)
        
        # Calculate class weights for imbalanced data
        class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
        class_weight_dict = {i: w for i, w in enumerate(class_weights)}
        
        # Save to HDF5 for efficient loading
        output_file = output_dir / "sequences.h5"
        with h5py.File(output_file, 'w') as f:
            f.create_dataset('X', data=X, compression='gzip')
            f.create_dataset('y', data=y, compression='gzip')
            f.create_dataset('timestamps', data=timestamps, compression='gzip')
            
            # Ensure subjects and recordings are strings before encoding
            subjects_encoded = [str(s).encode() for s in all_subjects]
            recordings_encoded = [str(r).encode() for r in all_recordings]
            
            f.create_dataset('subjects', data=subjects_encoded, compression='gzip')
            f.create_dataset('recordings', data=recordings_encoded, compression='gzip')
            
            # Save metadata
            f.attrs['seq_len'] = self.seq_len
            f.attrs['n_features'] = X.shape[-1]
            f.attrs['n_sequences'] = len(X)
            f.attrs['feature_names'] = [name.encode() for name in self.feature_names]
            f.attrs['normalized'] = self.normalize_features
            f.attrs['class_weights'] = list(class_weights)
        
        # Save metadata as separate files
        metadata = {
            'n_sequences': len(X),
            'sequence_shape': X.shape,
            'feature_names': self.feature_names,
            'label_distribution': pd.Series(y).value_counts().to_dict(),
            'class_weights': class_weight_dict,
            'normalized': self.normalize_features
        }
        
        metadata_file = output_dir / "metadata.json"
        import json
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.logger.info(f"Saved sequences to {output_file}")
        self.logger.info(f"Label distribution: {metadata['label_distribution']}")
        
        return {
            'sequences_file': str(output_file),
            'metadata_file': str(metadata_file),
            'scaler_file': None  # TODO: Save scaler if needed
        }
    
    def load_sequences(self, sequences_file: str) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Load sequences from HDF5 file.
        
        Args:
            sequences_file: Path to sequences HDF5 file
            
        Returns:
            Tuple of (X, y, metadata)
        """
        with h5py.File(sequences_file, 'r') as f:
            X = f['X'][:]
            y = f['y'][:]
            
            metadata = {
                'seq_len': f.attrs['seq_len'],
                'n_features': f.attrs['n_features'],
                'n_sequences': f.attrs['n_sequences'],
                'feature_names': [name.decode() if isinstance(name, bytes) else str(name) for name in f.attrs['feature_names']],
                'normalized': f.attrs['normalized'],
                'class_weights': f.attrs['class_weights']
            }
            
            if 'timestamps' in f:
                metadata['timestamps'] = f['timestamps'][:]
            if 'subjects' in f:
                metadata['subjects'] = [s.decode() if isinstance(s, bytes) else str(s) for s in f['subjects'][:]]
            if 'recordings' in f:
                metadata['recordings'] = [r.decode() if isinstance(r, bytes) else str(r) for r in f['recordings'][:]]
        
        return X, y, metadata
    
    def create_train_val_split(self, sequences_file: str, 
                             val_subjects: List[str] = None,
                             val_recordings: List[str] = None,
                             val_fraction: float = 0.2,
                             split_by_patient: bool = True) -> Dict[str, str]:
        """
        Create train/validation split avoiding data leakage.
        
        Args:
            sequences_file: Path to sequences file
            val_subjects: List of subjects for validation (leave-one-subject-out)
            val_recordings: List of recordings for validation (leave-one-recording-out)
            val_fraction: Fraction for validation if no explicit subjects/recordings given
            split_by_patient: Whether to split by patient to avoid data leakage (default: True)
            
        Returns:
            Dictionary with paths to train/val files
        """
        X, y, metadata = self.load_sequences(sequences_file)
        
        output_dir = Path(sequences_file).parent
        
        if val_subjects:
            # Leave-one-subject-out split
            subjects = metadata.get('subjects', ['unknown'] * len(X))
            val_mask = np.array([s in val_subjects for s in subjects])
        elif val_recordings:
            # Leave-one-recording-out split
            recordings = metadata.get('recordings', ['unknown'] * len(X))
            val_mask = np.array([r in val_recordings for r in recordings])
        else:
            # Default: Split by patient to avoid data leakage
            if split_by_patient and 'subjects' in metadata:
                subjects = metadata['subjects']
                unique_subjects = list(set(subjects))
                
                # Randomly select subjects for validation
                np.random.seed(42)
                n_val_subjects = max(1, int(len(unique_subjects) * val_fraction))
                val_subjects_selected = np.random.choice(unique_subjects, n_val_subjects, replace=False)
                
                self.logger.info(f"Selected {n_val_subjects} patients for validation: {val_subjects_selected}")
                self.logger.info(f"Remaining {len(unique_subjects) - n_val_subjects} patients for training")
                
                val_mask = np.array([s in val_subjects_selected for s in subjects])
            else:
                # Fallback to random split (NOT recommended for seizure prediction)
                self.logger.warning("Using random split - this may cause data leakage!")
                self.logger.warning("Consider using split_by_patient=True for proper evaluation")
                np.random.seed(42)
                val_mask = np.random.random(len(X)) < val_fraction
        
        train_mask = ~val_mask
        
        # Split data
        X_train, y_train = X[train_mask], y[train_mask]
        X_val, y_val = X[val_mask], y[val_mask]
        
        # Save splits
        train_file = output_dir / "train_sequences.h5"
        val_file = output_dir / "val_sequences.h5"
        
        with h5py.File(train_file, 'w') as f:
            f.create_dataset('X', data=X_train, compression='gzip')
            f.create_dataset('y', data=y_train, compression='gzip')
            for key, value in metadata.items():
                if isinstance(value, (list, np.ndarray)) and len(value) == len(X):
                    f.create_dataset(key, data=np.array(value)[train_mask], compression='gzip')
                else:
                    f.attrs[key] = value
        
        with h5py.File(val_file, 'w') as f:
            f.create_dataset('X', data=X_val, compression='gzip')
            f.create_dataset('y', data=y_val, compression='gzip')
            for key, value in metadata.items():
                if isinstance(value, (list, np.ndarray)) and len(value) == len(X):
                    f.create_dataset(key, data=np.array(value)[val_mask], compression='gzip')
                else:
                    f.attrs[key] = value
        
        self.logger.info(f"Train split: {len(X_train)} sequences")
        self.logger.info(f"Val split: {len(X_val)} sequences")
        
        return {
            'train_file': str(train_file),
            'val_file': str(val_file)
        }
    
    def create_patient_level_splits(self, sequences_file: str, 
                                  test_subjects: List[str] = None,
                                  val_subjects: List[str] = None,
                                  test_fraction: float = 0.2,
                                  val_fraction: float = 0.2) -> Dict[str, str]:
        """
        Create train/val/test splits at patient level to avoid data leakage.
        
        This is the RECOMMENDED approach for seizure prediction to ensure
        the model generalizes across patients, not just across time.
        
        Args:
            sequences_file: Path to sequences file
            test_subjects: Specific subjects for test set
            val_subjects: Specific subjects for validation set  
            test_fraction: Fraction of patients for test set
            val_fraction: Fraction of remaining patients for validation set
            
        Returns:
            Dictionary with paths to train/val/test files
        """
        X, y, metadata = self.load_sequences(sequences_file)
        
        if 'subjects' not in metadata:
            raise ValueError("Patient information not available for patient-level splits")
        
        subjects = metadata['subjects']
        unique_subjects = list(set(subjects))
        self.logger.info(f"Found {len(unique_subjects)} unique patients: {unique_subjects}")
        
        output_dir = Path(sequences_file).parent
        
        if test_subjects is None or val_subjects is None:
            # Automatically create patient splits
            np.random.seed(42)
            np.random.shuffle(unique_subjects)
            
            # Calculate split sizes
            n_test = max(1, int(len(unique_subjects) * test_fraction))
            n_val = max(1, int((len(unique_subjects) - n_test) * val_fraction))
            
            test_subjects = unique_subjects[:n_test]
            val_subjects = unique_subjects[n_test:n_test + n_val]
            train_subjects = unique_subjects[n_test + n_val:]
            
            self.logger.info(f"Automatic patient splits:")
            self.logger.info(f"  Test patients ({n_test}): {test_subjects}")
            self.logger.info(f"  Val patients ({n_val}): {val_subjects}")
            self.logger.info(f"  Train patients ({len(train_subjects)}): {train_subjects}")
        else:
            # Use provided splits
            train_subjects = [s for s in unique_subjects if s not in test_subjects and s not in val_subjects]
            self.logger.info(f"Manual patient splits:")
            self.logger.info(f"  Test patients: {test_subjects}")
            self.logger.info(f"  Val patients: {val_subjects}")
            self.logger.info(f"  Train patients: {train_subjects}")
        
        # Create masks
        test_mask = np.array([s in test_subjects for s in subjects])
        val_mask = np.array([s in val_subjects for s in subjects])
        train_mask = np.array([s in train_subjects for s in subjects])
        
        # Split data
        X_train, y_train = X[train_mask], y[train_mask]
        X_val, y_val = X[val_mask], y[val_mask]
        X_test, y_test = X[test_mask], y[test_mask]
        
        # Log class distributions
        self.logger.info(f"Train: {len(X_train)} sequences, class dist: {np.bincount(y_train)}")
        self.logger.info(f"Val: {len(X_val)} sequences, class dist: {np.bincount(y_val)}")
        self.logger.info(f"Test: {len(X_test)} sequences, class dist: {np.bincount(y_test)}")
        
        # Save splits
        train_file = output_dir / "train_sequences.h5"
        val_file = output_dir / "val_sequences.h5"
        test_file = output_dir / "test_sequences.h5"
        
        # Helper function to save split
        def save_split(file_path, X_split, y_split, mask):
            with h5py.File(file_path, 'w') as f:
                f.create_dataset('X', data=X_split, compression='gzip')
                f.create_dataset('y', data=y_split, compression='gzip')
                
                # Save metadata for this split
                for key, value in metadata.items():
                    if isinstance(value, (list, np.ndarray)) and len(value) == len(X):
                        split_value = np.array(value)[mask]
                        if key in ['subjects', 'recordings']:
                            # Handle string arrays - encode only if they're strings
                            f.create_dataset(key, data=[s.encode() if isinstance(s, str) else s for s in split_value], compression='gzip')
                        else:
                            f.create_dataset(key, data=split_value, compression='gzip')
                    else:
                        f.attrs[key] = value
        
        save_split(train_file, X_train, y_train, train_mask)
        save_split(val_file, X_val, y_val, val_mask)
        save_split(test_file, X_test, y_test, test_mask)
        
        # Save split information
        split_info = {
            'train_subjects': train_subjects,
            'val_subjects': val_subjects if val_subjects else [],
            'test_subjects': test_subjects if test_subjects else [],
            'train_sequences': len(X_train),
            'val_sequences': len(X_val),
            'test_sequences': len(X_test)
        }
        
        import json
        split_info_file = output_dir / "patient_splits.json"
        with open(split_info_file, 'w') as f:
            json.dump(split_info, f, indent=2)
        
        self.logger.info(f"Patient-level splits saved to {output_dir}")
        
        return {
            'train_file': str(train_file),
            'val_file': str(val_file),
            'test_file': str(test_file),
            'split_info_file': str(split_info_file)
        }


def main():
    """Command-line interface for sequence building."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Build LSTM sequences from HRV features')
    parser.add_argument('--input-dir', required=True, help='Directory with feature CSV files')
    parser.add_argument('--output-dir', default='/Volumes/Seizury/HRV/sequences', help='Output directory')
    parser.add_argument('--seq-len', type=int, default=36, help='Sequence length')
    parser.add_argument('--history', type=float, default=180.0, help='History length (seconds)')
    parser.add_argument('--stride', type=int, default=1, help='Sequence stride')
    parser.add_argument('--normalize', action='store_true', help='Normalize features')
    
    # Patient-level splitting options
    parser.add_argument('--create-splits', action='store_true', 
                       help='Create train/val/test splits (recommended for seizure prediction)')
    parser.add_argument('--split-by-patient', action='store_true', default=True,
                       help='Split by patient to avoid data leakage (default: True)')
    parser.add_argument('--test-fraction', type=float, default=0.2, 
                       help='Fraction of patients for test set')
    parser.add_argument('--val-fraction', type=float, default=0.2,
                       help='Fraction of remaining patients for validation set')
    parser.add_argument('--test-subjects', nargs='+', 
                       help='Specific subjects for test set (e.g., --test-subjects sub-01 sub-02)')
    parser.add_argument('--val-subjects', nargs='+',
                       help='Specific subjects for validation set')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    # Find all CSV files
    input_dir = Path(args.input_dir)
    csv_files = list(input_dir.glob("*_features.csv"))
    
    if not csv_files:
        print(f"No feature CSV files found in {input_dir}")
        return
    
    print(f"Found {len(csv_files)} CSV files")
    
    # Build sequences
    builder = LSTMSequenceBuilder(
        seq_len=args.seq_len,
        stride=args.stride,
        history_seconds=args.history,
        normalize_features=args.normalize
    )
    
    # Build main sequences file
    result = builder.build_dataset_sequences(
        [str(f) for f in csv_files],
        output_dir=args.output_dir
    )
    
    print(f"Sequences saved to: {result['sequences_file']}")
    
    # Create splits if requested
    if args.create_splits:
        print("\nCreating patient-level train/val/test splits...")
        
        split_result = builder.create_patient_level_splits(
            sequences_file=result['sequences_file'],
            test_subjects=args.test_subjects,
            val_subjects=args.val_subjects,
            test_fraction=args.test_fraction,
            val_fraction=args.val_fraction
        )
        
        print(f"Train sequences: {split_result['train_file']}")
        print(f"Validation sequences: {split_result['val_file']}")
        print(f"Test sequences: {split_result['test_file']}")
        print(f"Split info: {split_result['split_info_file']}")
        
        print("\n⚠️  IMPORTANT: Use these patient-level splits for training to avoid data leakage!")
        print("   The same patient should NEVER appear in both training and test sets.")
    
    else:
        print("\n💡 TIP: Use --create-splits to create proper patient-level train/val/test splits")
        print("   This prevents data leakage and ensures proper generalization testing.")


if __name__ == "__main__":
    main()