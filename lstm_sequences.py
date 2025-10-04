#!/usr/bin/env python3
"""
LSTM Sequence Construction Module

This module builds sequences from HRV feature CSVs for LSTM training with seizure prediction.
Designed to work with 60-minute overlapping windows (30-minute overlap) created by the 
data processing pipeline, providing dense supervision for every minute.

Key features:
- 60-minute sequences matching pipeline windows
- Dense supervision: predicts all 60 minutes per sequence  
- Proper ictal/post-ictal masking from training
- Patient-level splits to prevent data leakage
- Multi-class seizure proximity prediction

RECENT IMPROVEMENTS (Fixed Issues):
✅ training_mask validation: Now REQUIRED in CSV files - raises error if missing
✅ Unified masking: Combines training_mask with NaN exclusion into final_mask  
✅ NaN handling: Replaces NaN values with 0 and excludes from training via mask
✅ Proper normalization: Fits scaler ONLY on training data, saves for inference
✅ Class weights: Computes from trainable samples only, returns for loss reweighting
✅ Return values: Returns (sequences, labels, final_mask, timestamps, class_weights)
✅ Data types: Ensures float32 for sequences, proper boolean masks
✅ Backward compatibility: Handles old files while encouraging new format

The training_mask column must contain:
- 1 = valid training minute (preictal & interictal)  
- 0 = excluded minute (ictal, 30min postictal, or padded)

Usage:
    python lstm_sequences.py --input-dir /path/to/features --create-splits --normalize
"""

import numpy as np
import pandas as pd
from pathlib import Path
import h5py
from typing import Tuple, Dict, List, Optional
import logging
import joblib
import os
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight


class LSTMSequenceBuilder:
    """
    Build LSTM training sequences from HRV feature CSVs with proper medical constraints.
    
    Creates sequences of format (N, seq_len, n_features) where:
    - N: number of sequences
    - seq_len: sequence length (60 minutes for dense supervision)
    - n_features: number of HRV features
    
    Key improvements:
    - Requires training_mask column for proper seizure prediction
    - Combines training_mask with NaN exclusion into final_mask
    - Normalizes features only on training data to prevent leakage
    - Computes class weights from trainable samples for loss reweighting
    - Returns comprehensive data for masked training with TensorFlow
    
    The final_mask ensures that ictal, post-ictal, padded, and NaN minutes
    are excluded from loss computation during training.
    """
    
    def __init__(self, 
                 seq_len: int = 60,
                 stride: int = 30,
                 history_seconds: float = 3600.0,
                 window_stride_seconds: float = 60.0,
                 normalize_features: bool = True):
        """
        Initialize sequence builder.
        
        Args:
            seq_len: Sequence length (default: 60 for 60-minute windows)
            stride: Stride for sequence generation (default: 30 for 30-minute overlap)
            history_seconds: History length in seconds (default: 3600s = 60 minutes)
            window_stride_seconds: Stride between feature windows (default: 60s = 1 minute)
            normalize_features: Whether to normalize features
        """
        self.seq_len = seq_len
        self.stride = stride
        self.history_seconds = history_seconds
        self.window_stride_seconds = window_stride_seconds
        self.normalize_features = normalize_features
        
        # Verify consistency - for minute-level data
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
        
        # Require training mask column - critical for proper seizure prediction
        if 'training_mask' not in df.columns:
            raise ValueError(
                "Missing required 'training_mask' column in CSV file. "
                "This column must be present and contain:\n"
                "  1 = valid training minute (preictal & interictal)\n"
                "  0 = excluded minute (ictal, 30min postictal, or padded)\n"
                "Please ensure the data processing pipeline creates this column."
            )
        
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
            'bin_1', 'bin_2', 'bin_3', 'bin_4', 'mask', 'training_mask',
            'is_ictal', 'is_postictal', 'recent_seizure_flag', 'is_padded'
        ]
        
        feature_cols = [col for col in df.columns if col not in metadata_cols]
        return feature_cols
    
    def create_sequences_from_recording(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Create sequences from a single recording using 60-minute windows with 30-minute overlap.
        
        This matches the data processing pipeline which creates 60-minute overlapping windows
        with dense supervision (one prediction per minute within each window).
        
        Args:
            df: Features DataFrame for one recording (already contains overlapping windows)
            
        Returns:
            Tuple of (sequences, labels, timestamps, final_masks, class_weights) where:
            - sequences: (N, 60, n_features) for 60-minute windows - float32
            - labels: (N, 60, 4) dense multi-class bin labels for each minute in each window
            - timestamps: (N, 60) timestamps for each minute in each window  
            - final_masks: (N, 60) combined mask excluding NaNs, ictal, postictal, and padded
            - class_weights: Dictionary with class weights for loss reweighting
        """
        if len(df) < self.seq_len:
            return np.array([]), np.array([]), np.array([]), np.array([]), {}
        
        # Get feature columns
        if self.feature_names is None:
            self.feature_names = self.extract_feature_columns(df)
        
        # Extract features, labels, and training masks
        features = df[self.feature_names].values.astype(np.float32)  # (n_minutes, n_features)
        bin_labels = df[['bin_1', 'bin_2', 'bin_3', 'bin_4']].values  # (n_minutes, 4)
        timestamps = df['minute_time'].values
        training_masks = df['training_mask'].values.astype(bool)  # (n_minutes,)
        
        # Handle NaN values - replace with 0 and create NaN mask
        nan_mask = np.isnan(features).any(axis=1)
        if np.any(nan_mask):
            self.logger.warning(f"Found {np.sum(nan_mask)} windows with NaN features - replacing with 0")
            features[nan_mask] = 0.0
        
        # Create final mask: combine training mask and NaN exclusion
        # final_mask = training_mask * (1 - nan_mask) = training_mask AND NOT nan_mask
        final_masks = training_masks & (~nan_mask)  # (n_minutes,) - boolean array
        
        # Check for window information to determine sequence strategy
        if 'window_start_time' in df.columns and 'window_end_time' in df.columns:
            # Data already contains overlapping windows - group by window
            return self._create_window_based_sequences(df, features, bin_labels, timestamps, final_masks)
        else:
            # Fallback to sliding window approach for backward compatibility
            return self._create_sliding_window_sequences(features, bin_labels, timestamps, final_masks)
    
    def _create_window_based_sequences(self, df: pd.DataFrame, features: np.ndarray, 
                                     bin_labels: np.ndarray, timestamps: np.ndarray, 
                                     final_masks: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
        """Create sequences from pre-computed overlapping windows with dense supervision."""
        sequences = []
        seq_labels = []
        seq_timestamps = []
        seq_final_masks = []
        
        # Group by window
        for (window_start, window_end), window_group in df.groupby(['window_start_time', 'window_end_time']):
            window_group = window_group.sort_values('minute_time').reset_index(drop=True)
            
            if len(window_group) != self.seq_len:
                self.logger.warning(f"Window has {len(window_group)} minutes, expected {self.seq_len}")
                continue
            
            # Extract window data directly from pre-processed arrays
            window_start_idx = window_group.index[0] - df.index[0]  # Get relative index
            window_end_idx = window_start_idx + len(window_group)
            
            window_features = features[window_start_idx:window_end_idx].astype(np.float32)  # (60, n_features)
            window_bin_labels = bin_labels[window_start_idx:window_end_idx]  # (60, 4)
            window_timestamps = timestamps[window_start_idx:window_end_idx]  # (60,)
            window_final_masks = final_masks[window_start_idx:window_end_idx]  # (60,)
            
            sequences.append(window_features)
            seq_labels.append(window_bin_labels)
            seq_timestamps.append(window_timestamps)
            seq_final_masks.append(window_final_masks)
        
        if not sequences:
            return np.array([]), np.array([]), np.array([]), np.array([]), {}
        
        # Convert to numpy arrays
        sequences = np.array(sequences, dtype=np.float32)  # (n_windows, 60, n_features)
        seq_labels = np.array(seq_labels)  # (n_windows, 60, 4) - dense supervision
        seq_timestamps = np.array(seq_timestamps)  # (n_windows, 60)
        seq_final_masks = np.array(seq_final_masks, dtype=bool)  # (n_windows, 60)
        
        # Compute class weights from trainable data only
        class_weights = self._compute_class_weights(seq_labels, seq_final_masks)
        
        return sequences, seq_labels, seq_timestamps, seq_final_masks, class_weights
    
    def _create_sliding_window_sequences(self, features: np.ndarray, bin_labels: np.ndarray, 
                                       timestamps: np.ndarray, final_masks: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
        """Fallback sliding window approach for backward compatibility."""
        # Calculate number of sequences
        n_windows = len(features)
        n_sequences = (n_windows - self.seq_len) // self.stride + 1
        
        if n_sequences <= 0:
            return np.array([]), np.array([]), np.array([]), np.array([]), {}
        
        # Create sequences
        sequences = []
        seq_labels = []
        seq_timestamps = []
        seq_final_masks = []
        
        for i in range(n_sequences):
            start_idx = i * self.stride
            end_idx = start_idx + self.seq_len
            
            # Extract sequence and all labels (dense supervision)
            sequence = features[start_idx:end_idx].astype(np.float32)
            sequence_bins = bin_labels[start_idx:end_idx]  # All labels in sequence
            sequence_timestamps = timestamps[start_idx:end_idx]
            sequence_final_masks = final_masks[start_idx:end_idx]
            
            sequences.append(sequence)
            seq_labels.append(sequence_bins)
            seq_timestamps.append(sequence_timestamps)
            seq_final_masks.append(sequence_final_masks)
        
        # Convert to numpy arrays
        sequences = np.array(sequences, dtype=np.float32)  # (n_sequences, seq_len, n_features)
        seq_labels = np.array(seq_labels)  # (n_sequences, seq_len, 4) - dense supervision
        seq_timestamps = np.array(seq_timestamps)  # (n_sequences, seq_len)
        seq_final_masks = np.array(seq_final_masks, dtype=bool)  # (n_sequences, seq_len)
        
        # Compute class weights from trainable data only
        class_weights = self._compute_class_weights(seq_labels, seq_final_masks)
        
        return sequences, seq_labels, seq_timestamps, seq_final_masks, class_weights
    
    def _compute_class_weights(self, labels: np.ndarray, masks: np.ndarray) -> Dict:
        """
        Compute class weights for handling class imbalance.
        
        Args:
            labels: Dense supervision labels (n_sequences, seq_len, 4)
            masks: Final masks (n_sequences, seq_len) - True for trainable samples
            
        Returns:
            Dictionary with class weights for loss reweighting
        """
        if len(labels) == 0:
            return {i: 1.0 for i in range(4)}
        
        # Flatten labels and masks
        labels_flat = labels.reshape(-1, labels.shape[-1])  # (n_sequences * seq_len, 4)
        masks_flat = masks.flatten()  # (n_sequences * seq_len,)
        
        # Get trainable samples only
        trainable_labels = labels_flat[masks_flat]
        
        if len(trainable_labels) == 0:
            self.logger.warning("No trainable samples found - using uniform class weights")
            return {i: 1.0 for i in range(4)}
        
        # Convert one-hot to class indices
        trainable_classes = np.argmax(trainable_labels, axis=1)
        
        # Calculate class weights
        unique_classes = np.unique(trainable_classes)
        if len(unique_classes) > 1:
            class_weights = compute_class_weight('balanced', classes=unique_classes, y=trainable_classes)
            class_weight_dict = {int(cls): float(weight) for cls, weight in zip(unique_classes, class_weights)}
            
            # Fill missing classes with 1.0
            for i in range(4):
                if i not in class_weight_dict:
                    class_weight_dict[i] = 1.0
                    
            self.logger.info(f"Computed class weights: {class_weight_dict}")
        else:
            self.logger.warning("Only one class found in trainable data - using uniform weights")
            class_weight_dict = {i: 1.0 for i in range(4)}
        
        return class_weight_dict
    
    def create_sequences_from_csv(self, csv_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
        """
        Create sequences from features CSV file.
        
        Args:
            csv_path: Path to features CSV
            
        Returns:
            Tuple of (sequences, labels, timestamps, final_masks, class_weights)
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
        all_final_masks = []
        all_subjects = []
        all_recordings = []
        all_class_weights = []
        
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
                
                sequences, labels, timestamps, final_masks, class_weights = self.create_sequences_from_recording(recording_df)
                
                if len(sequences) > 0:
                    all_sequences.append(sequences)
                    all_labels.append(labels)
                    all_timestamps.append(timestamps)
                    all_final_masks.append(final_masks)
                    all_class_weights.append(class_weights)
                    
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
        X = np.vstack(all_sequences).astype(np.float32)
        y = np.concatenate(all_labels)
        timestamps = np.concatenate(all_timestamps)
        final_masks = np.concatenate(all_final_masks)
        
        self.logger.info(f"Created {len(X)} sequences with shape {X.shape}")
        
        # Combine class weights from all recordings
        combined_class_weights = {}
        if all_class_weights:
            # Average class weights across recordings
            for i in range(4):
                weights = [cw.get(i, 1.0) for cw in all_class_weights if cw]
                combined_class_weights[i] = float(np.mean(weights)) if weights else 1.0
        else:
            combined_class_weights = {i: 1.0 for i in range(4)}
        
        self.logger.info(f"Combined class weights: {combined_class_weights}")
        
        # DO NOT normalize features here - this will be done during train/val/test split
        # to prevent data leakage (scaler should only see training data)
        
        # Calculate training statistics using final masks
        if len(y.shape) == 3:  # Dense supervision: (n_windows, seq_len, 4)
            final_masks_flat = final_masks.flatten()  # (n_windows * seq_len,)
            n_trainable = final_masks_flat.sum()
            n_excluded = len(final_masks_flat) - n_trainable
            total_minutes = len(final_masks_flat)
        else:  # Single prediction format: (n_sequences, 4)
            n_trainable = final_masks.sum()
            n_excluded = len(final_masks) - n_trainable
            total_minutes = len(final_masks)
        
        self.logger.info(f"Training minutes: {n_trainable}/{total_minutes} "
                        f"({n_trainable/total_minutes*100:.1f}%)")
        self.logger.info(f"Excluded (ictal/postictal): {n_excluded} ({n_excluded/total_minutes*100:.1f}%)")
        
        # Save to HDF5 for efficient loading
        output_file = output_dir / "sequences.h5"
        with h5py.File(output_file, 'w') as f:
            f.create_dataset('X', data=X, compression='gzip')
            f.create_dataset('y', data=y, compression='gzip')
            f.create_dataset('timestamps', data=timestamps, compression='gzip')
            f.create_dataset('final_masks', data=final_masks, compression='gzip')
            
            # Ensure subjects and recordings are strings before encoding
            subjects_encoded = [str(s).encode() for s in all_subjects]
            recordings_encoded = [str(r).encode() for r in all_recordings]
            
            f.create_dataset('subjects', data=subjects_encoded, compression='gzip')
            f.create_dataset('recordings', data=recordings_encoded, compression='gzip')
            
            # Save metadata - convert numpy types to native Python types
            f.attrs['seq_len'] = int(self.seq_len)
            f.attrs['n_features'] = int(X.shape[-1]) 
            f.attrs['n_sequences'] = int(len(X))
            f.attrs['n_trainable'] = int(n_trainable)
            f.attrs['n_excluded'] = int(n_excluded)
            f.attrs['feature_names'] = [name.encode() for name in self.feature_names]
            f.attrs['normalized'] = False  # Features not normalized yet
            
            # Save class weights as JSON string
            import json
            f.attrs['class_weights'] = json.dumps(combined_class_weights).encode('utf-8')
        
        # Calculate label distribution using final masks
        if len(y.shape) == 3:  # Dense supervision
            y_flat = y.reshape(-1, y.shape[-1])
            final_masks_flat = final_masks.flatten()
            trainable_y = y_flat[final_masks_flat]
            if len(trainable_y) > 0:
                trainable_classes = np.argmax(trainable_y, axis=1)
                label_distribution = pd.Series(trainable_classes).value_counts().to_dict()
            else:
                label_distribution = {}
        else:  # Single prediction
            trainable_y = y[final_masks]
            if len(trainable_y) > 0:
                trainable_classes = np.argmax(trainable_y, axis=1)
                label_distribution = pd.Series(trainable_classes).value_counts().to_dict()
            else:
                label_distribution = {}
        
        # Save metadata as separate files
        metadata = {
            'n_sequences': len(X),
            'n_trainable': int(n_trainable),
            'n_excluded': int(n_excluded),
            'sequence_shape': X.shape,
            'labels_shape': y.shape,
            'feature_names': self.feature_names,
            'label_distribution': label_distribution,
            'class_weights': combined_class_weights,
            'normalized': False,  # Features not normalized yet - will be done during splits
            'dense_supervision': len(y.shape) == 3
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
    
    def load_sequences(self, sequences_file: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        """
        Load sequences from HDF5 file.
        
        Args:
            sequences_file: Path to sequences HDF5 file
            
        Returns:
            Tuple of (X, y, final_masks, metadata)
        """
        with h5py.File(sequences_file, 'r') as f:
            X = f['X'][:]
            y = f['y'][:]
            
            # Handle backward compatibility
            if 'final_masks' in f:
                final_masks = f['final_masks'][:]
            elif 'train_masks' in f:
                # Backward compatibility
                final_masks = f['train_masks'][:]
            else:
                # Create dummy masks if none exist
                self.logger.warning("No masks found in file - creating dummy masks")
                final_masks = np.ones(y.shape[:-1], dtype=bool)
            
            # Parse class weights
            class_weights = {}
            if 'class_weights' in f.attrs:
                try:
                    import json
                    class_weights_data = f.attrs['class_weights']
                    if isinstance(class_weights_data, bytes):
                        class_weights_str = class_weights_data.decode('utf-8')
                        class_weights = json.loads(class_weights_str)
                    elif isinstance(class_weights_data, str):
                        class_weights = json.loads(class_weights_data)
                    else:
                        # Fallback for old format
                        import ast
                        class_weights = ast.literal_eval(str(class_weights_data))
                    
                    # Ensure keys are integers
                    class_weights = {int(k): float(v) for k, v in class_weights.items()}
                except Exception as e:
                    self.logger.warning(f"Failed to parse class_weights: {e}")
                    class_weights = {i: 1.0 for i in range(4)}
            else:
                class_weights = {i: 1.0 for i in range(4)}
            
            metadata = {
                'seq_len': f.attrs['seq_len'],
                'n_features': f.attrs['n_features'],
                'n_sequences': f.attrs['n_sequences'],
                'feature_names': [name.decode() if isinstance(name, bytes) else str(name) for name in f.attrs['feature_names']],
                'normalized': f.attrs.get('normalized', False),
                'class_weights': class_weights
            }
            
            if 'timestamps' in f:
                metadata['timestamps'] = f['timestamps'][:]
            if 'subjects' in f:
                metadata['subjects'] = [s.decode() if isinstance(s, bytes) else str(s) for s in f['subjects'][:]]
            if 'recordings' in f:
                metadata['recordings'] = [r.decode() if isinstance(r, bytes) else str(r) for r in f['recordings'][:]]
        
        return X, y, final_masks, metadata
    
    def load_scaler(self, scaler_path: str) -> StandardScaler:
        """
        Load the fitted feature scaler for inference.
        
        Args:
            scaler_path: Path to the saved scaler file
            
        Returns:
            Fitted StandardScaler
        """
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
        
        scaler = joblib.load(scaler_path)
        self.logger.info(f"Loaded feature scaler from: {scaler_path}")
        return scaler
    
    def normalize_features_for_inference(self, X: np.ndarray, scaler_path: str) -> np.ndarray:
        """
        Normalize features using pre-fitted scaler for inference.
        
        Args:
            X: Input features (n_sequences, seq_len, n_features)
            scaler_path: Path to the saved scaler
            
        Returns:
            Normalized features
        """
        scaler = self.load_scaler(scaler_path)
        
        original_shape = X.shape
        X_reshaped = X.reshape(-1, X.shape[-1])
        X_normalized = scaler.transform(X_reshaped)
        
        return X_normalized.reshape(original_shape).astype(np.float32)
    
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
        X, y, final_masks, metadata = self.load_sequences(sequences_file)
        
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
        X_train, y_train, final_masks_train = X[train_mask], y[train_mask], final_masks[train_mask]
        X_val, y_val, final_masks_val = X[val_mask], y[val_mask], final_masks[val_mask]
        
        # Normalize features on training data only to prevent data leakage
        if self.normalize_features:
            self.logger.info("Normalizing features using training data statistics...")
            original_shape = X_train.shape
            X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
            X_val_reshaped = X_val.reshape(-1, X_val.shape[-1])
            
            # Fit scaler on training data only
            X_train_normalized = self.scaler.fit_transform(X_train_reshaped)
            X_train = X_train_normalized.reshape(original_shape).astype(np.float32)
            
            # Apply same scaling to validation data
            X_val_normalized = self.scaler.transform(X_val_reshaped)
            X_val = X_val_normalized.reshape(X_val.shape).astype(np.float32)
            
            # Save scaler for later use
            scaler_path = output_dir / "feature_scaler.joblib"
            joblib.dump(self.scaler, scaler_path)
            self.logger.info(f"Feature scaler saved to: {scaler_path}")
        
        # Save splits
        train_file = output_dir / "train_sequences.h5"
        val_file = output_dir / "val_sequences.h5"
        
        # Helper function to safely save metadata
        def save_metadata_safely(f, metadata_dict):
            for key, value in metadata_dict.items():
                if key == 'class_weights':
                    import json
                    f.attrs[key] = json.dumps(value).encode('utf-8')
                elif key == 'feature_names':
                    f.attrs[key] = [name.encode('utf-8') if isinstance(name, str) else str(name).encode('utf-8') for name in value]
                elif isinstance(value, (int, float, bool, str)):
                    f.attrs[key] = value
                elif isinstance(value, np.integer):
                    # Convert numpy integers to native Python int
                    f.attrs[key] = int(value)
                elif isinstance(value, np.floating):
                    # Convert numpy floats to native Python float
                    f.attrs[key] = float(value)
                elif isinstance(value, np.bool_):
                    # Convert numpy bool to native Python bool
                    f.attrs[key] = bool(value)
                elif isinstance(value, (list, tuple)) and all(isinstance(x, (int, float, str, np.integer, np.floating)) for x in value):
                    # Handle lists with mixed numpy/python types
                    converted = []
                    for x in value:
                        if isinstance(x, np.integer):
                            converted.append(int(x))
                        elif isinstance(x, np.floating):
                            converted.append(float(x))
                        else:
                            converted.append(x)
                    f.attrs[key] = converted
                else:
                    self.logger.warning(f"Skipping metadata key '{key}' with unsupported type: {type(value)}")

        with h5py.File(train_file, 'w') as f:
            f.create_dataset('X', data=X_train, compression='gzip')
            f.create_dataset('y', data=y_train, compression='gzip')
            f.create_dataset('final_masks', data=final_masks_train, compression='gzip')
            
            # Save array metadata
            for key, value in metadata.items():
                if isinstance(value, (list, np.ndarray)) and len(value) == len(X):
                    if key in ['subjects', 'recordings']:
                        split_value = np.array(value)[train_mask]
                        f.create_dataset(key, data=[s.encode() if isinstance(s, str) else s for s in split_value], compression='gzip')
                    else:
                        f.create_dataset(key, data=np.array(value)[train_mask], compression='gzip')
            
            # Save scalar metadata safely
            scalar_metadata = {k: v for k, v in metadata.items() if not (isinstance(v, (list, np.ndarray)) and len(v) == len(X))}
            scalar_metadata['normalized'] = self.normalize_features  # Update normalization status
            save_metadata_safely(f, scalar_metadata)
        
        with h5py.File(val_file, 'w') as f:
            f.create_dataset('X', data=X_val, compression='gzip')
            f.create_dataset('y', data=y_val, compression='gzip')
            f.create_dataset('final_masks', data=final_masks_val, compression='gzip')
            
            # Save array metadata
            for key, value in metadata.items():
                if isinstance(value, (list, np.ndarray)) and len(value) == len(X):
                    if key in ['subjects', 'recordings']:
                        split_value = np.array(value)[val_mask]
                        f.create_dataset(key, data=[s.encode() if isinstance(s, str) else s for s in split_value], compression='gzip')
                    else:
                        f.create_dataset(key, data=np.array(value)[val_mask], compression='gzip')
            
            # Save scalar metadata safely
            scalar_metadata = {k: v for k, v in metadata.items() if not (isinstance(v, (list, np.ndarray)) and len(v) == len(X))}
            scalar_metadata['normalized'] = self.normalize_features  # Update normalization status
            save_metadata_safely(f, scalar_metadata)
        
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
        X, y, final_masks, metadata = self.load_sequences(sequences_file)
        
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
        X_train, y_train, final_masks_train = X[train_mask], y[train_mask], final_masks[train_mask]
        X_val, y_val, final_masks_val = X[val_mask], y[val_mask], final_masks[val_mask]
        X_test, y_test, final_masks_test = X[test_mask], y[test_mask], final_masks[test_mask]
        
        # Normalize features on training data only to prevent data leakage
        if self.normalize_features:
            self.logger.info("Normalizing features using training data statistics...")
            original_train_shape = X_train.shape
            X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
            X_val_reshaped = X_val.reshape(-1, X_val.shape[-1])
            X_test_reshaped = X_test.reshape(-1, X_test.shape[-1])
            
            # Fit scaler on training data only
            X_train_normalized = self.scaler.fit_transform(X_train_reshaped)
            X_train = X_train_normalized.reshape(original_train_shape).astype(np.float32)
            
            # Apply same scaling to val and test data
            X_val_normalized = self.scaler.transform(X_val_reshaped)
            X_val = X_val_normalized.reshape(X_val.shape).astype(np.float32)
            
            X_test_normalized = self.scaler.transform(X_test_reshaped)
            X_test = X_test_normalized.reshape(X_test.shape).astype(np.float32)
            
            # Save scaler for later use
            scaler_path = output_dir / "feature_scaler.joblib"
            joblib.dump(self.scaler, scaler_path)
            self.logger.info(f"Feature scaler saved to: {scaler_path}")
        
        # Log class distributions - handle dense supervision format
        def get_class_distribution(y_split):
            if len(y_split.shape) == 3:  # Dense supervision: (n_sequences, seq_len, 4)
                y_flat = y_split.reshape(-1, y_split.shape[-1])  # Flatten to (n_minutes, 4)
                class_indices = np.argmax(y_flat, axis=1)  # Convert one-hot to class indices
                return np.bincount(class_indices, minlength=4)
            else:  # Single prediction: (n_sequences, 4)
                class_indices = np.argmax(y_split, axis=1)
                return np.bincount(class_indices, minlength=4)
        
        train_dist = get_class_distribution(y_train)
        val_dist = get_class_distribution(y_val)
        test_dist = get_class_distribution(y_test)
        
        self.logger.info(f"Train: {len(X_train)} sequences, class dist: {train_dist}")
        self.logger.info(f"Val: {len(X_val)} sequences, class dist: {val_dist}")
        self.logger.info(f"Test: {len(X_test)} sequences, class dist: {test_dist}")
        
        # Save splits
        train_file = output_dir / "train_sequences.h5"
        val_file = output_dir / "val_sequences.h5"
        test_file = output_dir / "test_sequences.h5"
        
        # Helper function to save split
        def save_split(file_path, X_split, y_split, final_masks_split, mask):
            with h5py.File(file_path, 'w') as f:
                f.create_dataset('X', data=X_split, compression='gzip')
                f.create_dataset('y', data=y_split, compression='gzip')
                f.create_dataset('final_masks', data=final_masks_split, compression='gzip')
                
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
                        # Handle HDF5-incompatible types by converting to HDF5-safe formats
                        if key == 'normalized':
                            f.attrs[key] = self.normalize_features  # Update normalization status
                        elif key == 'class_weights':
                            # Convert dict to JSON string for HDF5 storage
                            import json
                            f.attrs[key] = json.dumps(value).encode('utf-8')
                        elif key == 'feature_names':
                            # Convert list of strings to encoded bytes
                            f.attrs[key] = [name.encode('utf-8') if isinstance(name, str) else str(name).encode('utf-8') for name in value]
                        elif isinstance(value, (int, float, bool, str)):
                            # Safe scalar types
                            f.attrs[key] = value
                        elif isinstance(value, np.integer):
                            # Convert numpy integers to native Python int
                            f.attrs[key] = int(value)
                        elif isinstance(value, np.floating):
                            # Convert numpy floats to native Python float
                            f.attrs[key] = float(value)
                        elif isinstance(value, np.bool_):
                            # Convert numpy bool to native Python bool
                            f.attrs[key] = bool(value)
                        elif isinstance(value, (list, tuple)) and all(isinstance(x, (int, float, str, np.integer, np.floating)) for x in value):
                            # Handle lists with mixed numpy/python types
                            converted = []
                            for x in value:
                                if isinstance(x, np.integer):
                                    converted.append(int(x))
                                elif isinstance(x, np.floating):
                                    converted.append(float(x))
                                else:
                                    converted.append(x)
                            f.attrs[key] = converted
                        else:
                            # Skip complex objects that can't be saved as HDF5 attributes
                            self.logger.warning(f"Skipping metadata key '{key}' with unsupported type: {type(value)}")
        
        save_split(train_file, X_train, y_train, final_masks_train, train_mask)
        save_split(val_file, X_val, y_val, final_masks_val, val_mask)
        save_split(test_file, X_test, y_test, final_masks_test, test_mask)
        
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
    parser.add_argument('--seq-len', type=int, default=60, help='Sequence length (60 minutes for full windows)')
    parser.add_argument('--history', type=float, default=3600.0, help='History length (seconds, 3600s = 60 minutes)')
    parser.add_argument('--stride', type=int, default=30, help='Sequence stride (30 minutes for window overlap)')
    parser.add_argument('--normalize', action='store_true', help='Normalize features (applied during train/val/test split to prevent data leakage)')
    
    # Patient-level splitting options
    parser.add_argument('--create-splits', action='store_true', 
                       help='Create train/val/test splits (recommended for seizure prediction)') #USE THIS
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
        
        if args.normalize:
            print(f"Feature scaler saved for inference: {Path(args.output_dir) / 'feature_scaler.joblib'}")
        
        print("\nIMPORTANT: Use these patient-level splits for training to avoid data leakage!")
        print("   The same patient should NEVER appear in both training and test sets.")
        print("   Features are normalized using ONLY training data statistics.")
    
    else:
        print("\nTIP: Use --create-splits to create proper patient-level train/val/test splits")
        print("   This prevents data leakage and ensures proper generalization testing.")
        print("   Use --normalize to apply feature scaling during splits (recommended)")


if __name__ == "__main__":
    main()