#!/usr/bin/env python3
"""
Enhanced Batch Predictor with File-Level Caching
Efficient batch prediction with full feature matrix caching
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Tuple, Any
from pathlib import Path
import logging
from tqdm import tqdm
from .file_feature_cache import FileFeatureCache

logger = logging.getLogger(__name__)


class BatchPredictorV2:
    """
    Enhanced batch predictor with file-level feature caching
    """
    
    def __init__(self, 
                 batch_size: int = 1000,
                 show_progress: bool = True,
                 skip_errors: bool = True,
                 use_file_cache: bool = True,
                 file_cache_dir: str = "file_feature_cache"):
        """
        Initialize batch predictor
        
        Args:
            batch_size: Number of samples to process at once
            show_progress: Whether to show progress bar
            skip_errors: Whether to skip failed samples and continue
            use_file_cache: Whether to use file-level caching
            file_cache_dir: Directory for file cache
        """
        self.batch_size = batch_size
        self.show_progress = show_progress
        self.skip_errors = skip_errors
        self.error_log = []
        self.use_file_cache = use_file_cache
        
        # Initialize file cache if enabled
        if use_file_cache:
            self.file_cache = FileFeatureCache(cache_dir=file_cache_dir)
        else:
            self.file_cache = None
    
    def extract_features_batch(self,
                              df: pd.DataFrame,
                              feature_extractor: Any,
                              smiles_columns: List[str],
                              feature_type: str = 'combined',
                              combination_method: str = 'mean') -> Tuple[np.ndarray, List[int]]:
        """
        Extract features for entire dataframe in batches
        
        Returns:
            features: Complete feature matrix
            failed_indices: List of failed row indices
        """
        n_samples = len(df)
        failed_indices = []
        all_features = []
        
        # Setup progress bar
        if self.show_progress:
            pbar = tqdm(total=n_samples, desc="Extracting features", unit="samples")
        
        # Process in batches for memory efficiency
        n_batches = (n_samples + self.batch_size - 1) // self.batch_size
        
        for batch_idx in range(n_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min((batch_idx + 1) * self.batch_size, n_samples)
            batch_df = df.iloc[start_idx:end_idx]
            
            # Extract features for batch
            for idx, row in batch_df.iterrows():
                try:
                    # Get SMILES from specified columns
                    smiles_list = []
                    for col in smiles_columns:
                        if col in row and pd.notna(row[col]):
                            smiles_list.append(row[col])
                        else:
                            smiles_list.append(None)
                    
                    # Extract features
                    feat = feature_extractor.extract_combination(
                        smiles_list,
                        feature_type=feature_type,
                        combination_method=combination_method
                    )
                    
                    if feat is not None:
                        all_features.append(feat)
                    else:
                        failed_indices.append(idx)
                        # Add placeholder for failed sample
                        all_features.append(np.zeros_like(all_features[0]) if all_features else None)
                        
                except Exception as e:
                    failed_indices.append(idx)
                    self.error_log.append({
                        'index': idx,
                        'smiles': smiles_list,
                        'error': str(e)
                    })
                    # Add placeholder
                    if all_features:
                        all_features.append(np.zeros_like(all_features[0]))
                    else:
                        # Determine feature size
                        feature_size = feature_extractor.get_feature_size(feature_type, combination_method)
                        all_features.append(np.zeros(feature_size))
                    
                    if not self.skip_errors:
                        raise
            
            # Update progress
            if self.show_progress:
                pbar.update(end_idx - start_idx)
        
        if self.show_progress:
            pbar.close()
        
        # Convert to numpy array
        if all_features:
            X = np.array(all_features)
        else:
            feature_size = feature_extractor.get_feature_size(feature_type, combination_method)
            X = np.zeros((n_samples, feature_size))
        
        return X, failed_indices
    
    def predict_with_cache(self,
                          df: pd.DataFrame,
                          model: Any,
                          feature_extractor: Any,
                          smiles_columns: List[str],
                          feature_type: str = 'combined',
                          combination_method: str = 'mean',
                          input_file: Optional[str] = None) -> Tuple[np.ndarray, List[int]]:
        """
        Predict with file-level caching support
        
        Returns:
            predictions: Array of predictions
            failed_indices: List of failed indices
        """
        n_samples = len(df)
        
        # Get feature extractor parameters
        morgan_bits = getattr(feature_extractor, 'morgan_bits', 1024)
        morgan_radius = getattr(feature_extractor, 'morgan_radius', 2)
        
        # Try to load from cache
        X = None
        failed_indices = []
        
        if self.use_file_cache and input_file and self.file_cache:
            print("\nChecking file-level cache...")
            X = self.file_cache.load_features(
                file_path=input_file,
                feature_type=feature_type,
                morgan_bits=morgan_bits,
                morgan_radius=morgan_radius,
                smiles_columns=smiles_columns,
                combination_method=combination_method,
                descriptor_count=getattr(feature_extractor, 'descriptor_count', 115)
            )
            
            if X is not None:
                print("Loaded features from cache")
                print(f"   Shape: {X.shape}")
                print("   Cache hit - skipping feature extraction")
        
        # If not cached, extract features
        if X is None:
            print("\nExtracting features (not in cache)...")
            X, failed_indices = self.extract_features_batch(
                df=df,
                feature_extractor=feature_extractor,
                smiles_columns=smiles_columns,
                feature_type=feature_type,
                combination_method=combination_method
            )
            
            # Clean NaN values
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Save to cache for next time
            if self.use_file_cache and input_file and self.file_cache:
                print("Saving features to cache...")
                self.file_cache.save_features(
                    features=X,
                    file_path=input_file,
                    feature_type=feature_type,
                    morgan_bits=morgan_bits,
                    morgan_radius=morgan_radius,
                    smiles_columns=smiles_columns,
                    combination_method=combination_method,
                    descriptor_count=getattr(feature_extractor, 'descriptor_count', 115),
                    row_count=n_samples,
                    failed_indices=failed_indices
                )
                print("   Features cached for next use")
        
        # Perform prediction
        print("\nRunning model prediction...")
        try:
            predictions = model.predict(X)
            
            # Mark failed indices with NaN
            for idx in failed_indices:
                if idx < len(predictions):
                    predictions[idx] = np.nan
            
            print(f"   Predictions complete: {len(predictions)} samples")
            
        except Exception as e:
            print(f"Prediction failed: {e}")
            predictions = np.full(n_samples, np.nan)
        
        return predictions, failed_indices
    
    def save_error_log(self, filepath: str = "prediction_errors.log"):
        """Save error log to file"""
        if self.error_log:
            with open(filepath, 'w') as f:
                f.write("Index\tSMILES\tError\n")
                for error in self.error_log:
                    smiles_str = ','.join([s if s else 'None' for s in error['smiles']])
                    f.write(f"{error['index']}\t{smiles_str}\t{error['error']}\n")
            print(f"Error log saved to {filepath} ({len(self.error_log)} errors)")
    
    def get_statistics(self, predictions: np.ndarray) -> Dict[str, float]:
        """Calculate statistics for predictions"""
        valid_preds = predictions[~np.isnan(predictions)]
        
        if len(valid_preds) == 0:
            return {
                'count': 0,
                'mean': np.nan,
                'std': np.nan,
                'min': np.nan,
                'max': np.nan,
                'success_rate': 0.0
            }
        
        return {
            'count': len(valid_preds),
            'mean': np.mean(valid_preds),
            'std': np.std(valid_preds),
            'min': np.min(valid_preds),
            'max': np.max(valid_preds),
            'success_rate': len(valid_preds) / len(predictions) * 100
        }
