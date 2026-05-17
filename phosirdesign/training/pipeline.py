#!/usr/bin/env python3
"""
Config-driven training pipeline
One-command training workflow for reproducible model comparison
"""

import os
import sys
import argparse
import warnings
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import traceback
from tqdm import tqdm
from time import perf_counter

# Add project root to path when this module is executed directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Import custom modules
from phosirdesign.config.system import (
    ExperimentConfig, ConfigManager, ConfigValidator,
    BatchExperimentConfig, load_config
)
from phosirdesign.core.feature_extractor import FeatureExtractor
from phosirdesign.models.base import ModelFactory, evaluate_model, generate_model_filename
from phosirdesign.training.logger import TrainingLogger
from sklearn.model_selection import KFold
import joblib
from phosirdesign.utils.timing import TimingTracker
from phosirdesign.utils.file_feature_cache import FileFeatureCache
from phosirdesign.utils.training_progress import get_active_monitor
from phosirdesign.utils.workflow_progress import write_progress_event

warnings.filterwarnings('ignore')


def _build_progress_metrics(avg_metrics: Dict, final_metrics: Dict, fold_metrics: List[Dict]) -> Dict[str, float]:
    """Build compact CV metrics for progress displays."""
    progress_metrics: Dict[str, float] = {}

    for metric in ("r2", "mae", "rmse"):
        mean_key = f"{metric}_mean"
        std_key = f"{metric}_std"
        if mean_key in avg_metrics:
            progress_metrics[metric] = float(avg_metrics[mean_key])
            if std_key in avg_metrics:
                progress_metrics[std_key] = float(avg_metrics[std_key])
        elif metric in final_metrics:
            progress_metrics[metric] = float(final_metrics[metric])

    mse_values = []
    for fold in fold_metrics:
        if "mse" in fold:
            mse_values.append(float(fold["mse"]))
        elif "rmse" in fold:
            rmse = float(fold["rmse"])
            mse_values.append(rmse * rmse)

    if mse_values:
        progress_metrics["mse"] = float(np.mean(mse_values))
        progress_metrics["mse_std"] = float(np.std(mse_values))
    elif "mse" in final_metrics:
        progress_metrics["mse"] = float(final_metrics["mse"])
    elif "rmse" in progress_metrics:
        progress_metrics["mse"] = float(progress_metrics["rmse"] ** 2)

    return {
        key: value
        for key, value in progress_metrics.items()
        if np.isfinite(value)
    }


# ========================================
#           Training Pipeline
# ========================================

class TrainingPipeline:
    """Config-driven training pipeline"""
    
    def __init__(self, config: ExperimentConfig):
        """
        Initialize training pipeline
        
        Args:
            config: Experiment configuration
        """
        self.config = config
        self.logger = None
        self.data = None
        self.features = None
        self.targets = None
        
        # Validate config
        if not ConfigValidator.validate_all(config):
            raise ValueError("Configuration validation failed")
        
        print("\n" + "="*60)
        print(f"INFO Training pipeline initialized: {config.name}")
        print("="*60)
        print(f"Model: {config.model.model_type}")
        print(f"Feature: {config.feature.feature_type}")
        print(f"Cross-validation: {config.training.n_folds} folds")
        # Initialize fine-grained timer
        self.timing = TimingTracker()
        
    def load_data(self, target_col: str = None) -> pd.DataFrame:
        """
        Load data
        
        Args:
            target_col: If provided, filter data for this target; else load all
        """
        if target_col is None:
            # Compatibility: if no target specified, load all data (initial check)
            print(f"\nINFO Loading data: {self.config.data.data_path}")
            with self.timing.measure('data_load_train'):
                df = pd.read_csv(self.config.data.data_path)
            print(f"   Raw data: {len(df)} rows, {len(df.columns)} columns")
            
            # Check available target columns
            available_targets = []
            target_stats = {}
            for target in self.config.data.target_columns:
                if target in df.columns:
                    available_targets.append(target)
                    n_valid = df[target].notna().sum()
                    target_stats[target] = n_valid
                    print(f"   {target}: {n_valid} valid values")
            
            if not available_targets:
                raise ValueError(f"No target columns found: {self.config.data.target_columns}")
            
            # Show data selection info based on multi-target strategy
            if not hasattr(self.config.data, 'multi_target_strategy'):
                self.config.data.multi_target_strategy = 'independent'
            
            if self.config.data.multi_target_strategy == 'intersection':
                # Compute intersection size
                valid_mask = pd.Series([True] * len(df))
                for target in available_targets:
                    valid_mask &= df[target].notna()
                n_intersection = valid_mask.sum()
                print(f"\n   Multi-target strategy: intersection")
                print(f"      Rows with values for all targets: {n_intersection}")
                print(f"      Data utilization: {n_intersection/len(df)*100:.1f}%")
            elif self.config.data.multi_target_strategy == 'independent':
                print(f"\n   Multi-target strategy: independent")
                print(f"      Each target uses its own valid data")
            elif self.config.data.multi_target_strategy == 'union':
                print(f"\n   Multi-target strategy: union")
                print(f"      Use all data; missing values will be filled")
            
            self.available_targets = available_targets
            self.target_stats = target_stats
            self.raw_data = df  # save raw data
            return df
        else:
            # Load and filter data for specific target
            if not hasattr(self, 'raw_data'):
                with self.timing.measure('data_load_train'):
                    df = pd.read_csv(self.config.data.data_path)
                self.raw_data = df
            else:
                df = self.raw_data.copy()
            
            # Handle missing values
            if not hasattr(self.config.data, 'nan_handling'):
                self.config.data.nan_handling = 'skip'  # default
            if not hasattr(self.config.data, 'multi_target_strategy'):
                self.config.data.multi_target_strategy = 'independent'  # default
            
            # Process data per multi-target and missing-value strategy
            if self.config.data.multi_target_strategy == 'intersection':
                # Intersection: only rows with values for all targets
                print(f"\n   Using intersection mode for {target_col}")
                valid_mask = pd.Series([True] * len(df))
                for target in self.config.data.target_columns:
                    if target in df.columns:
                        valid_mask &= df[target].notna()
                
                # Check SMILES columns
                if self.config.feature.feature_type in ['morgan', 'descriptors', 'combined']:
                    for col in self.config.data.smiles_columns:
                        if col in df.columns:
                            valid_mask &= df[col].notna()
                
                df_valid = df[valid_mask].copy()
                n_dropped = len(df) - len(df_valid)
                print(f"   Valid {target_col} rows: {len(df_valid)} (intersection mode)")
                # set data and return directly
                self.data = df_valid
                return df_valid
                
            elif self.config.data.multi_target_strategy == 'union':
                # Union: use all data with nan_handling strategy
                print(f"\n   Using union mode for {target_col}")
                df_valid = df.copy()
                
                # Fill missing values per nan_handling strategy
                if self.config.data.nan_handling != 'skip':
                    # Handle missing target values
                    if target_col in df_valid.columns:
                        n_missing = df_valid[target_col].isna().sum()
                        if n_missing > 0:
                            if self.config.data.nan_handling == 'mean':
                                mean_val = df_valid[target_col].mean()
                                df_valid[target_col].fillna(mean_val, inplace=True)
                                print(f"   Filled {n_missing} missing values with mean {mean_val:.4f}")
                            elif self.config.data.nan_handling == 'median':
                                median_val = df_valid[target_col].median()
                                df_valid[target_col].fillna(median_val, inplace=True)
                                print(f"   Filled {n_missing} missing values with median {median_val:.4f}")
                            elif self.config.data.nan_handling == 'zero':
                                df_valid[target_col].fillna(0, inplace=True)
                                print(f"   Filled {n_missing} missing values with 0")
                
                # Missing SMILES still need to be skipped
                if self.config.feature.feature_type in ['morgan', 'descriptors', 'combined']:
                    for col in self.config.data.smiles_columns:
                        if col in df_valid.columns:
                            mask = df_valid[col].notna()
                            n_missing = (~mask).sum()
                            if n_missing > 0:
                                df_valid = df_valid[mask]
                                print(f"   Skipped {n_missing} rows with missing SMILES")
                
                print(f"   Valid {target_col} rows: {len(df_valid)} (union mode)")
                # set data and return directly
                self.data = df_valid
                return df_valid
                
            elif self.config.data.multi_target_strategy == 'independent':
                # Independent: each target handled separately (original logic)
                pass
                
            # Handle missing values under independent strategy (original logic)
            if self.config.data.multi_target_strategy == 'independent' and self.config.data.nan_handling == 'skip':
                # Filter valid data: only check current target and SMILES columns
                valid_mask = pd.Series([True] * len(df))
                
                # Check target column
                if target_col in df.columns:
                    valid_mask &= df[target_col].notna()
                else:
                    raise ValueError(f"Target column not found: {target_col}")
                
                # Check SMILES columns (for molecular features)
                if self.config.feature.feature_type in ['morgan', 'descriptors', 'combined']:
                    for col in self.config.data.smiles_columns:
                        if col in df.columns:
                            valid_mask &= df[col].notna()
                        else:
                            print(f"   Warning: SMILES column not found: {col}")
                elif self.config.feature.feature_type == 'tabular':
                    # Tabular data does not need SMILES columns
                    pass
                
                df_valid = df[valid_mask].copy()
                n_dropped = len(df) - len(df_valid)
                if n_dropped > 0:
                    print(f"   Valid {target_col} rows: {len(df_valid)} (dropped {n_dropped} rows with missing values)")
                else:
                    print(f"   Valid {target_col} rows: {len(df_valid)}")
                    
            else:
                # Other missing value strategies
                df_valid = df.copy()
                
                # Handle missing target values
                if target_col in df_valid.columns:
                    n_missing = df_valid[target_col].isna().sum()
                    if n_missing > 0:
                        if self.config.data.nan_handling == 'mean':
                            mean_val = df_valid[target_col].mean()
                            df_valid[target_col].fillna(mean_val, inplace=True)
                            print(f"   Filled {n_missing} target missing values with mean {mean_val:.4f}")
                        elif self.config.data.nan_handling == 'median':
                            median_val = df_valid[target_col].median()
                            df_valid[target_col].fillna(median_val, inplace=True)
                            print(f"   Filled {n_missing} target missing values with median {median_val:.4f}")
                        elif self.config.data.nan_handling == 'zero':
                            df_valid[target_col].fillna(0, inplace=True)
                            print(f"   Filled {n_missing} target missing values with 0")
                        elif self.config.data.nan_handling == 'forward':
                            df_valid[target_col].fillna(method='ffill', inplace=True)
                            df_valid[target_col].fillna(method='bfill', inplace=True)
                            print(f"   Filled {n_missing} target missing values with forward/backward fill")
                        elif self.config.data.nan_handling == 'interpolate':
                            df_valid[target_col] = df_valid[target_col].interpolate()
                            df_valid[target_col].fillna(method='bfill', inplace=True)
                            df_valid[target_col].fillna(method='ffill', inplace=True)
                            print(f"   Filled {n_missing} target missing values with interpolation")
                
                # Missing SMILES must be skipped
                if self.config.feature.feature_type in ['morgan', 'descriptors', 'combined']:
                    for col in self.config.data.smiles_columns:
                        if col in df_valid.columns:
                            mask = df_valid[col].notna()
                            n_missing = (~mask).sum()
                            if n_missing > 0:
                                df_valid = df_valid[mask]
                                print(f"   Skipped {n_missing} rows with missing SMILES (column: {col})")
                
                print(f"   Valid {target_col} rows: {len(df_valid)}")
            
            self.data = df_valid
            return df_valid
    
    def extract_features(self) -> np.ndarray:
        """Extract features"""
        if self.data is None:
            self.load_data()
        
        print(f"\nINFO Extracting {self.config.feature.feature_type} features...")
        
        # Start feature extraction timer
        with self.timing.measure('feature_extraction', {'type': self.config.feature.feature_type}):
            self._extract_features_internal()
        
        # Compute throughput
        if self.features is not None:
            self.timing.calculate_throughput('feature_extraction', len(self.features))
            
        return self.features
    
    def _extract_features_internal(self) -> np.ndarray:
        """Internal feature extraction implementation"""
        # Initialize feature extractor
        feature_extractor = FeatureExtractor(
            use_cache=self.config.feature.use_cache,
            cache_dir=self.config.feature.cache_dir,
            morgan_bits=self.config.feature.morgan_bits if hasattr(self.config.feature, 'morgan_bits') else None,
            morgan_radius=self.config.feature.morgan_radius if hasattr(self.config.feature, 'morgan_radius') else None,
            descriptor_count=getattr(self.config.feature, 'descriptor_count', 85)
        )
        
        # Check if molecular data (SMILES present)
        has_smiles = any(col in self.data.columns for col in self.config.data.smiles_columns)
        
        if has_smiles and self.config.feature.feature_type in ['morgan', 'descriptors', 'combined']:
            # Molecular feature extraction; prefer file-level cache and slice by subset indices
            features = None
            try:
                file_cache = FileFeatureCache(cache_dir='file_feature_cache')
                X_full = file_cache.load_features(
                    file_path=str(self.config.data.data_path),
                    feature_type=self.config.feature.feature_type,
                    morgan_bits=getattr(self.config.feature, 'morgan_bits', 1024),
                    morgan_radius=getattr(self.config.feature, 'morgan_radius', 2),
                    smiles_columns=self.config.data.smiles_columns,
                    combination_method=getattr(self.config.feature, 'combination_method', 'mean'),
                    descriptor_count=getattr(self.config.feature, 'descriptor_count', 85)
                )
                if X_full is not None:
                    # Use original index to select current target subset
                    subset_index = self.data.index.to_numpy()
                    features = X_full[subset_index]
                    print("   Training features loaded from file-level cache (sliced to current subset)")
            except Exception:
                features = None

            if features is None:
                # If no cache, compute once for entire training file and write cache
                try:
                    raw_df = getattr(self, 'raw_data', None)
                    if raw_df is None:
                        raw_df = pd.read_csv(self.config.data.data_path)
                        self.raw_data = raw_df

                    print("   File-level cache miss; extracting features for entire training file...")
                    feats_full = []
                    for _, row in tqdm(raw_df.iterrows(), total=len(raw_df), desc="Extract molecular features (full file)"):
                        smiles_list = []
                        for col in self.config.data.smiles_columns:
                            if col in row and pd.notna(row[col]):
                                smiles_list.append(row[col])
                            else:
                                smiles_list.append(None)
                        f = feature_extractor.extract_combination(
                            smiles_list,
                            feature_type=self.config.feature.feature_type,
                            combination_method=self.config.feature.combination_method
                        )
                        feats_full.append(f)
                    X_full = np.array(feats_full)

                    # Write cache for reuse by subsequent targets/stages
                    try:
                        file_cache.save_features(
                            features=X_full,
                            file_path=str(self.config.data.data_path),
                            feature_type=self.config.feature.feature_type,
                            morgan_bits=getattr(self.config.feature, 'morgan_bits', 1024),
                            morgan_radius=getattr(self.config.feature, 'morgan_radius', 2),
                            smiles_columns=self.config.data.smiles_columns,
                            combination_method=getattr(self.config.feature, 'combination_method', 'mean'),
                            descriptor_count=getattr(self.config.feature, 'descriptor_count', 85),
                            row_count=len(raw_df),
                            failed_indices=[]
                        )
                        print("   Cached training features (full file)")
                    except Exception:
                        pass

                    # Slice to current subset
                    subset_index = self.data.index.to_numpy()
                    features = X_full[subset_index]
                except Exception:
                    # Fallback to original row-wise extraction logic
                    features = []
                    for _, row in tqdm(self.data.iterrows(), total=len(self.data), desc="Extract molecular features"):
                        smiles_list = []
                        for col in self.config.data.smiles_columns:
                            if col in row and pd.notna(row[col]):
                                smiles_list.append(row[col])
                            else:
                                smiles_list.append(None)
                        feat = feature_extractor.extract_combination(
                            smiles_list,
                            feature_type=self.config.feature.feature_type,
                            combination_method=self.config.feature.combination_method
                        )
                        features.append(feat)
                    features = np.array(features)
        else:
            # Tabular feature extraction
            print("   Detected tabular data; using generic feature extraction...")
            
            # Get target columns to exclude
            target_cols = self.config.data.target_columns if hasattr(self.config.data, 'target_columns') else []
            
            # Use DataFrame extraction method
            if hasattr(feature_extractor, 'extract_from_dataframe'):
                features = feature_extractor.extract_from_dataframe(
                    self.data,
                    smiles_columns=self.config.data.smiles_columns if has_smiles else None,
                    target_columns=target_cols,
                    feature_type='tabular' if not has_smiles else 'auto'
                )
            else:
                # Fallback: use all non-target columns as features
                feature_cols = [col for col in self.data.columns if col not in target_cols]
                features = self.data[feature_cols].values
        
        print(f"   Feature shape: {features.shape}")
        
        # Handle NaN and Inf
        n_nan = np.isnan(features).sum()
        n_inf = np.isinf(features).sum()
        if n_nan > 0 or n_inf > 0:
            print(f"   Found {n_nan} NaN and {n_inf} Inf; processing...")
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        self.features = features
        return features
    
    def prepare_target(self, target_col: str) -> np.ndarray:
        """Prepare target variable"""
        if self.data is None:
            self.load_data()
        
        y = self.data[target_col].values
        
        # Unit conversion
        if target_col == 'PLQY' and y.max() > 1.5:
            print(f"   Convert PLQY: percent -> decimal")
            y = y / 100
        
        return y
    
    def train_single_target(self, target_col: str) -> Dict:
        """
        Train a single target
        
        Args:
            target_col: Target column name
        
        Returns:
            Training result
        """
        print(f"\n{'='*60}")
        print(f"Training target: {target_col}")
        print(f"{'='*60}")
        target_start = perf_counter()
        monitor = get_active_monitor()
        progress_file = os.environ.get("TRAIN_PROGRESS_FILE")
        
        # Load and filter data for this target
        self.load_data(target_col=target_col)
        
        # Extract features (per-target)
        self.features = None  # reset features
        self.extract_features()
        
        X = self.features
        y = self.prepare_target(target_col)
        
        print(f"   Samples: {len(X)}")
        print(f"   Features: {X.shape[1]}")
        print(f"   Target range: [{y.min():.2f}, {y.max():.2f}]")
        write_progress_event(progress_file, {
            "type": "training_context",
            "model": self.config.model.model_type,
            "target": target_col,
            "samples": len(X),
            "feature_count": int(X.shape[1]),
            "feature_type": self.config.feature.feature_type,
            "training_data_rows": len(getattr(self, "raw_data", self.data)),
        })
        
        # Optimization features removed; use default parameters
        
        # Create trainer
        trainer = ModelFactory.create_trainer(
            self.config.model.model_type,
            self.config.model.hyperparameters,
            self.config.training.n_folds
        )
        
        # Initialize logger
        if self.config.logging.auto_save:
            if self.logger is None:
                self.logger = TrainingLogger(
                    project_name=self.config.logging.project_name,
                    base_dir=self.config.logging.base_dir,
                    auto_save=self.config.logging.auto_save,
                    save_plots=self.config.logging.save_plots
                )
            
            # Start experiment
            experiment_id = self.logger.start_experiment(
                model_type=self.config.model.model_type,
                target=target_col,
                feature_type=self.config.feature.feature_type,
                hyperparameters=self.config.model.hyperparameters,
                n_folds=self.config.training.n_folds,
                n_samples=len(X),
                n_features=X.shape[1],
                config=self.config.to_dict()
            )
        
        # Perform cross-validation
        kf = KFold(
            n_splits=self.config.training.n_folds,
            shuffle=True,
            random_state=self.config.data.random_seed
        )
        
        all_predictions = np.zeros_like(y)
        fold_models = []
        fold_metrics = []
        
        # Initialize feature importance recorder (if enabled)
        feature_importance_recorder = None
        if self.config.training.save_feature_importance:
            from phosirdesign.utils import FeatureImportanceRecorder
            feature_importance_recorder = FeatureImportanceRecorder(
                save_dir=Path(self.config.logging.base_dir) / self.config.logging.project_name,
                model_name=self.config.model.model_type,
                target=target_col
            )
        
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X), 1):
            fold_start = perf_counter()
            # Log fold start
            if self.logger:
                self.logger.log_fold_start(fold_idx, train_idx.tolist(), val_idx.tolist())
            
            # Split data
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Create and train model
            from phosirdesign.models import BaseModel
            model = BaseModel(self.config.model.model_type, self.config.model.hyperparameters)
            early_rounds = self.config.training.early_stopping_rounds if self.config.training.early_stopping else None
            if self.config.training.verbose > 0:
                print(
                    f"    Model fit start: {self.config.model.model_type} "
                    f"{target_col} fold {fold_idx}/{self.config.training.n_folds}",
                    flush=True,
                )
            if monitor:
                monitor.fold_started(
                    self.config.model.model_type,
                    target_col,
                    fold_idx,
                    self.config.training.n_folds,
                )
            write_progress_event(progress_file, {
                "type": "fold_started",
                "model": self.config.model.model_type,
                "target": target_col,
                "fold": fold_idx,
                "total": self.config.training.n_folds,
            })
            
            # Record per-fold training time
            with self.timing.measure(f'fold_{fold_idx}_training', {'fold': fold_idx, 'samples': len(train_idx)}):
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    verbose=False,
                    early_stopping_rounds=early_rounds
                )
            
            # Predict
            with self.timing.measure(f'fold_{fold_idx}_prediction', {'fold': fold_idx, 'samples': len(val_idx)}):
                y_train_pred = model.predict(X_train)
                y_val_pred = model.predict(X_val)
                all_predictions[val_idx] = y_val_pred
            
            # Evaluate
            train_metrics = evaluate_model(y_train, y_train_pred)
            val_metrics = evaluate_model(y_val, y_val_pred)
            
            fold_metrics.append(val_metrics)
            fold_models.append(model)
            
            # Extract and record feature importance (if enabled and supported)
            if feature_importance_recorder:
                try:
                    # Try extracting feature importance from model
                    importance = FeatureImportanceRecorder.extract_importance_from_model(model.model)
                    if importance is not None:
                        # Generate feature names (if needed)
                        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
                        feature_importance_recorder.add_fold_importance(
                            fold_idx, importance, feature_names
                        )
                except Exception as e:
                    if self.config.training.verbose > 1:
                        print(f"    Warning: unable to extract feature importance: {e}")
            
            # Log fold end
            if self.logger:
                self.logger.log_fold_end(
                    y_train=y_train,
                    y_train_pred=y_train_pred,
                    y_val=y_val,
                    y_val_pred=y_val_pred,
                    metrics={**val_metrics, 'train_rmse': train_metrics['rmse'], 'train_r2': train_metrics['r2']}
                )
            
            # Show progress
            if self.config.training.verbose > 0:
                print(f"\n  Fold {fold_idx}/{self.config.training.n_folds}:")
                print(f"    Train - RMSE: {train_metrics['rmse']:.4f}, R^2: {train_metrics['r2']:.4f}")
                print(f"    Val   - RMSE: {val_metrics['rmse']:.4f}, R^2: {val_metrics['r2']:.4f}")
                print(f"    Fold elapsed: {perf_counter() - fold_start:.2f}s", flush=True)
        
        # Compute overall metrics
        print(
            f"\nAll CV folds complete for {self.config.model.model_type} / {target_col}; "
            "computing aggregate metrics...",
            flush=True,
        )
        if monitor:
            monitor.step(f"CV folds complete: {self.config.model.model_type} / {target_col}")
        write_progress_event(progress_file, {
            "type": "training_step",
            "message": f"CV folds complete: {self.config.model.model_type} / {target_col}",
        })
        final_metrics = evaluate_model(y, all_predictions)
        print(
            f"Aggregate metrics computed for {self.config.model.model_type} / {target_col}",
            flush=True,
        )
        
        # Compute average metrics
        avg_metrics = {}
        for metric in self.config.training.metrics:
            values = [fold[metric] for fold in fold_metrics if metric in fold]
            if values:
                avg_metrics[f"{metric}_mean"] = np.mean(values)
                avg_metrics[f"{metric}_std"] = np.std(values)
        
        print(f"\nCross-validation results:")
        for metric in self.config.training.metrics:
            if f"{metric}_mean" in avg_metrics:
                print(f"   {metric.upper()}: {avg_metrics[f'{metric}_mean']:.4f} +/- {avg_metrics[f'{metric}_std']:.4f}")
        
        # Save feature importance (if enabled)
        if feature_importance_recorder:
            print(
                f"Saving fold feature importance for {self.config.model.model_type} / {target_col}...",
                flush=True,
            )
            try:
                feature_importance_recorder.save_importance()
                print(
                    f"Fold feature importance saved for {self.config.model.model_type} / {target_col}",
                    flush=True,
                )
            except Exception as e:
                if self.config.training.verbose > 0:
                    print(f"   Warning: failed to save feature importance: {e}")
        
        # Train final model
        final_model = None
        if self.config.training.save_final_model:
            print(
                f"\nTraining final model (all data): {self.config.model.model_type} / {target_col} "
                f"with {len(X)} samples and {X.shape[1]} features...",
                flush=True,
            )
            if monitor:
                monitor.step(f"final model training: {self.config.model.model_type} / {target_col}")
            write_progress_event(progress_file, {
                "type": "training_step",
                "message": f"final model training: {self.config.model.model_type} / {target_col}",
            })
            final_model = BaseModel(self.config.model.model_type, self.config.model.hyperparameters)
            with self.timing.measure('final_model_training'):
                final_model.fit(X, y, verbose=False)
            print(
                f"Final model training complete: {self.config.model.model_type} / {target_col}",
                flush=True,
            )
            
            # Save model
            model_dir = Path(self.config.logging.base_dir) / self.config.logging.project_name / "models"
            model_dir.mkdir(parents=True, exist_ok=True)
            
            model_filename = generate_model_filename(
                self.config.model.model_type,
                target_col,
                "_final"
            )
            model_path = model_dir / model_filename
            final_model.save(model_path)
            print(f"   Model saved: {model_path}")
            
            # Save final model feature importance (if enabled)
            if self.config.training.save_feature_importance:
                print(
                    f"Saving final model feature importance for {self.config.model.model_type} / {target_col}...",
                    flush=True,
                )
                try:
                    from phosirdesign.utils import FeatureImportanceRecorder
                    importance = FeatureImportanceRecorder.extract_importance_from_model(final_model.model)
                    if importance is not None:
                        # Create a new recorder for the final model
                        final_importance_recorder = FeatureImportanceRecorder(
                            save_dir=Path(self.config.logging.base_dir) / self.config.logging.project_name,
                            model_name=f"{self.config.model.model_type}_final",
                            target=target_col
                        )
                        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
                        final_importance_recorder.add_fold_importance(0, importance, feature_names)
                        final_importance_recorder.save_importance()
                        print(
                            f"Final model feature importance saved for {self.config.model.model_type} / {target_col}",
                            flush=True,
                        )
                except Exception as e:
                    if self.config.training.verbose > 1:
                        print(f"   Warning: failed to save final model feature importance: {e}")

        # If test set provided, perform test evaluation (final model only)
        test_evaluation = None
        test_predictions = None
        if getattr(self.config.data, 'test_data_path', None):
            try:
                test_path = Path(self.config.data.test_data_path)
                print(f"\n" + "="*50)
                print("Test Evaluation")
                print("="*50)
                print(f"File: {test_path.name}")
                if test_path.exists():
                    print("Status: file exists")
                    print(f"Path: {test_path.resolve()}")
                    with self.timing.measure('data_load_test'):
                        df_test = pd.read_csv(test_path)
                    # Prepare test features: same as training flow
                    feature_extractor = FeatureExtractor(
                        use_cache=self.config.feature.use_cache,
                        cache_dir=self.config.feature.cache_dir,
                        morgan_bits=self.config.feature.morgan_bits if hasattr(self.config.feature, 'morgan_bits') else None,
                        morgan_radius=self.config.feature.morgan_radius if hasattr(self.config.feature, 'morgan_radius') else None,
                        descriptor_count=getattr(self.config.feature, 'descriptor_count', 85)
                    )
                    has_smiles = any(col in df_test.columns for col in self.config.data.smiles_columns)
                    if has_smiles and self.config.feature.feature_type in ['morgan', 'descriptors', 'combined']:
                        # Prefer file-level cache
                        X_test = None
                        try:
                            file_cache = FileFeatureCache(cache_dir='file_feature_cache')
                            X_test = file_cache.load_features(
                                file_path=str(test_path),
                                feature_type=self.config.feature.feature_type,
                                morgan_bits=getattr(self.config.feature, 'morgan_bits', 1024),
                                morgan_radius=getattr(self.config.feature, 'morgan_radius', 2),
                                smiles_columns=self.config.data.smiles_columns,
                                combination_method=getattr(self.config.feature, 'combination_method', 'mean'),
                                descriptor_count=getattr(self.config.feature, 'descriptor_count', 85)
                            )
                            if X_test is not None:
                                print("\nLoaded test features from file-level cache; skipping extraction")
                                print(f"   Shape: {X_test.shape}")
                                print("   Selecting model for inference and predicting")
                        except Exception as _e:
                            # On cache failure, silently fallback to normal extraction
                            X_test = None

                        if X_test is None:
                            feats = []
                            for _, row in tqdm(df_test.iterrows(), total=len(df_test), desc="Extract molecular features (test)"):
                                smiles_list = []
                                for col in self.config.data.smiles_columns:
                                    if col in row and pd.notna(row[col]):
                                        smiles_list.append(row[col])
                                    else:
                                        smiles_list.append(None)
                                with self.timing.measure('feature_extract_test_single'):
                                    f = feature_extractor.extract_combination(
                                        smiles_list,
                                        feature_type=self.config.feature.feature_type,
                                        combination_method=self.config.feature.combination_method
                                    )
                                feats.append(f)
                            X_test = np.array(feats)

                            # Write file-level cache for reuse by other targets
                            try:
                                file_cache.save_features(
                                    features=X_test,
                                    file_path=str(test_path),
                                    feature_type=self.config.feature.feature_type,
                                    morgan_bits=getattr(self.config.feature, 'morgan_bits', 1024),
                                    morgan_radius=getattr(self.config.feature, 'morgan_radius', 2),
                                    smiles_columns=self.config.data.smiles_columns,
                                    combination_method=getattr(self.config.feature, 'combination_method', 'mean'),
                                    descriptor_count=getattr(self.config.feature, 'descriptor_count', 85),
                                    row_count=len(df_test),
                                    failed_indices=[]
                                )
                                print("Cached test features; subsequent targets will reuse")
                            except Exception:
                                pass
                    else:
                        target_cols = self.config.data.target_columns if hasattr(self.config.data, 'target_columns') else []
                        with self.timing.measure('feature_extract_test_tabular'):
                            X_test = feature_extractor.extract_from_dataframe(
                                df_test,
                                smiles_columns=self.config.data.smiles_columns if has_smiles else None,
                                target_columns=target_cols,
                                feature_type='tabular' if not has_smiles else 'auto'
                            )
                    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

                    # Choose inference model (final model if saved; otherwise fold ensemble)
                    model_for_inference = final_model
                    if model_for_inference is None and len(fold_models) > 0:
                        print(f"   Using ensemble of fold models, count: {len(fold_models)}")
                        with self.timing.measure('test_predict_oof_ensemble'):
                            preds_list = []
                            for j, m in enumerate(fold_models, 1):
                                print(f"   Fold {j}/{len(fold_models)} prediction start")
                                p = m.predict(X_test)
                                preds_list.append(p)
                                print(f"   Fold {j} prediction complete")
                        test_predictions = np.mean(np.vstack(preds_list), axis=0)
                        print(f"   Ensemble prediction complete; output shape: {np.array(test_predictions).shape}")
                    else:
                        print("   Using final model for prediction")
                        with self.timing.measure('test_predict_final_model'):
                            test_predictions = model_for_inference.predict(X_test)
                        print(f"   Final model prediction complete; output shape: {np.array(test_predictions).shape}")

                    # If test set contains current target column, compute metrics
                    if target_col in df_test.columns:
                        y_test = df_test[target_col].values
                        if target_col == 'PLQY' and y_test.max() > 1.5:
                            y_test = y_test / 100
                        test_evaluation = evaluate_model(y_test, test_predictions)
                        
                        # Detailed test results output
                        print(f"\nTest results ({target_col}):")
                        print(f"   Samples: {len(y_test)}")
                        print(f"   - RMSE: {test_evaluation['rmse']:.4f}")
                        print(f"   - MAE:  {test_evaluation['mae']:.4f}")
                        print(f"   - R^2:  {test_evaluation['r2']:.4f}")
                        print(f"   - MAPE: {test_evaluation.get('mape', 0):.2f}%")

                    # Save test predictions
                    if self.logger:
                        exp_dir = Path(self.config.logging.base_dir) / self.config.logging.project_name
                        exports_dir = exp_dir / 'exports'
                        exports_dir.mkdir(parents=True, exist_ok=True)
                        out_csv = exports_dir / f"test_predictions_{self.config.model.model_type}_{target_col}.csv"
                        df_out = df_test.copy()
                        df_out['prediction'] = test_predictions
                        df_out.to_csv(out_csv, index=False)
                        
                        # Save test metrics (if available)
                        if test_evaluation is not None:
                            out_json = exports_dir / f"test_metrics_{self.config.model.model_type}_{target_col}.json"
                            import json as _json
                            with open(out_json, 'w') as f:
                                _json.dump(test_evaluation, f, indent=2, ensure_ascii=True)
                        
                        # Output save info
                        print(f"\nTest results saved:")
                        print(f"   Prediction file: {out_csv.name}")
                        if test_evaluation is not None:
                            print(f"   Metrics file: {out_json.name}")
                        print(f"   Output directory: {exports_dir}")
                        print("="*50)
                else:
                    print(f"   Test dataset path does not exist: {test_path}")
                    print(f"      Current working directory: {Path.cwd()}")
                    # Try alternative paths
                    alternative_paths = [
                        Path(test_path.name),  # current directory
                        Path("../data") / test_path.name,  # ../data directory
                        Path("data") / test_path.name,  # data directory
                    ]
                    for alt_path in alternative_paths:
                        if alt_path.exists():
                            print(f"      File may be at: {alt_path}")
            except Exception as e:
                print(f"   Test evaluation failed: {e}")
                import traceback
                if self.config.training.verbose > 1:
                    traceback.print_exc()
        
        # End experiment
        if self.logger:
            self.logger.end_experiment(final_metrics)
            try:
                timing_summary = self.timing.get_summary()
                for k, v in timing_summary.get('records', {}).items():
                    self.logger.add_timing(k, v.get('duration', 0))
            except Exception:
                pass
        
        # Print and save timing statistics
        if self.config.training.verbose > 0:
            print("\n" + "="*50)
            print("Timing Summary")
            print("="*50)
            self.timing.print_summary()
        
        # Save timing reports
        if self.logger:
            try:
                exp_dir = Path(self.config.logging.base_dir) / self.config.logging.project_name
                timing_dir = exp_dir / 'timing'
                timing_dir.mkdir(parents=True, exist_ok=True)
                
                # Save JSON format
                self.timing.save_report(
                    timing_dir / f"timing_{self.config.model.model_type}_{target_col}.json",
                    format='json'
                )
                
                # Save text format
                self.timing.save_report(
                    timing_dir / f"timing_{self.config.model.model_type}_{target_col}.txt",
                    format='txt'
                )
                
                if self.config.training.verbose > 0:
                    print(f"\nTiming reports saved to: {timing_dir}")
            except Exception as e:
                if self.config.training.verbose > 1:
                    print(f"Warning: failed to save timing reports: {e}")
            
            # Export publication-oriented data
            if self.config.logging.export_for_publication:
                self.logger.export_for_publication(experiment_id)
        progress_metrics = _build_progress_metrics(avg_metrics, final_metrics, fold_metrics)
        if monitor:
            monitor.target_completed(
                self.config.model.model_type,
                target_col,
                perf_counter() - target_start,
                metrics=progress_metrics,
            )
        write_progress_event(progress_file, {
            "type": "target_completed",
            "model": self.config.model.model_type,
            "target": target_col,
            "elapsed": perf_counter() - target_start,
            "metrics": progress_metrics,
        })
        
        return {
            'target': target_col,
            'final_metrics': final_metrics,
            'avg_metrics': avg_metrics,
            'fold_metrics': fold_metrics,
            'predictions': all_predictions,
            'true_values': y,
            'test_metrics': test_evaluation,
            'test_predictions_saved': self.config.data.test_data_path is not None
        }
    
    def train_all_targets(self) -> Dict:
        """Train all targets"""
        results = {}
        
        for target in self.available_targets:
            try:
                print(f"\nTraining target: {target}")
                result = self.train_single_target(target)
                results[target] = result
            except Exception as e:
                print(f"Training {target} failed: {e}")
                results[target] = {'error': str(e)}
        
        return results
    
    def run(self, targets: Optional[List[str]] = None) -> Dict:
        """
        Run the training pipeline
        
        Args:
            targets: List of targets to train; None means all
        
        Returns:
            Training results dictionary
        """
        print(f"\nINFO Start training: {self.config.name}")
        
        # Initial data load to check available targets
        self.load_data()
        
        # Determine targets
        if targets:
            targets_to_train = [t for t in targets if t in self.available_targets]
        else:
            targets_to_train = self.available_targets
        
        if not targets_to_train:
            raise ValueError("No trainable targets")
        
        print(f"\nWill train {len(targets_to_train)} targets: {targets_to_train}")
        
        # Train all targets
        results = {}
        monitor = get_active_monitor()
        progress_file = os.environ.get("TRAIN_PROGRESS_FILE")
        for target_idx, target in enumerate(targets_to_train, 1):
            try:
                print(f"\nTraining target: {target}")
                if monitor:
                    monitor.target_started(
                        self.config.model.model_type,
                        target,
                        target_idx,
                        len(targets_to_train),
                    )
                write_progress_event(progress_file, {
                    "type": "target_started",
                    "model": self.config.model.model_type,
                    "target": target,
                    "index": target_idx,
                    "total": len(targets_to_train),
                })
                result = self.train_single_target(target)
                results[target] = result
            except Exception as e:
                print(f"Training {target} failed: {e}")
                results[target] = {'error': str(e)}
        
        # Print summary
        self.print_summary(results)
        
        return results
    
    def print_summary(self, results: Dict):
        """Print training summary"""
        print("\n" + "="*60)
        print("Training Summary")
        print("="*60)
        
        for target, result in results.items():
            if 'error' in result:
                print(f"\nERROR {target}: failed - {result['error']}")
            else:
                print(f"\nOK {target}:")
                if 'final_metrics' in result:
                    metrics = result['final_metrics']
                    print(f"   RMSE: {metrics.get('rmse', 'N/A'):.4f}" if isinstance(metrics.get('rmse'), (int, float)) else f"   RMSE: N/A")
                    print(f"   MAE:  {metrics.get('mae', 'N/A'):.4f}" if isinstance(metrics.get('mae'), (int, float)) else f"   MAE: N/A")
                    print(f"   R^2:  {metrics.get('r2', 'N/A'):.4f}" if isinstance(metrics.get('r2'), (int, float)) else f"   R^2: N/A")
                if 'avg_metrics' in result:
                    avg = result['avg_metrics']
                    print(f"   CV average: RMSE={avg.get('rmse', 'N/A'):.4f}" if isinstance(avg.get('rmse'), (int, float)) else f"   CV average: N/A")


# ========================================
#           Batch Training Pipeline
# ========================================

class BatchTrainingPipeline:
    """Batch training pipeline"""
    
    def __init__(self, batch_config: BatchExperimentConfig):
        """
        Initialize batch training pipeline
        
        Args:
            batch_config: Batch experiment configuration
        """
        self.batch_config = batch_config
        self.results = {}
    
    def run(self) -> Dict:
        """Run batch training"""
        configs = self.batch_config.generate_configs()
        
        print(f"\nINFO Batch training: {len(configs)} experiments")
        print("="*60)
        
        for i, config in enumerate(configs, 1):
            print(f"\n[{i}/{len(configs)}] Experiment: {config.name}")
            
            try:
                pipeline = TrainingPipeline(config)
                result = pipeline.run()
                self.results[config.name] = {
                    'config': config,
                    'results': result,
                    'status': 'success'
                }
            except Exception as e:
                print(f"ERROR Experiment failed: {e}")
                self.results[config.name] = {
                    'config': config,
                    'error': str(e),
                    'status': 'failed'
                }
        
        # Aggregate results
        self.print_summary()
        
        return self.results
    
    def print_summary(self):
        """Print batch training summary"""
        print("\n" + "="*60)
        print("Batch Training Summary")
        print("="*60)
        
        success_count = sum(1 for r in self.results.values() if r['status'] == 'success')
        print(f"\nSuccess: {success_count}/{len(self.results)}")
        
        # Find best models
        best_models = {}
        for name, result in self.results.items():
            if result['status'] == 'success':
                for target, target_result in result['results'].items():
                    if 'final_metrics' in target_result:
                        key = f"{target}_rmse"
                        rmse = target_result['final_metrics']['rmse']
                        if key not in best_models or rmse < best_models[key]['value']:
                            best_models[key] = {
                                'experiment': name,
                                'value': rmse
                            }
        
        if best_models:
            print("\nBest models:")
            for key, info in best_models.items():
                target = key.replace('_rmse', '')
                print(f"   {target}: {info['experiment']} (RMSE: {info['value']:.4f})")


# ========================================
#           Command-Line Interface
# ========================================

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Config-driven machine learning training pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Config-related arguments
    parser.add_argument('config', nargs='?', help='Config file path or template name')
    parser.add_argument('--template', '-t', help='Use predefined template')
    parser.add_argument('--list-templates', action='store_true', help='List available templates')
    parser.add_argument('--wizard', action='store_true', help='Use interactive config wizard')
    parser.add_argument('--save-config', help='Save config to file')
    
    # Training-related arguments
    parser.add_argument('--target', help='Targets to train (comma-separated)')
    parser.add_argument('--dry-run', action='store_true', help='Validate config only; do not run training')
    parser.add_argument('--test-data', dest='test_data', help='Optional: CSV path for test set to evaluate after full training')
    
    # Overrides
    parser.add_argument('--model', help='Model type')
    parser.add_argument('--feature', help='Feature type')
    parser.add_argument('--folds', type=int, help='Number of CV folds')
    parser.add_argument('--project', help='Project name')
    
    args = parser.parse_args()
    
    # Config manager
    manager = ConfigManager()
    
    # List templates
    if args.list_templates:
        print("\nAvailable templates:")
        for template in manager.list_templates():
            desc = manager.templates[template].description
            print(f"  - {template}: {desc}")
        return
    
    # Config wizard
    if args.wizard:
        config = manager.create_from_wizard()
    
    # Load config
    elif args.config:
        # Try as template
        if args.config in manager.list_templates():
            config = manager.get_template(args.config)
        # As file path
        else:
            config = load_config(args.config)
    
    # Use template
    elif args.template:
        config = manager.get_template(args.template)
    
    # Default config
    else:
        print("Using default config (xgboost_quick)")
        config = manager.get_template('xgboost_quick')
    
    # Overrides
    if args.model:
        config.model.model_type = args.model
    if args.feature:
        config.feature.feature_type = args.feature
    if args.folds:
        config.training.n_folds = args.folds
    if args.project:
        config.logging.project_name = args.project
    # Test set argument
    if args.test_data:
        config.data.test_data_path = args.test_data
    
    # Save config
    if args.save_config:
        path = manager.save_config(config, args.save_config, 'yaml')
        print(f"Config saved: {path}")
    
    # Validate config
    if not ConfigValidator.validate_all(config):
        print("Configuration validation failed")
        return
    
    # Dry run
    if args.dry_run:
        print("\nConfig:")
        print(config.to_yaml())
        print("\nOK Configuration valid (dry-run mode)")
        return
    
    # Run training
    try:
        pipeline = TrainingPipeline(config)
        
        # Determine targets
        targets = None
        if args.target:
            targets = [t.strip() for t in args.target.split(',')]
        
        # Execute training
        results = pipeline.run(targets)
        
        print("\nTraining complete!")
        
        # Save final config
        if config.logging.auto_save:
            final_config_path = (
                Path(config.logging.base_dir) / 
                config.logging.project_name / 
                "experiment_config.yaml"
            )
            config.to_yaml(str(final_config_path))
            print(f"Experiment config saved: {final_config_path}")
        
    except Exception as e:
        print(f"\nERROR Training failed: {e}")
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
